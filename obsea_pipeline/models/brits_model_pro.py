"""
BRITS (Bidirectional Recurrent Imputation for Time Series) wrapper using PyPOTS (PRO Version).
Enhanced capacity, deeper RNN, larger scale training.

Pipeline B: NaN-preserving preprocessing.
"""

import numpy as np
import pandas as pd
import torch
from pypots.imputation import BRITS
import logging
from typing import List, Optional
from .stl_mixin import STLResidualMixin

logger = logging.getLogger(__name__)

class BRITSProImputer(STLResidualMixin):
    """
    PRO Wrapper for BRITS imputation model.
    Increases rnn_hidden_size, epochs, and window overlap.
    
    Pipeline B: NaN-preserving preprocessing.
    - Scales data using observed-only statistics
    - Preserves NaN positions through scaling and windowing
    """
    def __init__(self, n_steps: int, n_features: int, rnn_hidden_size: int = 512, epochs: int = 200, batch_size: int = 64, max_gap_size: Optional[int] = None):
        logger.info(f"Initializing BRITS PRO with hidden_size={rnn_hidden_size}, epochs={epochs}")
        self.n_steps = n_steps
        self.max_gap_size = max_gap_size
        
        # Dynamically adjust n_steps based on max_gap_size
        if self.max_gap_size is not None:
            # -------------------------------------------------------------
            # Adaptive Context Window (Context > Gap)
            # -------------------------------------------------------------
            min_context = self.max_gap_size * 3
            
            # Hardware/RNN unroll safety clamp: min 24 points, max 1024 points for RNNs
            self.n_steps = int(max(24, min(min_context, 1024)))
            
            if self.n_steps % 2 != 0:
                self.n_steps += 1
                
            logger.info(f"  [BRITS PRO] Adjusted adaptive n_steps to {self.n_steps} based on max_gap_size={self.max_gap_size}")

        self.model = BRITS(
            n_steps=self.n_steps,
            n_features=n_features,
            rnn_hidden_size=rnn_hidden_size,
            epochs=epochs,
            batch_size=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            saving_path="brits_pro_models"  
        )
        self.feature_columns = None
        
        # NaN-aware scaler parameters (computed during fit)
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def _fit_scaler(self, data: np.ndarray):
        """Fit scaler on observed (non-NaN) values only."""
        self.scaler_mean_ = np.nanmean(data, axis=0)
        self.scaler_std_ = np.nanstd(data, axis=0)
        self.scaler_std_[self.scaler_std_ == 0] = 1.0

    def _transform_preserving_nan(self, data: np.ndarray) -> np.ndarray:
        """Scale data while preserving NaN positions."""
        return (data - self.scaler_mean_) / self.scaler_std_

    def _inverse_transform(self, data_scaled: np.ndarray) -> np.ndarray:
        """Inverse scale."""
        return data_scaled * self.scaler_std_ + self.scaler_mean_

    def fit(self, df: pd.DataFrame, target_var: str, multivariate_vars: Optional[List[str]] = None):
        """Fit BRITS on time series data — NaN-preserving Pipeline B."""
        self.feature_columns = [target_var] + (multivariate_vars if multivariate_vars else [])
        self.target_var = target_var
        
        # EXTRACT RESIDUALS
        df_residuals = self.apply_stl_extraction(df.copy(), target_var)
        data = df_residuals[self.feature_columns].values.astype(np.float32)
        
        # NaN-aware scaling (fit on observed values only, preserve NaN)
        self._fit_scaler(data)
        data_scaled = self._transform_preserving_nan(data)
        
        # Create windows — NaN positions are preserved
        X = self._create_windows(data_scaled)
        
        nan_count = np.isnan(X).sum()
        total_count = X.size
        logger.info(f"  [BRITS PRO] Training windows: {X.shape}, NaN ratio: {nan_count/total_count:.2%}")
        
        dataset = {"X": X}
        self.model.fit(dataset)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Impute missing values — returns ABSOLUTE values."""
        data = df[self.feature_columns].values.astype(np.float32)
        data_scaled = self._transform_preserving_nan(data)
        X = self._create_windows(data_scaled)
        
        dataset = {"X": X}
        predictions = self.model.predict(dataset)
        
        imputed_scaled = self._reconstruct_from_windows(predictions["imputation"], len(df))
        imputed_residuals = self._inverse_transform(imputed_scaled)
        
        predicted_residuals = pd.Series(imputed_residuals[:, 0], index=df.index)
        
        # RECONSTRUCT ABSOLUTE
        predicted_absolute = self.apply_stl_reconstruction(df, predicted_residuals)
        
        return predicted_absolute

    def _create_windows(self, data):
        """Create overlapping windows for RNN — PRO uses 1/4 step for more overlap."""
        n_samples = len(data)
        windows = []
        starts = []
        step = self.n_steps // 4  # More overlap in PRO version
        
        if n_samples <= self.n_steps:
            if n_samples > 0:
                padded = np.pad(data, ((0, self.n_steps - n_samples), (0, 0)), mode='constant', constant_values=np.nan)
                windows.append(padded)
                starts.append(0)
        else:
            for i in range(0, n_samples - self.n_steps + 1, step):
                windows.append(data[i:i+self.n_steps])
                starts.append(i)
                
            last_start = starts[-1] if starts else 0
            expected_last = n_samples - self.n_steps
            if last_start < expected_last:
                windows.append(data[expected_last:])
                starts.append(expected_last)
                
        self._current_window_starts = starts
        return np.array(windows)

    def _reconstruct_from_windows(self, windows, original_len):
        """Reconstruct by averaging overlapping windows."""
        reconstructed = np.zeros((original_len, windows.shape[2]))
        counts = np.zeros((original_len, 1))
        
        for i, win in enumerate(windows):
            start = self._current_window_starts[i]
            end = start + self.n_steps
            actual_end = min(end, original_len)
            win_len = actual_end - start
            
            reconstructed[start:actual_end] += win[:win_len]
            counts[start:actual_end] += 1
            
        counts[counts == 0] = 1
        return reconstructed / counts
