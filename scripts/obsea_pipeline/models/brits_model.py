"""
BRITS (Bidirectional Recurrent Imputation for Time Series) wrapper using PyPOTS.

Pipeline B: NaN-preserving preprocessing.
BRITS uses bidirectional RNN with temporal decay — designed natively for NaN.
We must NOT pre-fill NaN before passing to the model.
"""

import numpy as np
import pandas as pd
import torch
from pypots.imputation import BRITS
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class BRITSImputer:
    """
    Wrapper for BRITS imputation model.
    
    Pipeline B: NaN-preserving preprocessing.
    - Scales data using observed-only statistics
    - Preserves NaN positions through scaling and windowing
    - PyPOTS internally generates observation masks from NaN positions
    """
    def __init__(self, n_steps: int, n_features: int, rnn_hidden_size: int = 128, epochs: int = 50, batch_size: int = 16, max_gap_size: Optional[int] = None):
        """
        Initialize BRITS imputer.
        """
        self.n_steps = n_steps
        self.max_gap_size = max_gap_size

        # Dynamically adjust n_steps based on max_gap_size
        if self.max_gap_size is not None:
            if self.max_gap_size <= 12: # Micro/Short gaps
                self.n_steps = max(self.n_steps, 24) # Ensure at least 2x gap size, or a reasonable minimum
            elif self.max_gap_size > 12: # Long/Gigant gaps
                self.n_steps = min(max(self.n_steps, self.max_gap_size * 2), 128) # Up to 128 max, at least 2x gap size
            logger.info(f"  [BRITS] Adjusted n_steps to {self.n_steps} based on max_gap_size={self.max_gap_size}")

        self.model = BRITS(
            n_steps=self.n_steps,
            n_features=n_features,
            rnn_hidden_size=rnn_hidden_size,
            epochs=epochs,
            batch_size=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            saving_path="brits_models"
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
        
        data = df[self.feature_columns].values.astype(np.float64)
        
        # NaN-aware scaling
        self._fit_scaler(data)
        data_scaled = self._transform_preserving_nan(data)
        
        # Create windows — NaN positions are preserved
        X = self._create_windows(data_scaled)
        
        nan_count = np.isnan(X).sum()
        total_count = X.size
        logger.info(f"  [BRITS] Training windows: {X.shape}, NaN ratio: {nan_count/total_count:.2%}")
        
        dataset = {"X": X}
        self.model.fit(dataset)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Impute missing values — returns ABSOLUTE values."""
        data = df[self.feature_columns].values.astype(np.float64)
        data_scaled = self._transform_preserving_nan(data)
        X = self._create_windows(data_scaled)
        
        dataset = {"X": X}
        predictions = self.model.predict(dataset)
        
        imputed_scaled = self._reconstruct_from_windows(predictions["imputation"], len(df))
        imputed = self._inverse_transform(imputed_scaled)
        
        return pd.Series(imputed[:, 0], index=df.index)

    def _create_windows(self, data):
        """Create overlapping windows for RNN input."""
        n_samples = len(data)
        windows = []
        step = self.n_steps // 2
        for i in range(0, n_samples - self.n_steps + 1, step):
            windows.append(data[i:i+self.n_steps])
        return np.array(windows)

    def _reconstruct_from_windows(self, windows, original_len):
        """Reconstruct by averaging overlapping windows."""
        reconstructed = np.zeros((original_len, windows.shape[2]))
        counts = np.zeros((original_len, 1))
        
        step = self.n_steps // 2
        for i, win in enumerate(windows):
            start = i * step
            end = start + self.n_steps
            if end > original_len: break
            
            reconstructed[start:end] += win
            counts[start:end] += 1
            
        counts[counts == 0] = 1
        return reconstructed / counts
