"""
SAITS (Self-Attention-based Imputation for Time Series) wrapper using PyPOTS.

Pipeline B: NaN-preserving preprocessing.
SAITS is designed to receive NaN values natively and learn to impute them
using self-attention masks. We must NOT pre-fill NaN before passing to the model.
"""

import numpy as np
import pandas as pd
import torch
from pypots.imputation import SAITS
try:
    from pypots.nn.functional import calc_mae
except ImportError:
    from pypots.utils.metrics import calc_mae
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class SAITSImputer:
    """
    Wrapper for SAITS imputation model.
    
    Pipeline B: NaN-preserving preprocessing.
    - Scales data using observed-only statistics
    - Preserves NaN positions through scaling and windowing
    - PyPOTS internally generates observation masks from NaN positions
    """
    def __init__(self, n_steps: int, n_features: int, n_layers: int = 2, d_model: int = 128, d_ffn: int = 256, n_heads: int = 4, dropout: float = 0.1, epochs: int = 50, batch_size: int = 16):
        self.model = SAITS(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_model // n_heads,
            d_v=d_model // n_heads,
            d_ffn=d_ffn,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            saving_path='saits_models',
        )
        self.n_steps = n_steps
        self.feature_columns = None
        
        # NaN-aware scaler parameters (computed during fit)
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def _fit_scaler(self, data: np.ndarray):
        """Fit scaler on observed (non-NaN) values only."""
        self.scaler_mean_ = np.nanmean(data, axis=0)
        self.scaler_std_ = np.nanstd(data, axis=0)
        self.scaler_std_[self.scaler_std_ == 0] = 1.0
        logger.info(f"  [SAITS] Scaler fit on observed values. Mean: {self.scaler_mean_[:3]}, Std: {self.scaler_std_[:3]}")

    def _transform_preserving_nan(self, data: np.ndarray) -> np.ndarray:
        """Scale data while preserving NaN positions."""
        data_scaled = (data - self.scaler_mean_) / self.scaler_std_
        # NaN - number = NaN, so NaN positions are automatically preserved
        return data_scaled

    def _inverse_transform(self, data_scaled: np.ndarray) -> np.ndarray:
        """Inverse scale."""
        return data_scaled * self.scaler_std_ + self.scaler_mean_

    def fit(self, df: pd.DataFrame, target_var: str, multivariate_vars: Optional[List[str]] = None):
        """Fit SAITS on time series data — NaN-preserving Pipeline B."""
        self.feature_columns = [target_var] + (multivariate_vars if multivariate_vars else [])
        self.target_var = target_var
        
        data = df[self.feature_columns].values.astype(np.float64)
        
        # NaN-aware scaling: fit on observed values only, preserve NaN
        self._fit_scaler(data)
        data_scaled = self._transform_preserving_nan(data)
        
        # Create windows — NaN positions are preserved inside windows
        X = self._create_windows(data_scaled)
        
        nan_count = np.isnan(X).sum()
        total_count = X.size
        logger.info(f"  [SAITS] Training windows: {X.shape}, NaN ratio: {nan_count/total_count:.2%}")
        
        # PyPOTS internally creates observation masks from NaN positions
        dataset = {"X": X}
        self.model.fit(dataset)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Impute missing values — returns ABSOLUTE values (not residuals)."""
        data = df[self.feature_columns].values.astype(np.float64)
        data_scaled = self._transform_preserving_nan(data)
        X = self._create_windows(data_scaled)
        
        dataset = {"X": X}
        predictions = self.model.predict(dataset)
        
        # Reconstruct full series from overlapping windows
        imputed_scaled = self._reconstruct_from_windows(predictions["imputation"], len(df))
        imputed = self._inverse_transform(imputed_scaled)
        
        # Return target variable as absolute values
        return pd.Series(imputed[:, 0], index=df.index)

    def _create_windows(self, data):
        """Create overlapping windows for Transformer input."""
        n_samples = len(data)
        windows = []
        step = self.n_steps // 2
        for i in range(0, n_samples - self.n_steps + 1, step):
            windows.append(data[i:i+self.n_steps])
        return np.array(windows)

    def _reconstruct_from_windows(self, windows, original_len):
        """Reconstruct by averaging overlapping windows."""
        reconstructed = np.full((original_len, windows.shape[2]), np.nan)
        counts = np.zeros((original_len, 1))
        
        step = self.n_steps // 2
        for i, win in enumerate(windows):
            start = i * step
            end = start + self.n_steps
            if end > original_len: break
            
            # For first assignment, just copy
            if np.isnan(reconstructed[start:end]).all():
                reconstructed[start:end] = win
            else:
                # Average with existing values
                reconstructed[start:end] = (reconstructed[start:end] * counts[start:end] + win) / (counts[start:end] + 1)
            counts[start:end] += 1
            
        # Fill any tail gaps with interpolation
        return pd.DataFrame(reconstructed).interpolate().bfill().ffill().values
