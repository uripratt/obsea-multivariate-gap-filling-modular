"""
SAITS (Self-Attention-based Imputation for Time Series) wrapper using PyPOTS.
"""

import numpy as np
import pandas as pd
import torch
from pypots.imputation import SAITS
# pypots.utils.metrics is deprecated - import from pypots.nn.functional instead
try:
    from pypots.nn.functional import calc_mae
except ImportError:
    from pypots.utils.metrics import calc_mae
import logging
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class SAITSImputer:
    """
    Wrapper for SAITS imputation model.
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
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, target_var: str, multivariate_vars: Optional[List[str]] = None):
        """Fit SAITS on time series data."""
        self.feature_columns = [target_var] + (multivariate_vars if multivariate_vars else [])
        data = df[self.feature_columns].values
        
        # Scale data for Deep Learning model
        data_scaled = self.scaler.fit_transform(data)
        
        # SAITS expects (n_samples, n_steps, n_features)
        # We need to create sliding windows for training
        X = self._create_windows(data_scaled)
        
        # PyPOTS models use a dictionary for training
        dataset = {"X": X}
        self.model.fit(dataset)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Impute missing values."""
        data = df[self.feature_columns].values
        data_scaled = self.scaler.transform(data)
        X = self._create_windows(data_scaled)
        
        dataset = {"X": X}
        predictions = self.model.predict(dataset)
        
        # Flatten predictions back to original series
        # This is a simplification: we take the average of overlapping windows or the middle point
        # For production, we'll map back carefully.
        imputed_values_scaled = self._reconstruct_from_windows(predictions["imputation"], len(df))
        imputed_values = self.scaler.inverse_transform(imputed_values_scaled)
        
        return pd.Series(imputed_values[:, 0], index=df.index)

    def _create_windows(self, data):
        """Helper to create windows for Transformer."""
        # Simple non-overlapping or sliding windows
        n_samples = len(data)
        windows = []
        for i in range(0, n_samples - self.n_steps + 1, self.n_steps // 2):
            windows.append(data[i:i+self.n_steps])
        return np.array(windows)

    def _reconstruct_from_windows(self, windows, original_len):
        """Simple reconstruction (last window wins or averaging)."""
        reconstructed = np.full((original_len, windows.shape[2]), np.nan)
        counts = np.zeros((original_len, 1))
        
        step = self.n_steps // 2
        for i, win in enumerate(windows):
            start = i * step
            end = start + self.n_steps
            if end > original_len: break
            
            # Simple average blending
            if np.isnan(reconstructed[start:end]).all():
                reconstructed[start:end] = win
            else:
                reconstructed[start:end] = (reconstructed[start:end] * counts[start:end] + win) / (counts[start:end] + 1)
            counts[start:end] += 1
            
        # Fill any remaining NaNs with original data if possible or interpolation
        return pd.DataFrame(reconstructed).interpolate().bfill().ffill().values
