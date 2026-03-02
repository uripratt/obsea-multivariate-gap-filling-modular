"""
ImputeFormer (Low Rankness-Induced Transformers) wrapper using PyPOTS.
"""

import numpy as np
import pandas as pd
import torch
from pypots.imputation import ImputeFormer
import logging
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ImputeFormerImputer:
    """
    Wrapper for ImputeFormer imputation model.
    """
    def __init__(self, n_steps: int, n_features: int, n_layers: int = 2, d_input_embed: int = 128, d_ffn: int = 256, n_temporal_heads: int = 4, dropout: float = 0.1, epochs: int = 50, batch_size: int = 16):
        self.model = ImputeFormer(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=n_layers,
            d_input_embed=d_input_embed,
            d_learnable_embed=d_input_embed,
            d_proj=d_input_embed // 2,
            d_ffn=d_ffn,
            n_temporal_heads=n_temporal_heads,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            saving_path='imputeformer_models',
        )
        self.n_steps = n_steps
        self.feature_columns = None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, target_var: str, multivariate_vars: Optional[List[str]] = None):
        """Fit ImputeFormer on time series data."""
        self.feature_columns = [target_var] + (multivariate_vars if multivariate_vars else [])
        data = df[self.feature_columns].values
        
        # Scale data for Deep Learning model
        data_scaled = self.scaler.fit_transform(data)
        
        # Create windows
        X = self._create_windows(data_scaled)
        
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
        
        imputed_values_scaled = self._reconstruct_from_windows(predictions["imputation"], len(df))
        imputed_values = self.scaler.inverse_transform(imputed_values_scaled)
        
        return pd.Series(imputed_values[:, 0], index=df.index)

    def _create_windows(self, data):
        """Helper to create windows."""
        n_samples = len(data)
        windows = []
        for i in range(0, n_samples - self.n_steps + 1, self.n_steps // 2):
            windows.append(data[i:i+self.n_steps])
        return np.array(windows)

    def _reconstruct_from_windows(self, windows, original_len):
        """Reconstruct the original series by averaging overlapping windows."""
        reconstructed = np.zeros((original_len, windows.shape[2]))
        counts = np.zeros((original_len, 1))
        
        step = self.n_steps // 2
        for i, win in enumerate(windows):
            start = i * step
            end = start + self.n_steps
            if end > original_len: break
            
            reconstructed[start:end] += win
            counts[start:end] += 1
            
        # Avoid division by zero
        counts[counts == 0] = 1
        return reconstructed / counts
