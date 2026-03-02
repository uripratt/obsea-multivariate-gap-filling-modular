"""
BRITS (Bidirectional Recurrent Imputation for Time Series) wrapper using PyPOTS.
"""

import numpy as np
import pandas as pd
import torch
from pypots.imputation import BRITS
import logging
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class BRITSImputer:
    """
    Wrapper for BRITS imputation model.
    """
    def __init__(self, n_steps: int, n_features: int, rnn_hidden_size: int = 128, epochs: int = 50, batch_size: int = 16):
        self.model = BRITS(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=rnn_hidden_size,
            epochs=epochs,
            batch_size=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            saving_path="brits_models"  # Provide explicit path to fix PyPOTS warning
        )
        self.n_steps = n_steps
        self.feature_columns = None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, target_var: str, multivariate_vars: Optional[List[str]] = None):
        """Fit BRITS on time series data."""
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
        """Helper to create windows for RNN."""
        n_samples = len(data)
        windows = []
        for i in range(0, n_samples - self.n_steps + 1, self.n_steps // 2):
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
