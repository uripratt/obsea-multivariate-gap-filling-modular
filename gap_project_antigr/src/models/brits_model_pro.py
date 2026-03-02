"""
BRITS (Bidirectional Recurrent Imputation for Time Series) wrapper using PyPOTS (PRO Version).
Enhanced capacity, deeper RNN, larger scale training.
"""

import numpy as np
import pandas as pd
import torch
from pypots.imputation import BRITS
import logging
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class BRITSProImputer:
    """
    PRO Wrapper for BRITS imputation model.
    Increases rnn_hidden_size, epochs, learning_rate variations.
    """
    def __init__(self, n_steps: int, n_features: int, rnn_hidden_size: int = 512, epochs: int = 200, batch_size: int = 64):
        logger.info(f"Initializing BRITS PRO with hidden_size={rnn_hidden_size}, epochs={epochs}")
        self.model = BRITS(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=rnn_hidden_size,
            epochs=epochs,
            batch_size=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            saving_path="brits_pro_models"  
        )
        self.n_steps = n_steps
        self.feature_columns = None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, target_var: str, multivariate_vars: Optional[List[str]] = None):
        """Fit BRITS on time series data."""
        self.feature_columns = [target_var] + (multivariate_vars if multivariate_vars else [])
        data = df[self.feature_columns].copy()
        
        # We assume standardization is handled outside, but usually neural nets need it.
        # Ensure we have double/float32
        data = data.astype(np.float32).values
        
        # Scale data for Deep Learning model
        data_scaled = self.scaler.fit_transform(data)
        
        # Create windows
        X = self._create_windows(data_scaled)
        
        dataset = {"X": X}
        self.model.fit(dataset)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Impute missing values."""
        data = df[self.feature_columns].astype(np.float32).values
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
        for i in range(0, n_samples - self.n_steps + 1, self.n_steps // 4): # More overlap in PRO version
            windows.append(data[i:i+self.n_steps])
        
        # If no windows were created (very short sequence)
        if not windows and n_samples > 0:
            padded = np.pad(data, ((0, self.n_steps - n_samples), (0, 0)), mode='constant', constant_values=np.nan)
            windows.append(padded)
            
        return np.array(windows)

    def _reconstruct_from_windows(self, windows, original_len):
        """Reconstruct by averaging overlapping windows."""
        reconstructed = np.zeros((original_len, windows.shape[2]))
        counts = np.zeros((original_len, 1))
        
        step = self.n_steps // 4 # Matching the overlap from create_windows
        for i, win in enumerate(windows):
            start = i * step
            end = start + self.n_steps
            # Limit end if we reached original_len (for the last window if padding was used)
            actual_end = min(end, original_len)
            win_len = actual_end - start
            
            reconstructed[start:actual_end] += win[:win_len]
            counts[start:actual_end] += 1
            
        counts[counts == 0] = 1
        return reconstructed / counts
