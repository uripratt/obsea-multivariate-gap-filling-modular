"""
ImputeFormer (Low Rankness-Induced Transformers) wrapper using PyPOTS.

Pipeline B: NaN-preserving preprocessing.
ImputeFormer is designed to receive NaN values natively and learn to impute them.
We must NOT pre-fill NaN before passing to the model.
"""

import numpy as np
import pandas as pd
import torch
from pypots.imputation import ImputeFormer
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ImputeFormerImputer:
    """
    Wrapper for ImputeFormer imputation model.
    
    Pipeline B: NaN-preserving preprocessing.
    - Scales data using observed-only statistics
    - Preserves NaN positions through scaling and windowing
    - PyPOTS internally generates observation masks from NaN positions
    """
    def __init__(self, target_var: str, predictor_vars: Optional[List[str]] = None, n_steps: int = 128, n_features: int = 1, n_layers: int = 2, d_input_embed: int = 128, d_ffn: int = 256, n_temporal_heads: int = 4, dropout: float = 0.1, epochs: int = 50, batch_size: int = 16, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', max_gap_size: Optional[int] = None):
        """
        Initialize ImputeFormer imputer.
        """
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_input_embed = d_input_embed
        self.d_ffn = d_ffn
        self.n_temporal_heads = n_temporal_heads
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.max_gap_size = max_gap_size

        self.feature_columns = None
        self.target_var = None # Will be set in fit

        # NaN-aware scaler parameters (computed during fit)
        self.scaler_mean_ = None
        self.scaler_std_ = None

        # -------------------------------------------------------------
        # SCALE-AWARE FRAMEWORK V4.0
        # Dynamic Sequence Length scaling based on max geometric gap size
        # -------------------------------------------------------------
        if self.max_gap_size is not None:
            original_steps = self.n_steps
            if self.max_gap_size <= 12: # Micro/Short (6 hours)
                self.n_steps = 12  # Fast, sharp context
            elif self.max_gap_size <= 144: # Medium (3 days)
                self.n_steps = 128  # Legal limit for VRAM
            else: # Long/Gigant
                self.n_steps = 128  # Hard cap to prevent OOM
            
            # Log this adjustment later in fit, as target_var is not available here
            # logger.info(f"Scale-Aware Triggered: Adjusted ImputeFormer n_steps from {original_steps} -> {self.n_steps} to match gap scale ({self.max_gap_size} pts).")

        self.model = ImputeFormer(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            d_input_embed=self.d_input_embed,
            d_learnable_embed=self.d_input_embed,
            d_proj=self.d_input_embed // 2,
            d_ffn=self.d_ffn,
            n_temporal_heads=self.n_temporal_heads,
            dropout=self.dropout,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            saving_path='imputeformer_models',
        )
        
        logger.info(f"Initialized ImputeFormer Imputer")
        logger.info(f"  n_steps: {self.n_steps}")
        logger.info(f"  Device: {self.device}")


    def _fit_scaler(self, data: np.ndarray):
        """Fit scaler on observed (non-NaN) values only."""
        self.scaler_mean_ = np.nanmean(data, axis=0)
        self.scaler_std_ = np.nanstd(data, axis=0)
        self.scaler_std_[self.scaler_std_ == 0] = 1.0

    def _transform_preserving_nan(self, data: np.ndarray) -> np.ndarray:
        """Scale data while preserving NaN positions."""
        # NaN - number = NaN, so NaN positions are automatically preserved
        return (data - self.scaler_mean_) / self.scaler_std_

    def _inverse_transform(self, data_scaled: np.ndarray) -> np.ndarray:
        """Inverse scale."""
        return data_scaled * self.scaler_std_ + self.scaler_mean_

    def fit(self, df: pd.DataFrame, target_var: str, multivariate_vars: Optional[List[str]] = None):
        """Fit ImputeFormer on time series data — NaN-preserving Pipeline B."""
        self.feature_columns = [target_var] + (multivariate_vars if multivariate_vars else [])
        self.target_var = target_var
        
        # Log dynamic n_steps adjustment if it happened
        if self.max_gap_size is not None:
            # Re-log with target_var now available
            # This assumes original_steps was stored in __init__ if needed,
            # but for simplicity, we'll just log the final n_steps here.
            logger.info(f"Scale-Aware Triggered: ImputeFormer n_steps for {target_var} set to {self.n_steps} to match gap scale ({self.max_gap_size} pts).")
        
        data = df[self.feature_columns].values.astype(np.float64)
        
        # NaN-aware scaling
        self._fit_scaler(data)
        data_scaled = self._transform_preserving_nan(data)
        
        # Create windows — NaN positions are preserved
        X = self._create_windows(data_scaled)
        
        nan_count = np.isnan(X).sum()
        total_count = X.size
        logger.info(f"  [ImputeFormer] Training windows: {X.shape}, NaN ratio: {nan_count/total_count:.2%}")
        
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
        """Create overlapping windows."""
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
