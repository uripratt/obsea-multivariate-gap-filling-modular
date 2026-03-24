"""
LSTM-based gap filling model for time series.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time series with gaps."""
    
    def __init__(
        self,
        data: pd.Series,
        sequence_length: int = 48,
        target_mask: Optional[np.ndarray] = None,
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        data : pd.Series
            Time series data
        sequence_length : int
            Length of input sequences
        target_mask : np.ndarray, optional
            Mask indicating which positions to predict
        """
        self.data = data.values
        self.sequence_length = sequence_length
        self.target_mask = target_mask
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx:idx + self.sequence_length]
        
        # Target is next value
        y = self.data[idx + self.sequence_length]
        
        # Convert to tensors
        x = torch.FloatTensor(x).unsqueeze(-1)  # Add feature dimension
        y = torch.FloatTensor([y])
        
        # Create mask for missing values
        mask = ~np.isnan(x.numpy())
        
        # Replace NaN with 0 (will be masked)
        x = torch.nan_to_num(x, 0.0)
        
        return x, y, torch.FloatTensor(mask)


class LSTMModel(nn.Module):
    """LSTM model for time series imputation."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        """
        Initialize LSTM model.
        
        Parameters
        ----------
        input_size : int
            Number of input features
        hidden_size : int
            Hidden layer size
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability
        bidirectional : bool
            Use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, 1)
        
    def forward(self, x, mask=None):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        if self.bidirectional:
            # Concatenate forward and backward
            output = lstm_out[:, -1, :]
        else:
            output = lstm_out[:, -1, :]
        
        # Final prediction
        out = self.fc(output)
        
        return out


class LSTMImputer:
    """LSTM-based imputation wrapper."""
    
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        sequence_length: int = 48,
        batch_size: int = 64,
        epochs: int = 100,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """Initialize LSTM imputer."""
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        
    def _normalize(self, data: pd.Series) -> np.ndarray:
        """Normalize data."""
        values = data.values
        valid_mask = ~np.isnan(values)
        
        self.scaler_mean = np.nanmean(values[valid_mask])
        self.scaler_std = np.nanstd(values[valid_mask])
        
        normalized = (values - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        return normalized
    
    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data."""
        return data * self.scaler_std + self.scaler_mean
    
    def fit(
        self,
        df: pd.DataFrame,
        target_var: str,
        validation_data: Optional[pd.DataFrame] = None,
    ):
        """
        Train the LSTM model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        target_var : str
            Variable to predict
        validation_data : pd.DataFrame, optional
            Validation data
            
        Returns
        -------
        self
        """
        logger.info(f"Training LSTM on {target_var} using device: {self.device}")
        
        # Clear GPU memory before training to prevent CUDA errors
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Normalize data
        data_normalized = self._normalize(df[target_var])
        
        # Create dataset
        dataset = TimeSeriesDataset(
            pd.Series(data_normalized, index=df.index),
            sequence_length=self.sequence_length,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        try:
            # Initialize model
            self.model = LSTMModel(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                num_batches = 0
                
                for x_batch, y_batch, mask_batch in dataloader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(x_batch)
                    loss = criterion(outputs, y_batch)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                
                # Log every 5 epochs for better file tracking
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f"      Bi-LSTM Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
            
            logger.info("Training complete")
            
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'cuda' in str(e):
                logger.warning(f"CUDA error detected, falling back to CPU: {e}")
                # Reset CUDA and switch to CPU
                torch.cuda.empty_cache()
                self.device = torch.device('cpu')
                # Recursive call with CPU
                return self.fit(df, target_var, validation_data)
            raise
        
        return self
    
    def predict(self, df: pd.DataFrame, target_var: str) -> pd.Series:
        """
        Predict missing values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with gaps
        target_var : str
            Variable to impute
            
        Returns
        -------
        pd.Series
            Imputed series
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Normalize
        data_normalized = (df[target_var].values - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        # For each gap, predict using surrounding context
        result = df[target_var].copy()
        gap_mask = df[target_var].isna()
        
        self.model.eval()
        with torch.no_grad():
            for idx in range(self.sequence_length, len(df)):
                if gap_mask.iloc[idx]:
                    # Get sequence
                    sequence = data_normalized[idx - self.sequence_length:idx]
                    
                    # Check if we have enough valid data
                    if not np.all(np.isnan(sequence)):
                        # Replace NaNs with forward fill for prediction
                        sequence_filled = pd.Series(sequence).fillna(method='ffill').fillna(method='bfill').values
                        
                        # Predict
                        x = torch.FloatTensor(sequence_filled).unsqueeze(0).unsqueeze(-1).to(self.device)
                        pred = self.model(x).cpu().numpy()[0, 0]
                        
                        # Denormalize and store
                        result.iloc[idx] = self._denormalize(pred)
                        
                        # Update normalized data for next predictions
                        data_normalized[idx] = pred
        
        return result
    
    def save(self, path: str):
        """Save model."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'config': {
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'sequence_length': self.sequence_length,
            }
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, path: str):
        """Load model."""
        checkpoint = torch.load(path)
        
        imputer = cls(**checkpoint['config'])
        imputer.model = LSTMModel(
            input_size=1,
            **checkpoint['config']
        )
        imputer.model.load_state_dict(checkpoint['model_state'])
        imputer.scaler_mean = checkpoint['scaler_mean']
        imputer.scaler_std = checkpoint['scaler_std']
        
        logger.info(f"Model loaded from {path}")
        
        return imputer
