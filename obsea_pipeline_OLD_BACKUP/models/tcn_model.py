"""
Temporal Convolutional Network (TCN) for gap filling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, List
from pathlib import Path
import logging
from obsea_pipeline.utils.hardware import get_device

logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)  # Causal: remove future
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(
        self,
        input_size: int = 1,
        num_channels: List[int] = [64, 128, 128, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super(TCNModel, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels,
                    kernel_size, dilation, dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Apply TCN
        y = self.network(x)
        
        # Take last timestep
        y = y[:, :, -1]
        
        # Output layer
        return self.fc(y)


class TCNImputer:
    """TCN-based imputation wrapper."""
    
    def __init__(
        self,
        num_channels: List[int] = [64, 128, 128, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        sequence_length: int = 96,
        batch_size: int = 64,
        epochs: int = 100,
        learning_rate: float = 0.002,
        device: str = get_device(),
    ):
        """Initialize TCN imputer."""
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
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
        """Train the TCN model."""
        logger.info(f"Training TCN on {target_var} using device: {self.device}")
        
        # Normalize
        data_normalized = self._normalize(df[target_var])
        
        # Import dataset from LSTM module (reuse)
        from .lstm_model import TimeSeriesDataset
        
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
        
        # Initialize model
        self.model = TCNModel(
            input_size=1,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
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
                
                # Forward
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        logger.info("Training complete")
        
        return self
    
    def predict(self, df: pd.DataFrame, target_var: str) -> pd.Series:
        """Predict missing values."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Normalize
        data_normalized = (df[target_var].values - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        result = df[target_var].copy()
        gap_mask = df[target_var].isna()
        
        self.model.eval()
        with torch.no_grad():
            for idx in range(self.sequence_length, len(df)):
                if gap_mask.iloc[idx]:
                    sequence = data_normalized[idx - self.sequence_length:idx]
                    
                    if not np.all(np.isnan(sequence)):
                        sequence_filled = pd.Series(sequence).fillna(method='ffill').fillna(method='bfill').values
                        
                        x = torch.FloatTensor(sequence_filled).unsqueeze(0).unsqueeze(-1).to(self.device)
                        pred = self.model(x).cpu().numpy()[0, 0]
                        
                        result.iloc[idx] = self._denormalize(pred)
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
                'num_channels': self.num_channels,
                'kernel_size': self.kernel_size,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length,
            }
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, path: str):
        """Load model."""
        checkpoint = torch.load(path)
        
        imputer = cls(**checkpoint['config'])
        imputer.model = TCNModel(
            input_size=1,
            **checkpoint['config']
        )
        imputer.model.load_state_dict(checkpoint['model_state'])
        imputer.scaler_mean = checkpoint['scaler_mean']
        imputer.scaler_std = checkpoint['scaler_std']
        
        logger.info(f"Model loaded from {path}")
        
        return imputer
