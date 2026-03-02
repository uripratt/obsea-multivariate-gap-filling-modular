
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class VARMADataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int = 10):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.n_samples = max(0, len(data) - sequence_length)
        
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        # Input: sequence of length p (or slightly more for context if needed)
        # But standard VAR(p) predicts y_t from y_{t-1}...y_{t-p}
        # Ideally, we train on sub-sequences.
        # Here we return the window x and next value y
        x = self.data[idx:idx+self.sequence_length]
        y = self.data[idx+self.sequence_length]
        return x, y

class CudaVARMA(nn.Module):
    def __init__(self, n_features: int, p: int = 1, q: int = 0, hidden_dim: int = 64):
        super().__init__()
        self.p = p
        self.q = q
        self.n_features = n_features
        
        # AR part: Multi-Layer Perceptron (Non-Linear VAR)
        # Input size: p * n_features
        input_dim = n_features * p
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_features)
        )
        
        # MA part (Still simplified/optional)
        self.ma = None
        if q > 0:
            self.ma = nn.Linear(n_features * q, n_features, bias=False)
            
    def forward(self, x, epsilon=None):
        # x shape: (batch, p, features)
        batch_size = x.shape[0]
        
        # Flatten input
        x_flat = x.reshape(batch_size, -1)
        
        # Forward pass through MLP
        pred = self.net(x_flat)
        
        if self.q > 0 and epsilon is not None:
             eps_flat = epsilon.reshape(batch_size, -1)
             pred += self.ma(eps_flat)
             
        return pred

class VARMAImputer:
    def __init__(
        self,
        p: int = 1,
        q: int = 0,
        batch_size: int = 128,
        epochs: int = 100,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True
    ):
        self.p = p
        self.q = q
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.verbose = verbose
        
        self.model = None
        self.mean = None
        self.std = None
        
    def fit(self, df: pd.DataFrame, target_var: str, predictor_vars: List[str]):
        # Clear GPU memory before training to prevent CUDA errors
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Select columns
        cols = [target_var] + predictor_vars
        data = df[cols].values
        
        # Differencing for Stationarity:
        # Train on delta(t) = x(t) - x(t-1)
        # This removes linear trends and slow drift.
        data_diff = np.diff(data, axis=0)
        
        # Normalize Differences
        self.mean = np.nanmean(data_diff, axis=0)
        self.std = np.nanstd(data_diff, axis=0)
        data_norm = (data_diff - self.mean) / (self.std + 1e-8)
        
        # Handle NaNs in diffs (caused by NaNs in original data)
        # Simple interpolation for training stability
        df_imputed = pd.DataFrame(data_norm, columns=cols).interpolate(limit_direction='both')
        data_clean = df_imputed.values
        
        # Create dataset
        # We need sequence length = max(p, q)
        seq_len = max(self.p, self.q)
        
        # Bidirectional Training Augmentation:
        # Augment with reversed sequences so the model learns to predict "backwards" too
        # (i.e., given future [t+p...t+1] predict [t])
        # This assumes time-reversibility of the physics/stats, which is roughly true for filling.
        data_reversed = data_clean[::-1].copy()
        data_augmented = np.concatenate([data_clean, data_reversed], axis=0)
        
        dataset = VARMADataset(data_augmented, sequence_length=seq_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        try:
            self.model = CudaVARMA(n_features=len(cols), p=self.p, q=self.q).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            self.model.train()
            
            # Training loop with file logging
            for epoch in range(self.epochs):
                total_loss = 0
                for x_batch, y_batch in dataloader:
                    x_batch = x_batch.to(self.device) # (B, seq_len, F)
                    y_batch = y_batch.to(self.device) # (B, F)
                    
                    optimizer.zero_grad()
                    
                    # For now assume q=0 or handle MA later
                    pred = self.model(x_batch)
                    
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                
                # Log every 10 epochs for file tracking
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"      VARMA Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")
                    
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'cuda' in str(e):
                logger.warning(f"CUDA error detected in VARMA, falling back to CPU: {e}")
                # Reset CUDA and switch to CPU
                torch.cuda.empty_cache()
                self.device = torch.device('cpu')
                # Recursive call with CPU
                return self.fit(df, target_var, predictor_vars)
            raise

    def predict_series(self, df: pd.DataFrame, target_var: str, predictor_vars: List[str]) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not fitted")
            
        cols = [target_var] + predictor_vars
        data_orig = df[cols].values
        
        # We need to work with differences.
        # But for prediction, we need valid context x[t-p]...x[t].
        # So we first fill the original data linearly to get a baseline for context.
        df_filled_baseline = df[cols].interpolate(limit_direction='both') # Baseline fill for context
        data_vals = df_filled_baseline.values
        
        # Calculate differences for the WHOLE filled series to get context
        data_diff = np.diff(data_vals, axis=0)
        
        # Normalize
        data_norm = (data_diff - self.mean) / (self.std + 1e-8)
        
        seq_len = max(self.p, self.q)
        n_samples = len(df)
        nan_mask = df[cols].isna().values
        target_idx = 0 
        
        self.model.eval()
        
        # We will predict VALUES directly by integrating diffs
        # 1. Forward Pass
        # We maintain a running 'current value' for reconstruction
        pred_fwd = data_vals.copy() # Start with baseline
        
        # Convert context to tensor
        # We need to extract windows from `data_norm` (the diffs)
        # But as we predict new diffs, we must update `data_norm`!
        # So we maintain a mutable diff array.
        current_diffs_fwd = data_norm.copy()
        
        with torch.no_grad():
            for t in range(seq_len + 1, n_samples):
                if nan_mask[t, target_idx]:
                    # Need context of DIFFS: [t-1-p ... t-1]
                    # Indexing: data_norm[i] is diff between i and i+1. 
                    # ... Wait, diff[i] = x[i+1] - x[i].
                    # Let's standardize: diff[i] = x[i+1] - x[i].
                    # To predict x[t], we need x[t] - x[t-1] (which is diff[t-1]).
                    # Input context is diff[t-1-seq_len : t-1].
                    
                    diff_idx = t - 1
                    x_seq = current_diffs_fwd[diff_idx - seq_len : diff_idx]
                    
                    if len(x_seq) != seq_len: continue # Edge case
                    
                    x_tensor = torch.FloatTensor(x_seq).unsqueeze(0).to(self.device).view(1, seq_len, -1)
                    pred_diff_norm = self.model(x_tensor).cpu().numpy()[0]
                    
                    # Store predicted diff for future context
                    current_diffs_fwd[diff_idx, target_idx] = pred_diff_norm[target_idx]
                    
                    # Reconstruct Value
                    # x[t] = x[t-1] + predicted_diff
                    # predicted_diff = pred_norm * std + mean
                    pred_diff_real = pred_diff_norm[target_idx] * (self.std[target_idx] + 1e-8) + self.mean[target_idx]
                    
                    # Use the *previously predicted* value if it was a gap, or actual if available
                    prev_val = pred_fwd[t-1, target_idx] 
                    pred_fwd[t, target_idx] = prev_val + pred_diff_real


        # 2. Backward Pass
        pred_bwd = data_vals.copy()
        current_diffs_bwd = data_norm.copy()
        
        with torch.no_grad():
            for t in range(n_samples - 2, seq_len, -1): # Start from end
                if nan_mask[t, target_idx]:
                    # Goal: Predict x[t] given x[t+1] using backward logic.
                    # x[t] = x[t+1] - (x[t+1] - x[t])
                    # We need to predict the diff leading UP to t+1?
                    # Or just use the bidirectional training?
                    # We trained on reversed diff sequences.
                    # Forward: seq(diffs) -> next_diff.
                    # Reversed: seq(reversed_diffs) -> next_reversed_diff.
                    
                    # Future window of diffs: diff[t ... t+seq_len-1]
                    # Reverse it to match training distribution
                    diff_window = current_diffs_bwd[t : t + seq_len]
                    x_seq_reversed = diff_window[::-1].copy()
                    
                    x_tensor = torch.FloatTensor(x_seq_reversed).unsqueeze(0).to(self.device).view(1, seq_len, -1)
                    pred_diff_norm = self.model(x_tensor).cpu().numpy()[0]
                    
                    # This predicts the "next" diff in the reversed sequence.
                    # Which corresponds to diff[t-1] (the one just before the window in normal time)
                    # i.e. x[t] - x[t-1].
                    
                    # Wait, if we are at t, and valid future is t+1...
                    # We want to predict x[t].
                    # We know x[t+1].
                    # x[t+1] - x[t] = diff[t].
                    # So we need diff[t].
                    # The reversed model predicts the *next* step after the window.
                    # If window was diff[t+1]...diff[t+p], reverse -> predicts diff[t]. Correct.
                    # So we update diff[t].
                    
                    current_diffs_bwd[t, target_idx] = pred_diff_norm[target_idx]
                    
                    # Reconstruct Value
                    # x[t] = x[t+1] - predicted_diff
                    # But wait, predicted diff is diff[t] = x[t+1] - x[t].
                    # Correct.
                    
                    pred_diff_real = pred_diff_norm[target_idx] * (self.std[target_idx] + 1e-8) + self.mean[target_idx]
                    
                    next_val = pred_bwd[t+1, target_idx]
                    pred_bwd[t, target_idx] = next_val - pred_diff_real

        # 3. Merge
        final_vals = data_orig[:, target_idx].copy()
        # If original was valid, keep it.
        # Else merge preds.
        
        is_gap = nan_mask[:, target_idx]
        if np.any(is_gap):
            in_gap = False
            start_gap = 0
            
            for i in range(1, n_samples-1): # Safe margins
                if is_gap[i] and not in_gap:
                    in_gap = True
                    start_gap = i
                elif not is_gap[i] and in_gap:
                    in_gap = False
                    end_gap = i
                    length = end_gap - start_gap
                    
                    for idx in range(start_gap, end_gap):
                        fwd_val = pred_fwd[idx, target_idx]
                        bwd_val = pred_bwd[idx, target_idx]
                        
                        pos = idx - start_gap
                        w_bwd = (pos + 1) / (length + 1)
                        w_fwd = 1.0 - w_bwd
                        
                        final_vals[idx] = (w_fwd * fwd_val) + (w_bwd * bwd_val)

        return pd.Series(final_vals, index=df.index)

