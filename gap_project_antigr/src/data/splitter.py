"""
Temporal train/validation/test splitting for time series.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series into train, validation, and test sets.
    
    Uses temporal ordering to prevent data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index
    train_ratio : float
        Fraction of data for training
    val_ratio : float
        Fraction of data for validation
    test_ratio : float
        Fraction of data for testing
        
    Returns
    ----------
    tuple of pd.DataFrame
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Split data into:")
    logger.info(f"  Train: {len(train_df)} records ({train_df.index[0]} to {train_df.index[-1]})")
    logger.info(f"  Val:   {len(val_df)} records ({val_df.index[0]} to {val_df.index[-1]})")
    logger.info(f"  Test:  {len(test_df)} records ({test_df.index[0]} to {test_df.index[-1]})")
    
    return train_df, val_df, test_df


def temporal_train_val_test_split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series by specific dates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index
    train_end : str
        End date for training set (inclusive)
    val_end : str
        End date for validation set (inclusive)
        
    Returns
    -------
    tuple of pd.DataFrame
        (train_df, val_df, test_df)
    """
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    train_df = df[df.index <= train_end_dt].copy()
    val_df = df[(df.index > train_end_dt) & (df.index <= val_end_dt)].copy()
    test_df = df[df.index > val_end_dt].copy()
    
    logger.info(f"Split data by dates:")
    logger.info(f"  Train: {len(train_df)} records (up to {train_end})")
    logger.info(f"  Val:   {len(val_df)} records ({train_end} to {val_end})")
    logger.info(f"  Test:  {len(test_df)} records (after {val_end})")
    
    return train_df, val_df, test_df


class TimeSeriesSplitter:
    """Time series cross-validation splitter."""
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size_days: int = 30,
        gap_days: int = 0,
    ):
        """
        Initialize splitter.
        
        Parameters
        ----------
        n_splits : int
            Number of splits
        test_size_days : int
            Size of test set in days
        gap_days : int
            Gap between train and test sets (to avoid leakage)
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days
    
    def split(self, df: pd.DataFrame):
        """
        Generate train/test indices for time series CV.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with datetime index
            
        Yields
        ------
        tuple of arrays
            (train_indices, test_indices)
        """
        n = len(df)
        test_size = int(self.test_size_days * 48)  # 48 records per day (30min freq)
        gap_size = int(self.gap_days * 48)
        
        # Calculate split points
        split_size = (n - test_size - gap_size) // self.n_splits
        
        for i in range(self.n_splits):
            # Expanding window
            train_end = split_size * (i + 1)
            test_start = train_end + gap_size
            test_end = min(test_start + test_size, n)
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
