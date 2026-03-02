"""
Data preprocessing utilities including normalization and outlier detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for OBSEA time series data.
    
    Handles:
    - Outlier detection and removal
    - Normalization/standardization
    - Storing preprocessing parameters for inverse transform
    """
    
    def __init__(
        self,
        outlier_method: str = 'zscore',
        outlier_threshold: float = 3.0,
        normalization: str = 'standard',
    ):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        outlier_method : str
            Method for outlier detection: 'zscore', 'iqr', or 'none'
        outlier_threshold : float
            Threshold for outlier detection
        normalization : str
            Normalization method: 'standard', 'minmax', 'robust', or 'none'
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.normalization = normalization
        
        self.scalers = {}
        self.outlier_masks = {}
        
    def detect_outliers(
        self,
        data: pd.Series,
        method: str = 'zscore',
        threshold: float = 3.0,
    ) -> np.ndarray:
        """
        Detect outliers in a data series.
        
        Parameters
        ----------
        data : pd.Series
            Input data
        method : str
            Detection method: 'zscore' or 'iqr'
        threshold : float
            Threshold value
            
        Returns
        -------
        np.ndarray
            Boolean mask where True indicates an outlier
        """
        valid_data = data.dropna()
        
        if len(valid_data) == 0:
            return np.zeros(len(data), dtype=bool)
        
        if method == 'zscore':
            z_scores = np.abs((valid_data - valid_data.mean()) / valid_data.std())
            outlier_mask = z_scores > threshold
            
        elif method == 'iqr':
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (valid_data < lower_bound) | (valid_data > upper_bound)
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Create full mask
        full_mask = np.zeros(len(data), dtype=bool)
        full_mask[data.notna()] = outlier_mask.values
        
        return full_mask
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        variables: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Remove outliers from specified variables.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        variables : list of str
            Variables to process
            
        Returns
        -------
        pd.DataFrame
            Dataframe with outliers set to NaN
        dict
            Number of outliers detected per variable
        """
        df_clean = df.copy()
        outlier_counts = {}
        
        if self.outlier_method == 'none':
            logger.info("Outlier detection disabled")
            return df_clean, outlier_counts
        
        for var in variables:
            if var not in df.columns:
                continue
                
            mask = self.detect_outliers(
                df[var],
                method=self.outlier_method,
                threshold=self.outlier_threshold,
            )
            
            self.outlier_masks[var] = mask
            outlier_counts[var] = mask.sum()
            
            df_clean.loc[mask, var] = np.nan
            
            if outlier_counts[var] > 0:
                logger.info(
                    f"Removed {outlier_counts[var]} outliers from {var} "
                    f"({100*outlier_counts[var]/len(df):.2f}%)"
                )
        
        return df_clean, outlier_counts
    
    def fit_scaler(self, data: pd.Series, method: str = 'standard'):
        """
        Fit a scaler to the data.
        
        Parameters
        ----------
        data : pd.Series
            Input data
        method : str
            Scaling method
            
        Returns
        -------
        scaler
            Fitted scaler object
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit on non-NaN values
        valid_data = data.dropna().values.reshape(-1, 1)
        scaler.fit(valid_data)
        
        return scaler
    
    def normalize(
        self,
        df: pd.DataFrame,
        variables: List[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize variables in the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        variables : list of str
            Variables to normalize
        fit : bool
            Whether to fit scalers (True) or use existing ones (False)
            
        Returns
        -------
        pd.DataFrame
            Normalized dataframe
        """
        if self.normalization == 'none':
            logger.info("Normalization disabled")
            return df.copy()
        
        df_norm = df.copy()
        
        for var in variables:
            if var not in df.columns:
                continue
            
            if fit:
                # Fit scaler
                self.scalers[var] = self.fit_scaler(df[var], self.normalization)
                logger.info(f"Fitted {self.normalization} scaler for {var}")
            
            # Transform
            if var in self.scalers:
                valid_mask = df[var].notna()
                if valid_mask.sum() > 0:
                    df_norm.loc[valid_mask, var] = self.scalers[var].transform(
                        df.loc[valid_mask, var].values.reshape(-1, 1)
                    ).flatten()
        
        return df_norm
    
    def inverse_normalize(
        self,
        df: pd.DataFrame,
        variables: List[str],
    ) -> pd.DataFrame:
        """
        Inverse transform normalized data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Normalized dataframe
        variables : list of str
            Variables to inverse transform
            
        Returns
        -------
        pd.DataFrame
            Original scale dataframe
        """
        if self.normalization == 'none':
            return df.copy()
        
        df_orig = df.copy()
        
        for var in variables:
            if var not in df.columns or var not in self.scalers:
                continue
            
            valid_mask = df[var].notna()
            if valid_mask.sum() > 0:
                df_orig.loc[valid_mask, var] = self.scalers[var].inverse_transform(
                    df.loc[valid_mask, var].values.reshape(-1, 1)
                ).flatten()
        
        return df_orig
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        variables: List[str],
        remove_outliers: bool = True,
    ) -> pd.DataFrame:
        """
        Fit preprocessor and transform data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        variables : list of str
            Variables to process
        remove_outliers : bool
            Whether to remove outliers
            
        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe
        """
        df_processed = df.copy()
        
        # Remove outliers
        if remove_outliers:
            df_processed, _ = self.remove_outliers(df_processed, variables)
        
        # Normalize
        df_processed = self.normalize(df_processed, variables, fit=True)
        
        return df_processed
    
    def transform(
        self,
        df: pd.DataFrame,
        variables: List[str],
    ) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        variables : list of str
            Variables to process
            
        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe
        """
        df_processed = df.copy()
        
        # Normalize (no outlier removal on test data)
        df_processed = self.normalize(df_processed, variables, fit=False)
        
        return df_processed
