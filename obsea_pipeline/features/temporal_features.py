"""
Temporal feature engineering for time series data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """
    Create temporal features from time series data.
    
    Includes:
    - Lag features
    - Rolling statistics
    - Cyclical time encodings
    - Time-based features
    """
    
    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        rolling_stats: Optional[List[str]] = None,
        include_cyclical: bool = True,
        include_time_features: bool = True,
    ):
        """
        Initialize feature engineer.
        
        Parameters
        ----------
        lags : list of int, optional
            Lag periods (in time steps)
        rolling_windows : list of int, optional
            Rolling window sizes (in time steps)
        rolling_stats : list of str, optional
            Statistics to compute: 'mean', 'std', 'min', 'max'
        include_cyclical : bool
            Include cyclical time encodings
        include_time_features : bool
            Include time-based features
        """
        self.lags = lags or [2, 4, 12, 24, 48, 336]  # 1h, 2h, 6h, 12h, 24h, 1week
        self.rolling_windows = rolling_windows or [12, 48, 336]  # 6h, 24h, 1week
        self.rolling_stats = rolling_stats or ['mean', 'std']
        self.include_cyclical = include_cyclical
        self.include_time_features = include_time_features
        
    def create_lag_features(
        self,
        df: pd.DataFrame,
        variable: str,
    ) -> pd.DataFrame:
        """
        Create lag features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        variable : str
            Variable name
            
        Returns
        -------
        pd.DataFrame
            Dataframe with lag features added
        """
        df_with_lags = df.copy()
        
        for lag in self.lags:
            col_name = f"{variable}_lag_{lag}"
            df_with_lags[col_name] = df[variable].shift(lag)
        
        logger.debug(f"Created {len(self.lags)} lag features for {variable}")
        
        return df_with_lags
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        variable: str,
    ) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        variable : str
            Variable name
            
        Returns
        -------
        pd.DataFrame
            Dataframe with rolling features added
        """
        df_with_rolling = df.copy()
        
        for window in self.rolling_windows:
            for stat in self.rolling_stats:
                col_name = f"{variable}_roll_{stat}_{window}"
                
                if stat == 'mean':
                    df_with_rolling[col_name] = df[variable].rolling(window, min_periods=1).mean()
                elif stat == 'std':
                    df_with_rolling[col_name] = df[variable].rolling(window, min_periods=1).std()
                elif stat == 'min':
                    df_with_rolling[col_name] = df[variable].rolling(window, min_periods=1).min()
                elif stat == 'max':
                    df_with_rolling[col_name] = df[variable].rolling(window, min_periods=1).max()
        
        logger.debug(
            f"Created {len(self.rolling_windows) * len(self.rolling_stats)} "
            f"rolling features for {variable}"
        )
        
        return df_with_rolling
    
    def create_cyclical_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create cyclical time encodings using sin/cos transformation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with datetime index
            
        Returns
        -------
        pd.DataFrame
            Dataframe with cyclical features added
        """
        df_with_cyclical = df.copy()
        
        # Hour of day (0-23)
        hour = df.index.hour
        df_with_cyclical['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df_with_cyclical['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6)
        day_of_week = df.index.dayofweek
        df_with_cyclical['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df_with_cyclical['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Month (1-12)
        month = df.index.month
        df_with_cyclical['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        df_with_cyclical['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        
        # Day of year (1-365/366)
        day_of_year = df.index.dayofyear
        max_day = 365 + df.index.is_leap_year.astype(int)
        df_with_cyclical['doy_sin'] = np.sin(2 * np.pi * (day_of_year - 1) / max_day)
        df_with_cyclical['doy_cos'] = np.cos(2 * np.pi * (day_of_year - 1) / max_day)
        
        logger.debug("Created cyclical time features")
        
        return df_with_cyclical
    
    def create_time_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create basic time-based features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with datetime index
            
        Returns
        -------
        pd.DataFrame
            Dataframe with time features added
        """
        df_with_time = df.copy()
        
        # Basic time features
        df_with_time['hour_of_day'] = df.index.hour
        df_with_time['day_of_week'] = df.index.dayofweek
        df_with_time['day_of_month'] = df.index.day
        df_with_time['day_of_year'] = df.index.dayofyear
        df_with_time['month'] = df.index.month
        df_with_time['year'] = df.index.year
        
        # Is weekend
        df_with_time['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Season (Northern Hemisphere)
        month = df.index.month
        df_with_time['season'] = 0  # Winter
        df_with_time.loc[month.isin([3, 4, 5]), 'season'] = 1  # Spring
        df_with_time.loc[month.isin([6, 7, 8]), 'season'] = 2  # Summer
        df_with_time.loc[month.isin([9, 10, 11]), 'season'] = 3  # Fall
        
        logger.debug("Created time-based features")
        
        return df_with_time
    
    def create_time_since_observation(
        self,
        df: pd.DataFrame,
        variable: str,
    ) -> pd.DataFrame:
        """
        Create feature for time since last valid observation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        variable : str
            Variable name
            
        Returns
        -------
        pd.DataFrame
            Dataframe with time-since-obs feature added
        """
        df_with_feature = df.copy()
        
        # Find time steps since last valid observation
        valid_mask = df[variable].notna()
        
        # Cumulative count reset at each valid observation
        time_since = (~valid_mask).cumsum() - (~valid_mask).cumsum().where(valid_mask).ffill().fillna(0)
        
        df_with_feature[f'{variable}_time_since_obs'] = time_since
        
        logger.debug(f"Created time-since-observation feature for {variable}")
        
        return df_with_feature
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_variable: str,
        multivariate_vars: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create all features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        target_variable : str
            Target variable for lag/rolling features
        multivariate_vars : list of str, optional
            Other variables to include as features
            
        Returns
        -------
        pd.DataFrame
            Dataframe with all features
        """
        df_features = df.copy()
        
        # Lag features
        df_features = self.create_lag_features(df_features, target_variable)
        
        # Rolling features
        df_features = self.create_rolling_features(df_features, target_variable)
        
        # Time since observation
        df_features = self.create_time_since_observation(df_features, target_variable)
        
        # Cyclical features
        if self.include_cyclical:
            df_features = self.create_cyclical_features(df_features)
        
        # Time features
        if self.include_time_features:
            df_features = self.create_time_features(df_features)
        
        # Multivariate features (include other variables directly)
        if multivariate_vars:
            for var in multivariate_vars:
                if var in df.columns and var != target_variable:
                    # Include the variable itself
                    pass  # Already in df_features
                    
                    # Optionally create lags for other variables too
                    # df_features = self.create_lag_features(df_features, var)
        
        n_features = len(df_features.columns) - len(df.columns)
        logger.debug(f"Created {n_features} temporal features")
        
        return df_features
    
    def transform(
        self,
        df: pd.DataFrame,
        target_variable: str,
        multivariate_vars: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Alias for fit_transform since this engineer is stateless.
        """
        return self.fit_transform(df, target_variable, multivariate_vars)
