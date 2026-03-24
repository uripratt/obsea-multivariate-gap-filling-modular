"""
Baseline gap-filling methods using various interpolation techniques.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)


class BaselineImputer:
    """
    Baseline imputation methods for time series gaps.
    
    Supported methods:
    - Linear interpolation
    - PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    - Cubic spline
    - Forward fill
    - Backward fill
    """
    
    def __init__(
        self,
        method: Literal['linear', 'pchip', 'spline', 'forward', 'backward'] = 'linear',
    ):
        """
        Initialize baseline imputer.
        
        Parameters
        ----------
        method : str
            Interpolation method to use
        """
        self.method = method
        self.name = f"Baseline_{method}"
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit method (no-op for baseline methods).
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data (not used)
        y : pd.Series, optional
            Target (not used)
            
        Returns
        -------
        self
        """
        # Baseline methods don't require training
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Impute missing values using the specified method.
        
        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with missing values in first column
            
        Returns
        -------
        pd.Series
            Imputed series
        """
        # Assume first column is the target variable
        series = X.iloc[:, 0].copy() if isinstance(X, pd.DataFrame) else X.copy()
        
        return self.impute(series)
    
    def impute(self, series: pd.Series) -> pd.Series:
        """
        Impute missing values in a series.
        
        Parameters
        ----------
        series : pd.Series
            Input series with missing values
            
        Returns
        -------
        pd.Series
            Imputed series
        """
        if self.method == 'linear':
            return self._linear_interpolation(series)
        elif self.method == 'pchip':
            return self._pchip_interpolation(series)
        elif self.method == 'spline':
            return self._spline_interpolation(series)
        elif self.method == 'forward':
            return series.fillna(method='ffill')
        elif self.method == 'backward':
            return series.fillna(method='bfill')
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _linear_interpolation(self, series: pd.Series) -> pd.Series:
        """
        Linear interpolation.
        
        Parameters
        ----------
        series : pd.Series
            Input series
            
        Returns
        -------
        pd.Series
            Interpolated series
        """
        return series.interpolate(method='linear', limit_direction='both')
    
    def _pchip_interpolation(self, series: pd.Series) -> pd.Series:
        """
        PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.
        
        Preserves monotonicity and is more stable than cubic spline.
        
        Parameters
        ----------
        series : pd.Series
            Input series
            
        Returns
        -------
        pd.Series
            Interpolated series
        """
        # Get valid (non-NaN) values
        valid_mask = series.notna()
        
        if valid_mask.sum() < 2:
            logger.warning("Not enough valid points for PCHIP interpolation")
            return series
        
        # Get indices and values
        valid_indices = np.where(valid_mask)[0]
        valid_values = series[valid_mask].values
        
        # Create interpolator
        interpolator = PchipInterpolator(valid_indices, valid_values)
        
        # Interpolate at missing positions
        all_indices = np.arange(len(series))
        interpolated = interpolator(all_indices)
        
        # Create result series
        result = series.copy()
        result.iloc[:] = interpolated
        
        return result
    
    def _spline_interpolation(self, series: pd.Series) -> pd.Series:
        """
        Cubic spline interpolation.
        
        Parameters
        ----------
        series : pd.Series
            Input series
            
        Returns
        -------
        pd.Series
            Interpolated series
        """
        # Get valid values
        valid_mask = series.notna()
        
        if valid_mask.sum() < 4:
            logger.warning("Not enough valid points for cubic spline (need >= 4), falling back to linear")
            return self._linear_interpolation(series)
        
        valid_indices = np.where(valid_mask)[0]
        valid_values = series[valid_mask].values
        
        # Create interpolator
        try:
            interpolator = CubicSpline(valid_indices, valid_values)
            
            # Interpolate
            all_indices = np.arange(len(series))
            interpolated = interpolator(all_indices)
            
            # Create result series
            result = series.copy()
            result.iloc[:] = interpolated
            
            return result
        
        except Exception as e:
            logger.warning(f"Cubic spline failed: {e}, falling back to linear")
            return self._linear_interpolation(series)


def impute_with_baseline(
    df: pd.DataFrame,
    variable: str,
    method: str = 'linear',
) -> pd.Series:
    """
    Convenience function to impute a variable using a baseline method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variable : str
        Variable to impute
    method : str
        Imputation method
        
    Returns
    -------
    pd.Series
        Imputed series
    """
    imputer = BaselineImputer(method=method)
    imputed = imputer.impute(df[variable])
    
    logger.info(f"Imputed {variable} using {method} interpolation")
    
    return imputed
