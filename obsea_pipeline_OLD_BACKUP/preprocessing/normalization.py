import numpy as np
import pandas as pd
import logging

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    pass

logger = logging.getLogger(__name__)

def robust_scale(series: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    """Robust percentile-based scaling for outlier resistance."""
    valid = series.dropna()
    if len(valid) == 0:
        return series
    
    low = valid.quantile(q_low)
    high = valid.quantile(q_high)
    median = valid.median()
    
    iqr = high - low
    if iqr == 0:
        return series - median
    
    return (series - median) / iqr

def log_transform(series: pd.Series, offset: float = 1.0) -> pd.Series:
    """Log1p transform for positively skewed oceanographic variables (e.g., WAVE height)."""
    series_positive = series.clip(lower=0)
    return np.log1p(series_positive + offset - 1)

def compute_anomaly(df: pd.DataFrame, var: str, groupby: list = ['dayofyear', 'hour']) -> pd.Series:
    """Compute climatological anomaly: T' = T - T̄(doy, hr)."""
    if var not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    series = df[var].copy()
    group_cols = {}
    
    if 'dayofyear' in groupby: group_cols['dayofyear'] = df.index.dayofyear
    if 'hour' in groupby: group_cols['hour'] = df.index.hour
    if 'month' in groupby: group_cols['month'] = df.index.month
    
    if not group_cols:
        return series - series.mean()
    
    group_df = pd.DataFrame(group_cols, index=df.index)
    group_df[var] = series
    climatology = group_df.groupby(list(group_cols.keys()))[var].transform('mean')
    
    return series - climatology

def stl_decompose(series: pd.Series, period: int = 48, robust: bool = True):
    """
    STL (Seasonal-Trend decomposition using LOESS). 
    Period of 48 = 24 hours at 30-minute resolution.
    """
    try:
        # STL requires no missing values temporarily
        series_filled = series.interpolate(method='linear').bfill().ffill()
        
        if len(series_filled.dropna()) < period * 2:
            logger.warning("Insufficient data for STL decomposition")
            return series, pd.Series(0, index=series.index), pd.Series(0, index=series.index)
            
        chunk_size = 35040 # Prevent OOM over large 10-year datasets
        trends, seasonals, resids = [], [], []
        
        for i in range(0, len(series_filled), chunk_size):
            chunk = series_filled.iloc[i : i + chunk_size]
            
            if len(chunk) < period * 2:
                trend_c = chunk.rolling(window=period, center=True, min_periods=1).mean()
                seasonal_c = pd.Series(0, index=chunk.index)
                resid_c = chunk - trend_c
            else:
                stl = STL(chunk, period=period, robust=robust)
                result = stl.fit()
                trend_c = result.trend
                seasonal_c = result.seasonal
                resid_c = result.resid
                
            trends.append(trend_c)
            seasonals.append(seasonal_c)
            resids.append(resid_c)
            
        return pd.concat(trends), pd.concat(seasonals), pd.concat(resids)
        
    except Exception as e:
        logger.error(f"STL decomposition failed: {e}")
        return series, pd.Series(0, index=series.index), pd.Series(0, index=series.index)

def apply_differencing(series: pd.Series, order: int = 1) -> pd.Series:
    """Apply differencing for statistical stationary ML."""
    result = series.copy()
    for _ in range(order):
        result = result.diff()
    return result

def check_stationarity(series: pd.Series) -> dict:
    """ADF Dickey-Fuller test wrapper."""
    try:
        series_clean = series.dropna()
        if len(series_clean) < 100:
            return {'error': 'Insufficient data'}
            
        result = adfuller(series_clean, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < 0.05
        
        return {
            'p_value': p_value,
            'is_stationary': is_stationary
        }
    except Exception as e:
        return {'error': str(e)}
