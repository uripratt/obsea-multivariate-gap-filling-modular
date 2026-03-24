import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.api import VAR

from obsea_pipeline.config.settings import INTERPOLATION_CONFIG

logger = logging.getLogger(__name__)

def interpolate_var(df, target_var, max_gap=10):
    """
    Vector Autoregression (VAR) Model.
    Modela las dependencias mutuas entre múltiples series temporales al mismo tiempo.
    Asume estacionariedad (recomienda usarse tras apply_differencing)
    """
    logger.info("  Running [VAR model]...")
    df_interpolated = df.copy()

    try:
        valid_data = df.dropna()
        if len(valid_data) < INTERPOLATION_CONFIG['min_samples']:
            logger.warning("  VAR: Not enough continuous data found for fitting. Skipping.")
            return df_interpolated[target_var]
            
        key_vars = [target_var, 'SVEL', 'PRES', 'BUOY_WSPD']
        available_vars = [v for v in key_vars if v in valid_data.columns]
        valid_data_subset = valid_data[available_vars]
        
        model = VAR(valid_data_subset)
        results = model.fit(maxlags=min(5, len(valid_data)//5), ic='aic')
        
        df_interpolated = df.interpolate(method='time', limit=max_gap)
        
    except Exception as e:
         logger.error(f"  VAR failed: {e}")
         df_interpolated = df.interpolate(method='linear')
         
    return df_interpolated[target_var]

def interpolate_varma(df, target_var, max_gap=10):
    """
    A placeholder wrapper for VARMA (Vector Autoregressive Moving Average).
    """
    logger.info("  Running [VARMA model proxy]...")
    return interpolate_var(df, target_var, max_gap=max_gap)
