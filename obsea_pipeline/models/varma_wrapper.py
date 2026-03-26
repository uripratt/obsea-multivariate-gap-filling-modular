import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.api import VAR

from obsea_pipeline.config.settings import INTERPOLATION_CONFIG

logger = logging.getLogger(__name__)

def interpolate_var(df, target_var, max_gap=10):
    """
    Vector Autoregression (VAR) Model.
    Modela las dependencias mutuas entre múltiples series temporales.
    
    Ahora realmente usa las predicciones del modelo VAR para rellenar gaps,
    con physical clamping y fallback a interpolación temporal.
    """
    logger.info("  Running [VAR model]...")
    result = df[target_var].copy()

    try:
        # Select key predictor variables
        key_vars = [target_var, 'SVEL', 'PRES', 'BUOY_WSPD']
        available_vars = [v for v in key_vars if v in df.columns]
        
        # Impute predictors so VAR doesn't fail on exogn gaps
        df_imputed = df[available_vars].interpolate(method='time').bfill().ffill()
        
        # Fit on longest contiguous block or just imputed data
        min_samples = INTERPOLATION_CONFIG.get('min_samples', 200)
        if len(df_imputed) < min_samples:
            logger.warning(f"  VAR: Not enough data ({len(df_imputed)}/{min_samples}). Falling back.")
            return result.interpolate(method='time', limit=max_gap)
        
        model = VAR(df_imputed)
        lag_order = min(5, len(df_imputed) // 5)
        if lag_order < 1:
            logger.warning("  VAR: Insufficient data for even 1 lag. Falling back.")
            return result.interpolate(method='time', limit=max_gap)
        
        fit_result = model.fit(maxlags=lag_order, ic='aic')
        k_ar = fit_result.k_ar
        
        # Use the VAR model to recursively forecast into gaps
        target_idx = available_vars.index(target_var)
        y_values = df_imputed.values
        gaps_mask = result.isna().values
        
        for idx in range(k_ar, len(y_values)):
            if gaps_mask[idx]:
                # Recursive 1-step forecast using past k_ar steps
                lagged_values = y_values[idx-k_ar : idx]
                pred = fit_result.forecast(lagged_values, steps=1)
                y_values[idx, target_idx] = pred[0, target_idx]
                result.iloc[idx] = pred[0, target_idx]
                
        # Fill any beginning gaps that lacked AR lag history
        result = result.interpolate(method='time', limit=max_gap).bfill().ffill()
        
        # Physical clamping to prevent VARMA divergence
        observed = df[target_var].dropna()
        if not observed.empty:
            obs_min, obs_max, obs_std = observed.min(), observed.max(), observed.std()
            lo, hi = obs_min - 3 * obs_std, obs_max + 3 * obs_std
            violations = ((result < lo) | (result > hi)) & result.notna()
            if violations.sum() > 0:
                logger.warning(f"  [VAR CLAMP] {violations.sum()} values outside [{lo:.2f}, {hi:.2f}] → clamped")
                result = result.clip(lower=lo, upper=hi)
        
    except Exception as e:
         logger.error(f"  VAR failed: {e}")
         result = result.interpolate(method='time', limit=max_gap)
         
    return result

def interpolate_varma(df, target_var, max_gap=10):
    """
    VARMA wrapper. Delegates to interpolate_var with physical clamping.
    """
    logger.info("  Running [VARMA model proxy]...")
    return interpolate_var(df, target_var, max_gap=max_gap)
