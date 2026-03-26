import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from obsea_pipeline.config.settings import HARDWARE_CONFIG

logger = logging.getLogger(__name__)


def _apply_physical_clamping(result: pd.Series, df: pd.DataFrame, target_var: str, margin_sigmas: float = 3.0) -> pd.Series:
    """
    Post-prediction physical constraint clamping.
    
    Clips imputed values to [observed_min - margin, observed_max + margin]
    where margin = margin_sigmas × std(observed). This prevents physically
    impossible predictions (e.g., 55°C sea temperature).
    
    Returns the clamped series and logs the number of clipped values.
    """
    observed = df[target_var].dropna()
    if observed.empty:
        return result
    
    obs_min = observed.min()
    obs_max = observed.max()
    obs_std = observed.std()
    
    lo = obs_min - margin_sigmas * obs_std
    hi = obs_max + margin_sigmas * obs_std
    
    # Count violations before clamping
    violations = ((result < lo) | (result > hi)) & result.notna()
    n_violations = violations.sum()
    
    if n_violations > 0:
        logger.warning(f"  [PHYSICAL CLAMP] {n_violations} values outside [{lo:.2f}, {hi:.2f}] → clamped")
        result = result.clip(lower=lo, upper=hi)
    
    return result


def _compute_dynamic_n_steps(max_gap_size: int, default: int = 128) -> int:
    """
    Compute n_steps dynamically based on gap size.
    
    For larger gaps, the model needs a wider window to capture the temporal
    context. Returns min(gap_size * 1.5, 512) to avoid memory issues.
    """
    if max_gap_size is None:
        return default
    
    # At minimum, n_steps should cover the gap + some context
    # At maximum, cap at 512 to avoid OOM
    dynamic = int(max(default, min(max_gap_size * 1.5, 512)))
    return dynamic


def interpolate_missforest(df: pd.DataFrame, target_var: str, predictor_vars: list = None):
    """Wrapper for MissForest Imputation."""
    try:
        from obsea_pipeline.models.missforest_model import MissForestImputer
        imputer = MissForestImputer()
        imputer.fit(df, target_var, multivariate_vars=predictor_vars)
        result = imputer.predict(df)
        return _apply_physical_clamping(result, df, target_var)
    except Exception as e:
        logger.error(f"MissForest failed: {e}")
        return df[target_var].interpolate(method='time')


def interpolate_saits(df: pd.DataFrame, target_var: str, predictor_vars: list = None, max_gap_size: int = None):
    """Wrapper for SAITS Imputation with dynamic n_steps and physical clamping."""
    try:
        from obsea_pipeline.models.saits_model import SAITSImputer
        n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
        n_steps = _compute_dynamic_n_steps(max_gap_size, default=128)
        
        logger.info(f"  [SAITS] n_steps={n_steps}, n_features={n_features}, max_gap={max_gap_size}")
        
        imputer = SAITSImputer(
            n_steps=n_steps, n_features=n_features, 
            epochs=HARDWARE_CONFIG.get('dl_epochs', 20), 
            batch_size=HARDWARE_CONFIG.get('dl_transformer_batch_size', 32),
            max_gap_size=max_gap_size
        )
        imputer.fit(df, target_var, multivariate_vars=predictor_vars)
        result = imputer.predict(df)
        return _apply_physical_clamping(result, df, target_var)
    except Exception as e:
        logger.error(f"SAITS failed: {e}")
        return df[target_var].interpolate(method='time', limit=max_gap_size)


def interpolate_imputeformer(df: pd.DataFrame, target_var: str, predictor_vars: list = None, max_gap_size: int = None):
    """Wrapper for ImputeFormer Imputation with dynamic n_steps and physical clamping."""
    try:
        from obsea_pipeline.models.imputeformer_model import ImputeFormerImputer
        n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
        n_steps = _compute_dynamic_n_steps(max_gap_size, default=128)
        
        logger.info(f"  [ImputeFormer] n_steps={n_steps}, n_features={n_features}, max_gap={max_gap_size}")
        
        imputer = ImputeFormerImputer(
            n_steps=n_steps, n_features=n_features, 
            epochs=HARDWARE_CONFIG.get('dl_epochs', 20), 
            batch_size=HARDWARE_CONFIG.get('dl_transformer_batch_size', 16),
            max_gap_size=max_gap_size
        )
        imputer.fit(df, target_var, multivariate_vars=predictor_vars)
        result = imputer.predict(df)
        return _apply_physical_clamping(result, df, target_var)
    except Exception as e:
        logger.error(f"ImputeFormer failed: {e}")
        return df[target_var].interpolate(method='time')


def interpolate_brits(df: pd.DataFrame, target_var: str, predictor_vars: list = None, max_gap_size: int = None):
    """Wrapper for BRITS Imputation with physical clamping."""
    try:
        from obsea_pipeline.models.brits_model import BRITSImputer
        n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
        imputer = BRITSImputer(
            n_steps=128, n_features=n_features, 
            epochs=HARDWARE_CONFIG.get('dl_epochs', 20), 
            batch_size=HARDWARE_CONFIG.get('dl_rnn_batch_size', 32),
            max_gap_size=max_gap_size
        )
        imputer.fit(df, target_var, multivariate_vars=predictor_vars)
        result = imputer.predict(df)
        return _apply_physical_clamping(result, df, target_var)
    except Exception as e:
        logger.error(f"BRITS failed: {e}")
        return df[target_var].interpolate(method='time')


def interpolate_brits_pro(df: pd.DataFrame, target_var: str, predictor_vars: list = None, max_gap_size: int = None):
    """Wrapper for BRITS Pro Imputation with physical clamping."""
    try:
        from obsea_pipeline.models.brits_model_pro import BRITSProImputer
        
        n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
        imputer = BRITSProImputer(
            n_steps=128, n_features=n_features, rnn_hidden_size=512, 
            epochs=HARDWARE_CONFIG.get('dl_epochs', 200), 
            batch_size=HARDWARE_CONFIG.get('dl_rnn_batch_size', 64),
            max_gap_size=max_gap_size
        )
        imputer.fit(df, target_var, multivariate_vars=predictor_vars)
        result = imputer.predict(df)
        return _apply_physical_clamping(result, df, target_var)
    except Exception as e:
        logger.error(f"BRITS Pro failed: {e}")
        return df[target_var].interpolate(method='time', limit=max_gap_size)
