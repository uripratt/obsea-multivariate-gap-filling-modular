import logging
import pandas as pd
from pathlib import Path
import sys

from obsea_pipeline.config.settings import HARDWARE_CONFIG

logger = logging.getLogger(__name__)

def interpolate_missforest(df: pd.DataFrame, target_var: str, predictor_vars: list = None, df_train: pd.DataFrame = None):
    """Wrapper for MissForest Imputation."""
    try:
        from obsea_pipeline.models.missforest_model import MissForestImputer
        imputer = MissForestImputer()
        train_data = df_train if df_train is not None else df
        imputer.fit(train_data, target_var, multivariate_vars=predictor_vars)
        return imputer.predict(df)
    except Exception as e:
        logger.error(f"MissForest failed: {e}")
        return df[target_var].interpolate(method='time')


def interpolate_saits(df: pd.DataFrame, target_var: str, predictor_vars: list = None, df_train: pd.DataFrame = None):
    """Wrapper for SAITS Imputation."""
    try:
        from obsea_pipeline.models.saits_model import SAITSImputer
        n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
        imputer = SAITSImputer(
            n_steps=128, n_features=n_features, 
            epochs=HARDWARE_CONFIG.get('dl_epochs', 20), 
            batch_size=HARDWARE_CONFIG.get('dl_transformer_batch_size', 32)
        )
        train_data = df_train if df_train is not None else df
        imputer.fit(train_data, target_var, multivariate_vars=predictor_vars)
        return imputer.predict(df)
    except Exception as e:
        logger.error(f"SAITS failed: {e}")
        return df[target_var].interpolate(method='time')


def interpolate_imputeformer(df: pd.DataFrame, target_var: str, predictor_vars: list = None, max_gap_size: int = None, df_train: pd.DataFrame = None):
    """Wrapper for ImputeFormer Imputation."""
    try:
        from obsea_pipeline.models.imputeformer_model import ImputeFormerImputer
        n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
        imputer = ImputeFormerImputer(
            n_steps=128, n_features=n_features, 
            epochs=HARDWARE_CONFIG.get('dl_epochs', 20), 
            batch_size=HARDWARE_CONFIG.get('dl_transformer_batch_size', 16),
            max_gap_size=max_gap_size
        )
        train_data = df_train if df_train is not None else df
        imputer.fit(train_data, target_var, multivariate_vars=predictor_vars)
        return imputer.predict(df)
    except Exception as e:
        logger.error(f"ImputeFormer failed: {e}")
        return df[target_var].interpolate(method='time')


def interpolate_brits(df: pd.DataFrame, target_var: str, predictor_vars: list = None, max_gap_size: int = None, df_train: pd.DataFrame = None):
    """Wrapper for BRITS Imputation."""
    try:
        from obsea_pipeline.models.brits_model import BRITSImputer
        n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
        imputer = BRITSImputer(
            n_steps=128, n_features=n_features, 
            epochs=HARDWARE_CONFIG.get('dl_epochs', 20), 
            batch_size=HARDWARE_CONFIG.get('dl_rnn_batch_size', 32),
            max_gap_size=max_gap_size
        )
        train_data = df_train if df_train is not None else df
        imputer.fit(train_data, target_var, multivariate_vars=predictor_vars)
        return imputer.predict(df)
    except Exception as e:
        logger.error(f"BRITS failed: {e}")
        return df[target_var].interpolate(method='time')


def interpolate_brits_pro(df: pd.DataFrame, target_var: str, predictor_vars: list = None, df_train: pd.DataFrame = None):
    """Wrapper for BRITS Pro Imputation."""
    try:
        from obsea_pipeline.models.brits_model_pro import BRITSProImputer
        
        n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
        imputer = BRITSProImputer(
            n_steps=128, n_features=n_features, rnn_hidden_size=512, 
            epochs=HARDWARE_CONFIG.get('dl_epochs', 200), 
            batch_size=HARDWARE_CONFIG.get('dl_rnn_batch_size', 64)
        )
        train_data = df_train if df_train is not None else df
        imputer.fit(train_data, target_var, multivariate_vars=predictor_vars)
        return imputer.predict(df)
    except Exception as e:
        logger.error(f"BRITS Pro failed: {e}")
        return df[target_var].interpolate(method='time')
