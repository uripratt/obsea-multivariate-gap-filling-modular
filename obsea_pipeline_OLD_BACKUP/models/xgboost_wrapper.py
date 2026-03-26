import logging
import pandas as pd
from typing import List

from obsea_pipeline.config.settings import HARDWARE_CONFIG
from obsea_pipeline.models.xgboost_model import XGBoostImputer
from obsea_pipeline.models.xgboost_model_pro import XGBoostProImputer

logger = logging.getLogger(__name__)

def interpolate_xgboost(df: pd.DataFrame, target_var: str, predictor_vars: List[str] = None, max_gap_size: int = None, df_train: pd.DataFrame = None):
    try:
        imputer = XGBoostImputer(
            xgb_params={'n_estimators': HARDWARE_CONFIG.get('xgb_n_estimators', 300)}
        )
        train_data = df_train if df_train is not None else df
        imputer.fit(train_data, target_var, multivariate_vars=predictor_vars)
        return imputer.predict(df, multivariate_vars=predictor_vars), imputer
    except Exception as e:
        logger.error(f"XGBoost failed: {e}")
        return df[target_var].interpolate(method='time', limit=max_gap_size), None

def interpolate_xgboost_pro(df: pd.DataFrame, target_var: str, predictor_vars: List[str] = None, max_gap_size: int = None, df_train: pd.DataFrame = None):
    try:
        imputer = XGBoostProImputer(
            xgb_params={'n_estimators': HARDWARE_CONFIG.get('xgb_n_estimators', 500)},
            bidirectional=True
        )
        train_data = df_train if df_train is not None else df
        imputer.fit(train_data, target_var, multivariate_vars=predictor_vars)
        return imputer.predict(df, multivariate_vars=predictor_vars), imputer
    except Exception as e:
        logger.error(f"XGBoost Pro failed: {e}")
        return df[target_var].interpolate(method='time', limit=max_gap_size), None
