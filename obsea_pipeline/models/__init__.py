"""Models module init with graceful degradation for Torch dependencies"""
import logging

logger = logging.getLogger(__name__)

# Basic models (No dependencies)
from .baseline import BaselineImputer, impute_with_baseline
from .xgboost_model import XGBoostImputer
from .varma_model import VARMAImputer

# Deep Learning Models (Protected imports)
try:
    from .lstm_model import LSTMImputer
    TORCH_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"  [IMPORT WARNING] LSTM/Torch models could not be loaded: {e}. These models will be unavailable.")
    LSTMImputer = None
    TORCH_AVAILABLE = False

__all__ = [
    'BaselineImputer',
    'impute_with_baseline',
    'XGBoostImputer',
    'LSTMImputer',
    'VARMAImputer',
    'TORCH_AVAILABLE'
]
