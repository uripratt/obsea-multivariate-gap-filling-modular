import logging
import pandas as pd
import numpy as np

from obsea_pipeline.config.settings import HARDWARE_CONFIG
from obsea_pipeline.models.multivariate_lstm_model import MultivariateLSTMImputer

logger = logging.getLogger(__name__)

def interpolate_bilstm(df: pd.DataFrame, target_var: str, predictor_vars: list = None, max_gap_size: int = None):
    """Wrapper for the Multivariate Bi-LSTM Imputer with physical clamping."""
    try:
        if predictor_vars is None:
            # Auto-select 4 most correlated features to provide support
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            candidates = [col for col in numeric_cols if col != target_var and '_QC' not in col]
            if candidates:
                corrs = df[candidates].corrwith(df[target_var]).abs().sort_values(ascending=False)
                predictor_vars = [target_var] + corrs.head(4).index.tolist()
            else:
                predictor_vars = [target_var]
        elif target_var not in predictor_vars:
            predictor_vars = [target_var] + predictor_vars

        imputer = MultivariateLSTMImputer(
            target_var=target_var,
            predictor_vars=predictor_vars,
            hidden_size=128,
            num_layers=2,
            batch_size=HARDWARE_CONFIG.get('dl_rnn_batch_size', 32),
            epochs=HARDWARE_CONFIG.get('dl_epochs', 50),
            max_gap_size=max_gap_size
        )
        
        # The class handles STL residuals, train/val split, scaling and early stopping internally
        imputer.fit(df[predictor_vars], verbose=False)
        
        predicted_filled = imputer.predict(df[predictor_vars])
        
        # Merge gaps back to original
        result_series = df[target_var].copy()
        gaps_to_fill = result_series.isna() & predicted_filled.notna()
        result_series.loc[gaps_to_fill] = predicted_filled.loc[gaps_to_fill]
        
        # Physical clamping: prevent impossible values
        observed = df[target_var].dropna()
        if not observed.empty:
            obs_min, obs_max, obs_std = observed.min(), observed.max(), observed.std()
            lo, hi = obs_min - 3 * obs_std, obs_max + 3 * obs_std
            violations = ((result_series < lo) | (result_series > hi)) & result_series.notna()
            if violations.sum() > 0:
                logger.warning(f"  [BiLSTM CLAMP] {violations.sum()} values outside [{lo:.2f}, {hi:.2f}] → clamped")
                result_series = result_series.clip(lower=lo, upper=hi)
        
        return result_series

    except Exception as e:
        logger.error(f"Multivariate Bi-LSTM failed: {e}")
        return df[target_var].interpolate(method='time', limit=max_gap_size)

