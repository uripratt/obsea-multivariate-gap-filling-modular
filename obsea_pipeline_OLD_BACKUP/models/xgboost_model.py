"""
XGBoost-based gap filling model with Bi-Directional and Recursive capabilities.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from pathlib import Path
import joblib
import logging

from ..features import TemporalFeatureEngineer

logger = logging.getLogger(__name__)


class XGBoostImputer:
    """
    XGBoost-based imputation using Bi-Directional Recursive strategy.
    
    Capabilities:
    - Bi-Directional: Trains Forward and Backward models and ensembles them.
    - Recursive: Uses previous predictions as lags for future predictions (Autoregressive).
    - Multivariate: Can use exogenous variables.
    """
    
    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        feature_config: Optional[Dict] = None,
        bidirectional: bool = False,
        max_gap_size: int = None,
    ):
        """
        Initialize XGBoost imputer.
        
        Parameters
        ----------
        xgb_params : dict, optional
            XGBoost hyperparameters
        feature_config : dict, optional
            Feature engineering configuration
        bidirectional : bool
            If True, trains both forward and backward models and averages predictions.
        """
        # Default XGBoost parameters 
        default_xgb_params = {
            'n_estimators': 100,    # Historically 100 for the good script
            'max_depth': 6,         # Prevent overfitting
            'learning_rate': 0.1,   # Historically 0.1
            'random_state': 42,
            'n_jobs': -1,
            'device': 'cuda',       # Try CUDA by default
            'tree_method': 'hist',  # Required for efficient GPU
        }
        
        # We REMOVE the Torch check because Torch is failing but XGBoost might succeed.
        # We will handle the fallback in fit() instead.
        
        self.xgb_params = {**default_xgb_params, **(xgb_params or {})}
        self.feature_config = feature_config or {}
        self.bidirectional = bidirectional
        self.max_gap_size = max_gap_size
        
        # Models storage
        self.models = {} # 'fwd', 'bwd'
        self.feature_engineers = {} # 'fwd', 'bwd'
        self.climatology = None
        
        self.is_residual_fwd = False
        self.is_residual_bwd = False
        
        self.feature_columns = None
        self.target_var = None
        
        # Physical range for value clamping (set during fit)
        self.observed_min = None
        self.observed_max = None
        
    def _create_feature_engineer(self):
        """Create a fresh feature engineer to prevent state cross-contamination."""
        
        # -------------------------------------------------------------
        # SCALE-AWARE FRAMEWORK V4.0
        # Dynamic Lag scaling based on max geometric gap size
        # -------------------------------------------------------------
        if self.max_gap_size is not None and not self.feature_config:
            if self.max_gap_size <= 12: # Micro/Short (6 hours)
                dynamic_lags = [1, 2, 3, 4, 6, 12] # Dense local lags
                logger.info(f"Scale-Aware Triggered: XGBoost tracking MICRO lags for {self.max_gap_size} pts gap.")
            elif self.max_gap_size <= 144: # Medium (3 days)
                dynamic_lags = [1, 2, 4, 12, 24, 48, 72] # Mix of local and daily
                logger.info(f"Scale-Aware Triggered: XGBoost tracking MEDIUM lags for {self.max_gap_size} pts gap.")
            else: # Long/Gigant
                # Skip immediate lags because they are just predicted noise deep inside a massive hole
                dynamic_lags = [24, 48, 96, 168, 336] # Deep daily and weekly structural lags
                logger.info(f"Scale-Aware Triggered: XGBoost tracking DEEP lags for {self.max_gap_size} pts gap.")
                
            return TemporalFeatureEngineer(
                lags=dynamic_lags,
                rolling_windows=[],
                rolling_stats=[],
                include_cyclical=False,
                include_time_features=True  
            )
            
        if not self.feature_config: # Use default simple config if feature_config is empty or None
            return TemporalFeatureEngineer(
                lags=[1, 2, 4, 12, 24], # Simple lags based on the old script
                rolling_windows=[],     # Remove rolling windows to prevent overfitting / NaN issues
                rolling_stats=[],
                include_cyclical=False,
                include_time_features=True  # Ensure hour, day_of_week, day_of_year are there
            )
        # If feature_config is provided, use it
        return TemporalFeatureEngineer(
            lags=self.feature_config.get('lags', [1, 2, 4, 12, 24]),
            rolling_windows=self.feature_config.get('rolling_windows', []),
            rolling_stats=self.feature_config.get('rolling_stats', []),
            include_cyclical=self.feature_config.get('include_cyclical', False),
            include_time_features=self.feature_config.get('include_time_features', True),
        )
    
    def fit(
        self,
        df: pd.DataFrame,
        target_var: str,
        multivariate_vars: Optional[List[str]] = None,
        validation_data: Optional[pd.DataFrame] = None,
    ):
        """
        Train the XGBoost model(s).
        """
        self.target_var = target_var
        
        # Store observed physical range for value clamping during prediction
        valid_vals = df[target_var].dropna()
        if len(valid_vals) > 0:
            self.observed_min = valid_vals.min()
            self.observed_max = valid_vals.max()
            logger.info(f"Observed range for {target_var}: [{self.observed_min:.2f}, {self.observed_max:.2f}]")
            
            # Store Climatology baseline from training data for long-gap recovery
            group_cols = [df.index.dayofyear, df.index.hour]
            self.climatology = df.groupby(group_cols)[target_var].mean().to_dict()
            logger.info(f"Stored Climatology baseline ({len(self.climatology)} anchor points) for {target_var}")
        
        # Train Forward Model
        logger.info(f"Training Forward Model for {target_var}...")
        self.models['fwd'], self.feature_columns, self.feature_engineers['fwd'], self.is_residual_fwd = self._fit_single(
            df, target_var, multivariate_vars, validation_data
        )
        
        # Train Backward Model if requested
        if self.bidirectional:
            logger.info(f"Training Backward Model for {target_var}...")
            # Reverse DataFrames
            df_rev = df.iloc[::-1]
            val_rev = validation_data.iloc[::-1] if validation_data is not None else None
            
            self.models['bwd'], _, self.feature_engineers['bwd'], self.is_residual_bwd = self._fit_single(
                df_rev, target_var, multivariate_vars, val_rev
            )
            
        return self

    def _fit_single(self, df, target_var, multivariate_vars, validation_data):
        """Internal fit for one direction with Auto-Supervised Residual Learning."""
        
        feat_eng = self._create_feature_engineer()
        
        # Original data features (no artificial masking)
        df_masked = df.copy()
        
        # 2. COMPUTE LINEAR BASE (Removed, not used for absolute prediction)
        # linear_base = df_masked[target_var].interpolate(method='time', limit_direction='both').bfill().ffill()
        
        # 3. RESIDUAL TARGET (Removed, not used for absolute prediction)
        # y_residual = df[target_var] - linear_base
        
        # 4. FEATURE ENGINEERING
        # CRITICAL BUG FIX (Empty Dataset): Features must be generated on a sequence where the
        # TARGET variable retains its NaNs. If we pre-fill the target with climatology before lag generation,
        # the model learns to simply map climatology to the original signal, destroying autoregressive learning.
        df_for_features = df_masked.copy()
        
        # We only interpolate MULTIVARIATE predictor variables so that a single missing exogenous 
        # sensor doesn't drop the entire training row. The target MUST remain with NaNs.
        if multivariate_vars:
            for var in multivariate_vars:
                if var in df_for_features.columns and var != target_var:
                    df_for_features[var] = df_for_features[var].interpolate(method='time').bfill().ffill()

        df_features = feat_eng.fit_transform(
            df_for_features, target_variable=target_var, multivariate_vars=multivariate_vars
        )
        
        if multivariate_vars:
            for var in multivariate_vars:
                if var in df.columns and var != target_var:
                    if var not in df_features.columns:
                        df_features[var] = df_for_features[var]
        
        # 5. EXTRACT TRAINING SET
        is_residual_mode = False  # DO NOT use residual mode, use absolute predicting
        # Simply extract all valid rows for training
        valid_mask = df[target_var].notna().values
        X = df_features.loc[valid_mask].copy()
        y = df.loc[valid_mask, target_var].copy()
        
        if target_var in X.columns:
            X = X.drop(columns=[target_var])
            
        complete_mask = y.notna()
        X = X[complete_mask]
        y = y[complete_mask]
        
        feature_columns = X.columns.tolist()
        
        # Create Model
        model = xgb.XGBRegressor(**self.xgb_params)
        
        try:
            model.fit(X, y)
        except xgb.core.XGBoostError as e:
            if "gpu" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"GPU training failed: {e}. Falling back to CPU.")
                # Update params to CPU
                cpu_params = self.xgb_params.copy()
                cpu_params['device'] = 'cpu'
                cpu_params['tree_method'] = 'hist'
                
                # Re-init model
                model = xgb.XGBRegressor(**cpu_params)
                model.fit(X, y)
                
                # Update self params so subsequent models use CPU directly
                self.xgb_params = cpu_params
            else:
                raise e
        
        return model, feature_columns, feat_eng, is_residual_mode

    def predict(self, df: pd.DataFrame, multivariate_vars: Optional[List[str]] = None) -> pd.Series:
        """
        Predict (impute) missing values.
        """
        if 'fwd' not in self.models:
             raise ValueError("Model not trained. Call fit() first.")
             
        # Predict Forward
        logger.info("Predicting Forward...")
        pred_fwd = self._predict_single(
            self.models['fwd'], self.feature_engineers['fwd'], 
            df, multivariate_vars, False # Always predict absolute
        )
        
        if not self.bidirectional:
            return pred_fwd
            
        # Predict Backward
        logger.info("Predicting Backward...")
        df_rev = df.iloc[::-1]
        pred_bwd_rev = self._predict_single(
            self.models['bwd'], self.feature_engineers['bwd'],
            df_rev, multivariate_vars, False # Always predict absolute
        )
        pred_bwd = pred_bwd_rev.iloc[::-1]
        
        # Ensemble (Simple Average)
        result = (pred_fwd + pred_bwd) / 2.0
        
        return result

    def _predict_single(self, model, feat_eng, df, multivariate_vars, is_residual_mode):
        """Internal predict for one direction with high-performance recursive loop."""
        target = self.target_var
        result = df[target].copy()
        
        gap_positions = np.where(result.isna())[0]
        if len(gap_positions) == 0:
            return result
            
        logger.debug(f"Total gaps to fill: {len(gap_positions)}")
        
        # Pre-allocate features dataframe
        df_tmp = df.copy()
        
        # CLIMATOLOGY BASELINE instead of linear interpolation
        # Uses mean per (dayofyear, hour) — physically realistic baseline
        group_cols = [df.index.dayofyear, df.index.hour]
        if self.climatology is not None:
            # Use the more robust climatology from the Training set if available
            climatology_series = df.index.to_series().apply(lambda x: self.climatology.get((x.dayofyear, x.hour), np.nan))
        else:
            # Fallback to local climatology if not available (shouldn't happen with trained model)
            climatology_series = df[target].groupby(group_cols).transform('mean')
            
        s_interp = df[target].fillna(climatology_series).interpolate(method='time', limit_direction='both').bfill().ffill()
        df_tmp[target] = df[target].fillna(s_interp) # Fill target with climatology for feature generation
        
        # CRITICAL: Interpolate multivariate variables to prevent NaN propagation
        if multivariate_vars:
             for col in multivariate_vars:
                 if col in df_tmp.columns and col != target:
                     # Fill with mean to avoid 0-biasing the model
                     df_tmp[col] = df_tmp[col].interpolate(method='time').bfill().ffill().fillna(df_tmp[col].mean()).fillna(0)

        # Fill NaNs generated by rolling/shifting (if any remain from feature engineering)
        df_tmp = df_tmp.bfill().fillna(0)

        # This gives us a base DataFrame with all feature columns initialized
        all_feats = feat_eng.transform(
            df_tmp, target_variable=target, multivariate_vars=multivariate_vars
        )
        
        # Validate that all expected columns are present
        missing_cols = [c for c in self.feature_columns if c not in all_feats.columns]
        if missing_cols:
             for col in missing_cols:
                 if col in df.columns:
                     # Use mean for padding instead of 0
                     all_feats[col] = df[col].fillna(df[col].mean()).fillna(0)
                 else:
                     all_feats[col] = 0.0

        # Working arrays for speed
        y_values = df[target].values.copy()
        y_values[np.isnan(y_values)] = s_interp[np.isnan(y_values)] # Use s_interp as initial guess for recursive update
        
        times = df.index
        
        # Identify dynamic features
        lags = feat_eng.lags
        windows = getattr(feat_eng, 'rolling_windows', []) # Get rolling_windows safely
        stats = getattr(feat_eng, 'rolling_stats', []) # Get rolling_stats safely
        
        # Cache column names for faster access
        lag_cols = [f"{target}_lag_{l}" for l in lags]
        base_vals = s_interp.values # Array for robust fallback blending
        consecutive_gap_idx = 0
        
        # Optimize prediction for single-row recursive logic
        # Single-row inference is heavily latency-bound by PCIe transfers. Predict on CPU avoids
        # "mismatched devices" warning when input is a pandas CPU DataFrame, and is usually faster for 1 row.
        try:
            model.set_params(device='cpu')
            if hasattr(model, 'get_booster'):
                model.get_booster().set_param({'device': 'cpu'})
        except Exception as e:
            logger.debug(f"Failed to enforce CPU mid-prediction: {e}")
        
        # Main Optimized Recursive Loop
        for pos_idx, pos in enumerate(gap_positions):
            idx = times[pos]
            
            # Dynamic Regression to Baseline based on depth into gap
            # Determines whether we are in a micro/short/medium gap vs a long/gigant gap
            # For consecutive NaNs, we decay the prediction towards the simple physical baseline
            if pos_idx > 0 and gap_positions[pos_idx - 1] == pos - 1:
                consecutive_gap_idx += 1
            else:
                consecutive_gap_idx = 0
            
            # 1. Update Lags (O(n_lags))
            for lag, col in zip(lags, lag_cols):
                if col in self.feature_columns:
                    # Previous values (can be observed or previously predicted)
                    val = y_values[pos - lag] if pos >= lag else np.nan
                    all_feats.at[idx, col] = val
            
            # 2. Update Rolling Stats (O(n_windows * window_size))
            if windows and stats: # Only execute if rolling features are configured
                for window in windows:
                    start_w = max(0, pos - window + 1)
                    window_data = y_values[start_w : pos + 1] # Includes pos (which is NaN)
                    
                    # Pandas handles NaNs in rolling windows by ignoring them (min_periods=1)
                    for stat in stats:
                        col = f"{target}_roll_{stat}_{window}"
                        if col in self.feature_columns:
                            if stat == 'mean':
                                val = np.nanmean(window_data) if not np.all(np.isnan(window_data)) else np.nan
                            elif stat == 'std':
                                val = np.nanstd(window_data) if not np.all(np.isnan(window_data)) else np.nan
                            else:
                                val = np.nan
                            all_feats.at[idx, col] = val
            
            # 3. Update Time Since Observation (if configured)
            col_ts = f"{target}_time_since_obs"
            if col_ts in self.feature_columns:
                if pos > 0:
                    # Relative to previous row (which was either updated or observed)
                    prev_val = all_feats.iloc[pos-1].get(col_ts, 0)
                    all_feats.at[idx, col_ts] = prev_val + 1
            
            # Removed: 3.5 Update Physics Features (Diff/Accel) & Gap Distance
            
            # 4. Predict
            X_row = all_feats.iloc[[pos]][self.feature_columns]
            
            # Force inference parameters to use CPU specifically during predict
            try:
                pred_absolute = model.predict(X_row, iteration_range=(0, model.best_iteration if hasattr(model, 'best_iteration') else 0))[0]
            except:
                pred_absolute = model.predict(X_row)[0]
            
            # The model predicts residual, so actual value is base + residual (Removed, now predicts absolute)
            # if is_residual_mode:
            #     pred_absolute = base_vals[pos] + pred
            # else:
            #     pred_absolute = pred
            
            # EXPOSURE BIAS MITIGATION: Dynamic Baseline Reversion
            # Deep into a gap, the autoregressive predictability drops to noise.
            # We decay the model's pure prediction towards the robust Climatology baseline.
            # CRITICAL FIX: Only start decaying after 48 steps (24 hours). Before that, trust XGBoost 100%.
            if consecutive_gap_idx <= 48:
                decay_factor = 1.0
            else:
                # lambda = 1/576 (576 steps = 12 days. At 12 days depth, e^-1 = 36% weight to XGBoost)
                decay_factor = np.exp(-(consecutive_gap_idx - 48) / 576.0)
                
            pred_absolute = (pred_absolute * decay_factor) + (base_vals[pos] * (1.0 - decay_factor))
            
            # VALUE CLAMPING: Prevent physically impossible predictions
            if self.observed_min is not None and self.observed_max is not None:
                obs_range = self.observed_max - self.observed_min
                pred_absolute = np.clip(pred_absolute, 
                                        self.observed_min - 0.05 * obs_range,
                                        self.observed_max + 0.05 * obs_range)
            
            # PHYSICAL GRADIENT CONSTRAINT: Prevent spikes (> 0.5 units per step)
            # This is critical for TEMP to avoid downward artifacts.
            if pos > 0:
                prev_val = y_values[pos - 1]
                max_delta = 0.5 # 0.5 degrees per 30 min is physically very large already
                pred_absolute = np.clip(pred_absolute, prev_val - max_delta, prev_val + max_delta)

            # 5. Commit to working array for next recursions
            y_values[pos] = pred_absolute
            result.at[idx] = pred_absolute
            
        return result

    def save(self, path: str):
        joblib.dump({
            'models': self.models,
            'feature_columns': self.feature_columns,
            'feature_config': self.feature_config,
            'xgb_params': self.xgb_params,
            'bidirectional': self.bidirectional,
            'target_var': self.target_var,
            'is_residual_fwd': self.is_residual_fwd,
            'is_residual_bwd': self.is_residual_bwd,
            'observed_min': self.observed_min,
            'observed_max': self.observed_max,
            'climatology': self.climatology,
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        imputer = cls(
            xgb_params=data['xgb_params'],
            feature_config=data['feature_config'],
            bidirectional=data.get('bidirectional', False)
        )
        imputer.models = data['models']
        # Restore feature engineers
        imputer.feature_engineers = {}
        if 'fwd' in imputer.models:
            imputer.feature_engineers['fwd'] = imputer._create_feature_engineer()
        if 'bwd' in imputer.models:
            imputer.feature_engineers['bwd'] = imputer._create_feature_engineer()
            
        imputer.feature_columns = data['feature_columns']
        imputer.target_var = data['target_var']
        imputer.is_residual_fwd = data.get('is_residual_fwd', False)
        imputer.is_residual_bwd = data.get('is_residual_bwd', False)
        imputer.observed_min = data.get('observed_min', None)
        imputer.observed_max = data.get('observed_max', None)
        imputer.climatology = data.get('climatology', None)
        return imputer
