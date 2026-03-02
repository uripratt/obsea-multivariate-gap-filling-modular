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
        
        # Models storage
        self.models = {} # 'fwd', 'bwd'
        self.feature_engineers = {} # 'fwd', 'bwd'
        
        self.is_residual_fwd = False
        self.is_residual_bwd = False
        
        self.feature_columns = None
        self.target_var = None
        
    def _create_feature_engineer(self):
        """Create a fresh feature engineer to prevent state cross-contamination."""
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
        
        # 1. AUTO-SUPERVISED MASKING
        valid_indices = np.where(df[target_var].notna())[0]
        train_masked_vals = df[target_var].values.copy()
        n_points = len(df)
        train_gap_mask_local = np.zeros(n_points, dtype=bool)
        
        np.random.seed(42)
        # We need enough gaps to train. Scale with data size.
        n_gaps = min(500, max(10, len(valid_indices) // 100))
        
        if n_gaps > 0 and len(valid_indices) > 500:
            # Pick random valid starting points
            starts_train = np.random.choice(valid_indices[:-100], n_gaps, replace=False)
            for start in starts_train:
                length = np.random.randint(6, 48) # 3 hours to 24 hours
                end = min(start + length, n_points)
                train_masked_vals[start:end] = np.nan
                train_gap_mask_local[start:end] = True
        
        # Only consider mask where we had ground truth
        train_gap_mask_local = train_gap_mask_local & df[target_var].notna().values
        
        df_masked = df.copy()
        df_masked[target_var] = train_masked_vals
        
        # 2. COMPUTE LINEAR BASE (Removed, not used for absolute prediction)
        # linear_base = df_masked[target_var].interpolate(method='time', limit_direction='both').bfill().ffill()
        
        # 3. RESIDUAL TARGET (Removed, not used for absolute prediction)
        # y_residual = df[target_var] - linear_base
        
        # 4. FEATURE ENGINEERING
        df_features = feat_eng.fit_transform(
            df_masked, target_variable=target_var, multivariate_vars=multivariate_vars
        )
        
        if multivariate_vars:
            for var in multivariate_vars:
                if var in df.columns and var != target_var:
                    if var not in df_features.columns:
                        df_features[var] = df[var]
        
        # ROBUSTNESS: Add is_observed binary mask
        df_features[f'{target_var}_is_observed'] = df_masked[target_var].notna().astype(float)
        
        # 5. EXTRACT TRAINING SET
        is_residual_mode = False  # DO NOT use residual mode, use absolute predicting
        
        # Use synthetic gaps for training so model learns to impute instead of 1-step-ahead prediction
        use_synthetic_gaps = train_gap_mask_local.sum() > 50
        
        if use_synthetic_gaps:
            logger.info("Training on synthetic gaps to simulate imputation.")
            X = df_features.loc[train_gap_mask_local].copy()
            y = df.loc[train_gap_mask_local, target_var].copy()
        else:
            logger.info("Insufficient synthetic gaps. Falling back to simple extracting.")
            valid_mask = df[target_var].notna()
            X = df_features.loc[valid_mask].copy()
            y = df.loc[valid_mask, target_var].copy()
        
        if target_var in X.columns:
            X = X.drop(columns=[target_var])
            
        complete_mask = X.notna().all(axis=1) & y.notna()
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
        
        # Linear base for fallback (Removed, not used for residual prediction)
        s_interp = df[target].interpolate(method='time', limit_direction='both').bfill().ffill()
        df_tmp[target] = df[target].fillna(s_interp) # Fill target with interpolated values for feature generation
        
        # CRITICAL: Interpolate multivariate variables to prevent NaN propagation
        if multivariate_vars:
             for col in multivariate_vars:
                 if col in df_tmp.columns and col != target:
                     df_tmp[col] = df_tmp[col].interpolate(method='time').bfill().ffill().fillna(0)

        # Removed: Rolling Statistics (Historical Context)
        # Removed: Physical Derivatives (Inertia)
        # Removed: Gap Distance (Time since last valid observation)
        
        # ROBUSTNESS: Add is_observed binary mask (1.0 = real data, 0.0 = gap)
        df_tmp[f'{target}_is_observed'] = df[target].notna().astype(float)
        
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
                     all_feats[col] = df[col].fillna(0)
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
        # base_vals = linear_base.values # Array for faster access (Removed)
        
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
        for pos in gap_positions:
            idx = times[pos]
            
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
            'is_residual_bwd': self.is_residual_bwd
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
        return imputer
