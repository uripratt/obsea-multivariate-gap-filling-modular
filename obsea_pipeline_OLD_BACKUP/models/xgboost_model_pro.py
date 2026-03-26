"""
XGBoost-based gap filling model with Bi-Directional and Recursive capabilities (PRO Version).
Includes advanced rolling statistics, deeper trees, and residual learning.
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


class XGBoostProImputer:
    """
    XGBoost-based imputation using Bi-Directional Recursive strategy.
    PRO Version: Enhanced estimators, deep rolling stats, residual mode.
    """
    
    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        feature_config: Optional[Dict] = None,
        bidirectional: bool = True,
        max_gap_size: int = None,
    ):
        """
        Initialize XGBoost Pro imputer.
        """
        # Default XGBoost parameters (Aggressive GPU default)
        default_xgb_params = {
            'n_estimators': 1500,   # High precision
            'max_depth': 8,         # Deeper trees to capture complex relations
            'learning_rate': 0.01,  # Lower LR for better convergence
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'device': 'cuda',       # Try CUDA by default
            'tree_method': 'hist',  # Required for efficient GPU
        }
        
        self.xgb_params = {**default_xgb_params, **(xgb_params or {})}
        self.feature_config = feature_config or {}
        self.bidirectional = bidirectional
        
        # Models storage
        self.models = {} # 'fwd', 'bwd'
        self.feature_engineers = {} # 'fwd', 'bwd'
        self.stl_components = {} # 'fwd': (trend, seasonal), 'bwd': ...
        
        self.is_residual_fwd = False
        self.is_residual_bwd = False
        self.max_gap_size = max_gap_size
        
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
                logger.info(f"Scale-Aware Triggered: XGBoost Pro tracking MICRO lags for {self.max_gap_size} pts gap.")
            elif self.max_gap_size <= 144: # Medium (3 days)
                dynamic_lags = [1, 2, 4, 12, 24, 48, 72] # Mix of local and daily
                logger.info(f"Scale-Aware Triggered: XGBoost Pro tracking MEDIUM lags for {self.max_gap_size} pts gap.")
            else: # Long/Gigant
                # Skip immediate lags because they are just predicted noise deep inside a massive hole
                dynamic_lags = [24, 48, 96, 168, 336] # Deep daily and weekly structural lags
                logger.info(f"Scale-Aware Triggered: XGBoost Pro tracking DEEP lags for {self.max_gap_size} pts gap.")
                
            return TemporalFeatureEngineer(
                lags=dynamic_lags,
                rolling_windows=[12, 48, 168],
                rolling_stats=['mean', 'std', 'min', 'max'],
                include_cyclical=True,
                include_time_features=True  
            )
            
        if not self.feature_config: 
            return TemporalFeatureEngineer(
                lags=[1, 2, 4, 12, 24, 48, 168], # Deep lags up to 1 week (30 min freq -> 168 is 3.5 days, 336 is 1 week. Let's use up to 1 week)
                rolling_windows=[12, 48, 168], # 6h, 24h, 3.5d
                rolling_stats=['mean', 'std', 'min', 'max'],
                include_cyclical=True,
                include_time_features=True  
            )
        return TemporalFeatureEngineer(
            lags=self.feature_config.get('lags', [1, 2, 4, 12, 24, 48]),
            rolling_windows=self.feature_config.get('rolling_windows', [12, 48]),
            rolling_stats=self.feature_config.get('rolling_stats', ['mean', 'std']),
            include_cyclical=self.feature_config.get('include_cyclical', True),
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
        
        # Train Forward Model
        logger.info(f"Training Forward Model for {target_var}...")
        self._current_direction = 'fwd'
        self.models['fwd'], self.feature_columns, self.feature_engineers['fwd'], self.is_residual_fwd = self._fit_single(
            df, target_var, multivariate_vars, validation_data
        )
        
        # Train Backward Model if requested
        if self.bidirectional:
            logger.info(f"Training Backward Model for {target_var}...")
            self._current_direction = 'bwd'
            # Reverse DataFrames
            df_rev = df.iloc[::-1].copy()
            val_rev = validation_data.iloc[::-1].copy() if validation_data is not None else None
            
            self.models['bwd'], _, self.feature_engineers['bwd'], self.is_residual_bwd = self._fit_single(
                df_rev, target_var, multivariate_vars, val_rev
            )
            
        return self

    def _fit_single(self, df, target_var, multivariate_vars, validation_data):
        """Internal fit for one direction with Auto-Supervised Residual Learning."""
        
        feat_eng = self._create_feature_engineer()
        
        # Original data features (no artificial masking)
        df_masked = df.copy()
        
        # 2. INTERNAL STL RESIDUAL LEARNING
        try:
            from statsmodels.tsa.seasonal import STL
            # CLIMATOLOGY FALLBACK (Prevents long gaps from becoming straight lines)
            group_cols = [df_masked.index.dayofyear, df_masked.index.hour]
            climatology = df_masked[target_var].groupby(group_cols).transform('mean')
            # Fill any remaining NaNs in climatology with linear
            climatology = climatology.interpolate(method='time', limit_direction='both').bfill().ffill()
            
            base_signal = df_masked[target_var].fillna(climatology)
            
            stl = STL(base_signal, period=48, robust=True)
            res = stl.fit()
            trend_comp = res.trend
            seasonal_comp = res.seasonal
            y_residual = df[target_var] - trend_comp - seasonal_comp
            # Save components to use in reconstruction
            self.stl_components[self._current_direction] = (trend_comp, seasonal_comp)
            logger.info("  -> Successfully extracted STL components for pure residual learning.")
            is_residual_mode = True
        except Exception as e:
            logger.warning(f"  -> STL Extraction failed: {e}. Falling back to climatology residual.")
            group_cols = [df_masked.index.dayofyear, df_masked.index.hour]
            climatology = df_masked[target_var].groupby(group_cols).transform('mean')
            climatology = climatology.interpolate(method='time', limit_direction='both').bfill().ffill()
            
            base_signal = df_masked[target_var].fillna(climatology)
            y_residual = df[target_var] - base_signal
            self.stl_components[self._current_direction] = (base_signal, pd.Series(0, index=df.index))
            is_residual_mode = False
        
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
                    group_cols = [df_for_features.index.dayofyear, df_for_features.index.hour]
                    var_climatology = df_for_features[var].groupby(group_cols).transform('mean')
                    var_climatology = var_climatology.interpolate(method='time').bfill().ffill()
                    df_for_features[var] = df_for_features[var].fillna(var_climatology)
        
        if is_residual_mode:
            # CRITICAL FIX: To predict a residual target, the autoregressive features (lags, diffs)
            # must also be derived from the RESIDUAL. Otherwise the model cannot generalize the anomaly.
            df_for_features[target_var] = y_residual
        
        df_features = feat_eng.fit_transform(
            df_for_features, target_variable=target_var, multivariate_vars=multivariate_vars
        )
        
        if multivariate_vars:
            for var in multivariate_vars:
                if var in df.columns and var != target_var:
                    if var not in df_features.columns:
                        df_features[var] = df_for_features[var]
        
        # ROBUSTNESS: Add is_observed binary mask
        df_features[f'{target_var}_is_observed'] = df_masked[target_var].notna().astype(float)
        
        # 5. EXTRACT TRAINING SET (Now always uses residuals)
        valid_mask = df[target_var].notna()
        X = df_features.loc[valid_mask].copy()
        y = y_residual.loc[valid_mask].copy()
        
        if target_var in X.columns:
            X = X.drop(columns=[target_var])
            
        complete_mask = y.notna() # ALLOW NaNs IN X (XGBoost handles them natively)
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
                cpu_params = self.xgb_params.copy()
                cpu_params['device'] = 'cpu'
                cpu_params['tree_method'] = 'hist'
                model = xgb.XGBRegressor(**cpu_params)
                model.fit(X, y)
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
        self._current_direction = 'fwd'
        pred_fwd = self._predict_single(
            self.models['fwd'], self.feature_engineers['fwd'], 
            df, multivariate_vars, self.is_residual_fwd
        )
        
        if not self.bidirectional:
            return pred_fwd
            
        # Predict Backward
        logger.info("Predicting Backward...")
        self._current_direction = 'bwd'
        df_rev = df.iloc[::-1].copy()
        pred_bwd_rev = self._predict_single(
            self.models['bwd'], self.feature_engineers['bwd'],
            df_rev, multivariate_vars, self.is_residual_bwd
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
        
        
        df_tmp = df.copy()
        
        # 1. Base Reconstruction using saved STL components
        try:
            trend_comp, seasonal_comp = self.stl_components[self._current_direction]
            base_reconstruction = trend_comp + seasonal_comp
            # The base_reconstruction is aligned with the training data index orientation
            base_series = base_reconstruction.reindex(df_tmp.index)
        except KeyError:
            # Fallback if components aren't available for some reason
            group_cols = [df_tmp.index.dayofyear, df_tmp.index.hour]
            climatology = df_tmp[target].groupby(group_cols).transform('mean')
            climatology = climatology.interpolate(method='time', limit_direction='both').bfill().ffill()
            base_series = df_tmp[target].fillna(climatology)
            
        s_interp = df_tmp[target].fillna(base_series)
        
        # When creating features, pretend gaps are filled with base_series
        df_tmp[target] = s_interp
        
        if is_residual_mode:
            # CRITICAL FIX: Generate features from the RESIDUAL space, mapping the fit() behavior
            df_tmp[target] = df_tmp[target] - base_series
        
        if multivariate_vars:
             for col in multivariate_vars:
                 if col in df_tmp.columns and col != target:
                     group_cols = [df_tmp.index.dayofyear, df_tmp.index.hour]
                     var_climatology = df_tmp[col].groupby(group_cols).transform('mean')
                     var_climatology = var_climatology.interpolate(method='time').bfill().ffill()
                     df_tmp[col] = df_tmp[col].fillna(var_climatology).fillna(0)
                     
        # Re-add Physical Derivatives (Inertia) - from old robust spec
        diff1 = df_tmp[target].diff()
        df_tmp[f'{target}_diff1'] = diff1.shift(1).fillna(0)
        df_tmp[f'{target}_diff2'] = diff1.diff().shift(1).fillna(0)
        df_tmp['gap_distance'] = 1.0

        df_tmp[f'{target}_is_observed'] = df[target].notna().astype(float)
        
        all_feats = feat_eng.transform(
            df_tmp, target_variable=target, multivariate_vars=multivariate_vars
        )
        
        missing_cols = [c for c in self.feature_columns if c not in all_feats.columns]
        if missing_cols:
             for col in missing_cols:
                 if col in df.columns:
                     all_feats[col] = df[col].fillna(0)
                 else:
                     all_feats[col] = 0.0

        # Ensure base_vals aligns perfectly with the current dataframe length
        if len(base_series) != len(df):
            # If lengths mismatch (e.g. valid data subset), fallback to simple interpolation for this pass
            base_vals = s_interp.values
        else:
            base_vals = base_series.values 

        # Working arrays for speed
        if is_residual_mode:
            y_values = df[target].values.copy() - base_vals
            y_values[np.isnan(y_values)] = 0.0 # Initial guess for pure residual is exactly 0
        else:
            y_values = df[target].values.copy()
            y_values[np.isnan(y_values)] = s_interp.values[np.isnan(y_values)]
        
        times = df.index
        
        lags = getattr(feat_eng, 'lags', [])
        windows = getattr(feat_eng, 'rolling_windows', [])
        stats = getattr(feat_eng, 'rolling_stats', [])
        
        lag_cols = [f"{target}_lag_{l}" for l in lags]
        
        try:
            model.set_params(device='cpu')
            if hasattr(model, 'get_booster'):
                model.get_booster().set_param({'device': 'cpu'})
        except Exception as e:
            pass
        
        consecutive_gap_idx = 0
        
        for pos_idx, pos in enumerate(gap_positions):
            idx = times[pos]
            
            # Dynamic Regression to Baseline based on depth into gap
            if pos_idx > 0 and gap_positions[pos_idx - 1] == pos - 1:
                consecutive_gap_idx += 1
            else:
                consecutive_gap_idx = 0
            
            # Update Lags 
            for lag, col in zip(lags, lag_cols):
                if col in self.feature_columns:
                    val = y_values[pos - lag] if pos >= lag else np.nan
                    all_feats.at[idx, col] = val
            
            # CRITICAL FIX (Rolling Stats Error Propagation):
            # Do NOT update rolling stats dynamically within the gap using our own predictions.
            # If a prediction has a tiny error, the rolling mean magnifies it infinitely.
            # Instead, we just keep the rolling stats frozen to their value at the START of the gap.
            # The XGBoost model naturally handles this missing dynamic information better than pure noise.
            if windows and stats and pos > 0:
                for window in windows:
                    for stat in stats:
                        col = f"{target}_roll_{stat}_{window}"
                        if col in self.feature_columns:
                            # Keep the value from the previous step (freezing it)
                            all_feats.at[idx, col] = all_feats.iloc[pos-1].get(col, np.nan)
            
            # Update Time Since Observation
            col_ts = f"{target}_time_since_obs"
            if col_ts in self.feature_columns and pos > 0:
                prev_val = all_feats.iloc[pos-1].get(col_ts, 0)
                all_feats.at[idx, col_ts] = prev_val + 1
            
            # Update diffs
            col_d1 = f"{target}_diff1"
            col_d2 = f"{target}_diff2"
            if col_d1 in self.feature_columns and pos >= 2:
                all_feats.at[idx, col_d1] = y_values[pos-1] - y_values[pos-2]
            if col_d2 in self.feature_columns and pos >= 3:
                all_feats.at[idx, col_d2] = y_values[pos-1] - (2 * y_values[pos-2]) + y_values[pos-3]
                
            if 'gap_distance' in self.feature_columns and pos > 0:
                 prev_dist = all_feats.iloc[pos-1]['gap_distance']
                 all_feats.at[idx, 'gap_distance'] = prev_dist + 1.0

            # Predict
            X_row = all_feats.iloc[[pos]][self.feature_columns]
            
            try:
                pred = model.predict(X_row, iteration_range=(0, model.best_iteration if hasattr(model, 'best_iteration') else 0))[0]
            except:
                pred = model.predict(X_row)[0]
            
            # EXPOSURE BIAS MITIGATION: Dynamic Residual Decay
            # At 576 steps (12 days), the autoregressive residual is mostly noise.
            # We decay it towards 0, gracefully merging into the pure STL baseline signal.
            # CRITICAL FIX: Grace period of 48 steps (24 hours) where we trust the model 100%.
            if consecutive_gap_idx <= 48:
                decay_factor = 1.0
            else:
                decay_factor = np.exp(-(consecutive_gap_idx - 48) / 576.0)
                
            pred = pred * decay_factor
            
            if is_residual_mode:
                pred_absolute = base_vals[pos] + pred
            else:
                pred_absolute = pred
            
            # VALUE CLAMPING: Prevent physically impossible predictions
            if self.observed_min is not None and self.observed_max is not None:
                obs_range = self.observed_max - self.observed_min
                pred_absolute = np.clip(pred_absolute, 
                                        self.observed_min - 0.1 * obs_range,
                                        self.observed_max + 0.1 * obs_range)
            
            # Store residual in y_values so next lag correctly pulls residual
            y_values[pos] = pred_absolute - base_vals[pos] if is_residual_mode else pred_absolute
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
            'stl_components': self.stl_components,
            'observed_min': self.observed_min,
            'observed_max': self.observed_max,
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        imputer = cls(
            xgb_params=data['xgb_params'],
            feature_config=data['feature_config'],
            bidirectional=data.get('bidirectional', True)
        )
        imputer.models = data['models']
        imputer.feature_engineers = {}
        if 'fwd' in imputer.models:
            imputer.feature_engineers['fwd'] = imputer._create_feature_engineer()
        if 'bwd' in imputer.models:
            imputer.feature_engineers['bwd'] = imputer._create_feature_engineer()
            
        imputer.feature_columns = data['feature_columns']
        imputer.target_var = data['target_var']
        imputer.is_residual_fwd = data.get('is_residual_fwd', False)
        imputer.is_residual_bwd = data.get('is_residual_bwd', False)
        imputer.stl_components = data.get('stl_components', {})
        imputer.observed_min = data.get('observed_min', None)
        imputer.observed_max = data.get('observed_max', None)
        return imputer
