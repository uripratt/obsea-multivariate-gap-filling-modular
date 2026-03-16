import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class STLResidualMixin:
    """
    A Mixin class to bestow any model with native STL residual learning capabilities.
    When inherited, it provides methods to automatically extract and reconstruct STL components.
    """
    
    def apply_stl_extraction(self, df: pd.DataFrame, target_var: str, period: int = 48) -> pd.DataFrame:
        """
        Extract STL components and return a modified DataFrame where the target
        variable is purely the unpredictable anomaly.
        """
        self.is_residual_mode = False
        self.trend_comp = None
        self.seasonal_comp = None
        self.target_var_stl = target_var
        
        df_mod = df.copy()
        try:
            from statsmodels.tsa.seasonal import STL
            logger.info(f"  [{self.__class__.__name__}] Extracting STL components for pure residual learning...")
            
            # CLIMATOLOGY FALLBACK (Prevents long gaps from becoming straight lines)
            group_cols = [df_mod.index.dayofyear, df_mod.index.hour]
            climatology = df_mod[target_var].groupby(group_cols).transform('mean')
            climatology = climatology.interpolate(method='time', limit_direction='both').bfill().ffill()
            
            base_signal = df_mod[target_var].fillna(climatology)
            
            stl = STL(base_signal, period=period, robust=True)
            res = stl.fit()
            
            self.trend_comp = res.trend
            self.seasonal_comp = res.seasonal
            
            # Substract components from reality to isolate the anomaly.
            # Gaps (NaNs) in original data naturally stay as NaNs in the residual array
            df_mod[target_var] = df_mod[target_var] - self.trend_comp - self.seasonal_comp
            
            self.is_residual_mode = True
            logger.info(f"  [{self.__class__.__name__}] Target converted to anomaly.")
            
        except Exception as e:
            logger.warning(f"  [{self.__class__.__name__}] STL Extraction failed: {e}. Falling back to climatology residual.")
            try:
                group_cols = [df_mod.index.dayofyear, df_mod.index.hour]
                climatology = df_mod[target_var].groupby(group_cols).transform('mean')
                climatology = climatology.interpolate(method='time', limit_direction='both').bfill().ffill()
                
                base_signal = df_mod[target_var].fillna(climatology)
                self.trend_comp = base_signal
                self.seasonal_comp = pd.Series(0, index=df_mod.index)
                df_mod[target_var] = df_mod[target_var] - base_signal
                self.is_residual_mode = False
            except Exception as e_inner:
                logger.error(f"  [{self.__class__.__name__}] Total residual failure: {e_inner}. Training on absolutes.")
                
        return df_mod

    def apply_stl_reconstruction(self, df: pd.DataFrame, predicted_residuals: pd.Series) -> pd.Series:
        """
        Sum the stored STL components back to the predicted residuals to recreate the absolute values.
        """
        if not getattr(self, 'is_residual_mode', False) and self.trend_comp is None:
            return predicted_residuals
            
        try:
            base_reconstruction = self.trend_comp + self.seasonal_comp
            
            # Ensure bases match index length
            if len(base_reconstruction) != len(predicted_residuals):
                # We attempt to index match if possible, otherwise we fallback
                base_reconstruction = base_reconstruction.reindex(predicted_residuals.index)
            
            # Adding them reconstructs the absolute values
            return predicted_residuals + base_reconstruction

        except Exception as e:
            logger.warning(f"  [{self.__class__.__name__}] STL Reconstruction failed: {e}. Returning residuals.")
            return predicted_residuals
