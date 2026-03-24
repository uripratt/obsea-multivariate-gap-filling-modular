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
        self.climatology_profile = None
        self.target_var_stl = target_var
        
        df_mod = df.copy()
        try:
            logger.info(f"  [{self.__class__.__name__}] Extracting Climatology components for pure residual learning...")
            
            group_cols = [df_mod.index.dayofyear, df_mod.index.hour]
            self.climatology_profile = df_mod[target_var].groupby(group_cols).mean()
            
            climatology = df_mod[target_var].groupby(group_cols).transform('mean')
            climatology = climatology.interpolate(method='time', limit_direction='both').bfill().ffill()
            
            df_mod[target_var] = df_mod[target_var] - climatology
            
            self.is_residual_mode = True
            logger.info(f"  [{self.__class__.__name__}] Target converted to anomaly.")
            
        except Exception as e:
            logger.error(f"  [{self.__class__.__name__}] Total residual failure: {e}. Training on absolutes.")
                
        return df_mod

    def apply_stl_reconstruction(self, df: pd.DataFrame, predicted_residuals: pd.Series) -> pd.Series:
        """
        Sum the stored STL components back to the predicted residuals to recreate the absolute values.
        """
        if not getattr(self, 'is_residual_mode', False) or self.climatology_profile is None:
            return predicted_residuals
            
        try:
            idx_multi = pd.MultiIndex.from_arrays([predicted_residuals.index.dayofyear, predicted_residuals.index.hour])
            global_mean = self.climatology_profile.mean()
            
            mapped_values = []
            for item in idx_multi:
                if item in self.climatology_profile.index:
                    mapped_values.append(self.climatology_profile.loc[item])
                else:
                    mapped_values.append(global_mean)
            
            base_reconstruction = pd.Series(mapped_values, index=predicted_residuals.index)
            base_reconstruction = base_reconstruction.interpolate(method='time', limit_direction='both').bfill().ffill()
            
            # Adding them reconstructs the absolute values
            return predicted_residuals + base_reconstruction

        except Exception as e:
            logger.warning(f"  [{self.__class__.__name__}] Climatology Reconstruction failed: {e}. Returning residuals.")
            return predicted_residuals
