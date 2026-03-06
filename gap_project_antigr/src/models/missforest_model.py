"""
MissForest-style imputation using Scikit-Learn's IterativeImputer.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging
from typing import List, Optional
from .stl_mixin import STLResidualMixin

logger = logging.getLogger(__name__)

class MissForestImputer(STLResidualMixin):
    """
    MissForest-style imputer utilizing IterativeImputer with RandomForest.
    """
    def __init__(self, max_iter=10, n_estimators=100, random_state=42):
        # CRITICAL FIX for CPU Exhaustion: Set n_jobs=1 instead of -1.
        self.rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, random_state=random_state)
        self.imputer = IterativeImputer(
            estimator=self.rf,
            max_iter=max_iter,
            random_state=random_state,
            verbose=0
        )
        self.feature_columns = None

    def fit(self, df: pd.DataFrame, target_var: str, multivariate_vars: Optional[List[str]] = None):
        """Fit the imputer on valid data using STL residual learning."""
        self.feature_columns = [target_var] + (multivariate_vars if multivariate_vars else [])
        
        # EXTRACT STL ANOMALY
        df_residuals = self.apply_stl_extraction(df.copy(), target_var, period=48)
        
        data = df_residuals[self.feature_columns].dropna()
        
        if data.empty:
            logger.warning("No complete data found for MissForest fit. Using all non-null target data.")
            data = df_residuals[[target_var]].dropna()
            self.feature_columns = [target_var]

        self.imputer.fit(df_residuals[self.feature_columns])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Impute missing values in the target variable."""
        if self.feature_columns is None:
            raise ValueError("Model not fitted.")
            
        data_imputed = self.imputer.transform(df[self.feature_columns])
        df_imputed = pd.DataFrame(data_imputed, columns=self.feature_columns, index=df.index)
        
        predicted_residuals = df_imputed[self.feature_columns[0]]
        
        # RECONSTRUCT ABSOLUTE
        predicted_absolute = self.apply_stl_reconstruction(df, predicted_residuals)
        
        return predicted_absolute
