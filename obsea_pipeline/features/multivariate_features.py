"""
Multivariate feature engineering.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


def create_interaction_features(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    interactions: List[str] = ['product', 'ratio', 'diff'],
) -> pd.DataFrame:
    """
    Create interaction features between two variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    var1 : str
        First variable
    var2 : str
        Second variable
    interactions : list of str
        Types of interactions: 'product', 'ratio', 'diff'
        
    Returns
    -------
    pd.DataFrame
        Dataframe with interaction features added
    """
    df_with_interactions = df.copy()
    
    if 'product' in interactions:
        df_with_interactions[f'{var1}_x_{var2}'] = df[var1] * df[var2]
    
    if 'ratio' in interactions:
        # Avoid division by zero
        df_with_interactions[f'{var1}_div_{var2}'] = df[var1] / (df[var2] + 1e-8)
    
    if 'diff' in interactions:
        df_with_interactions[f'{var1}_minus_{var2}'] = df[var1] - df[var2]
    
    logger.debug(f"Created {len(interactions)} interaction features between {var1} and {var2}")
    
    return df_with_interactions


def create_oceanographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific oceanographic features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with TEMP, PSAL columns
        
    Returns
    -------
    pd.DataFrame
        Dataframe with oceanographic features
    """
    df_with_ocean = df.copy()
    
    # Potential density approximation (simplified)
    # ρ ≈ 1000 + 0.8 * S - 0.2 * T (very simplified!)
    if 'TEMP' in df.columns and 'PSAL' in df.columns:
        temp = df['TEMP'].fillna(0)
        psal = df['PSAL'].fillna(0)
        
        df_with_ocean['density_approx'] = 1000 + 0.8 * psal - 0.2 * temp
        
        logger.debug("Created approximate density feature")
    
    # Temperature-salinity gradient (change rate)
    if 'TEMP' in df.columns:
        df_with_ocean['temp_gradient'] = df['TEMP'].diff()
    
    if 'PSAL' in df.columns:
        df_with_ocean['psal_gradient'] = df['PSAL'].diff()
    
    return df_with_ocean


class MultivariateFeatureEngineer:
    """Multivariate feature engineering."""
    
    def __init__(
        self,
        create_interactions: bool = True,
        create_domain_features: bool = True,
    ):
        """
        Initialize feature engineer.
        
        Parameters
        ----------
        create_interactions : bool
            Create interaction features
        create_domain_features : bool
            Create domain-specific features
        """
        self.create_interactions = create_interactions
        self.create_domain_features = create_domain_features
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        multivariate_vars: List[str],
    ) -> pd.DataFrame:
        """
        Create multivariate features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        multivariate_vars : list of str
            Variables to use for feature engineering
            
        Returns
        -------
        pd.DataFrame
            Dataframe with multivariate features
        """
        df_features = df.copy()
        
        # Interaction features (selective pairs)
        if self.create_interactions:
            # TEMP × PSAL
            if 'TEMP' in multivariate_vars and 'PSAL' in multivariate_vars:
                df_features = create_interaction_features(
                    df_features, 'TEMP', 'PSAL', ['product', 'diff']
                )
        
        # Domain-specific features
        if self.create_domain_features:
            df_features = create_oceanographic_features(df_features)
        
        n_features = len(df_features.columns) - len(df.columns)
        logger.info(f"Created {n_features} multivariate features")
        
        return df_features
