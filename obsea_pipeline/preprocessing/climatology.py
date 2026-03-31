import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger("climatology")

def calculate_climatology(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    Computes historical average for each (day_of_year, hour, minute).
    Uses the provided dataframe (ideally the 15-year Golden Dataset).
    """
    logger.info(f"Calculating climatology for {len(variables)} variables...")
    
    # Extract temporal components
    df_clim = df[variables].copy()
    df_clim['day_of_year'] = df_clim.index.dayofyear
    df_clim['hour'] = df_clim.index.hour
    df_clim['minute'] = df_clim.index.minute
    
    # Group by the composite temporal index
    climatology = df_clim.groupby(['day_of_year', 'hour', 'minute'])[variables].mean()
    
    logger.info(f"Climatology calculated. Found {len(climatology)} unique time bins.")
    return climatology

def apply_climatology_feature(df: pd.DataFrame, climatology: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    Adds '_CLIM' features to the target dataframe by matching temporal indices.
    """
    logger.info("Injecting climatological signals as features...")
    
    # Save index name to restore it later
    index_name = df.index.name if df.index.name else 'index'
    
    # Create temporary keys for merging
    df_target = df.copy()
    df_target['day_of_year'] = df_target.index.dayofyear
    df_target['hour'] = df_target.index.hour
    df_target['minute'] = df_target.index.minute
    
    # Join climatology
    df_merged = df_target.reset_index().merge(
        climatology, 
        on=['day_of_year', 'hour', 'minute'],
        how='left',
        suffixes=('', '_CLIM')
    ).set_index(index_name)
    
    # Cleanup keys
    df_merged.drop(columns=['day_of_year', 'hour', 'minute'], inplace=True)
    
    return df_merged

def generate_golden_climatology(golden_csv_path: str, output_path: str, variables: list):
    """
    Higher-level utility to pre-compute and save the climatology map.
    """
    if not Path(golden_csv_path).exists():
        logger.error(f"Golden dataset not found at {golden_csv_path}")
        return
        
    df_gold = pd.read_csv(golden_csv_path, index_col=0, parse_dates=True)
    climatology = calculate_climatology(df_gold, variables)
    climatology.to_csv(output_path)
    logger.info(f"Golden Climatology saved to {output_path}")
