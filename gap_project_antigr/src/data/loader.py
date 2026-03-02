"""
Data loading utilities for OBSEA time series.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def load_obsea_data(
    file_path: str,
    parse_dates: bool = True,
    date_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OBSEA CTD data from CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file
    parse_dates : bool
        Whether to parse the first column as datetime
    date_column : str, optional
        Name of the date column (if None, uses first column)
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe with datetime index
    """
    logger.info(f"Loading data from {file_path}")
    
    # Load CSV
    df = pd.read_csv(
        file_path,
        index_col=0 if date_column is None else date_column,
        parse_dates=[0] if parse_dates else False,
    )
    
    # Ensure datetime index
    if parse_dates:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    
    logger.info(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df


def get_variable_stats(df: pd.DataFrame, variables: List[str]) -> Dict[str, Dict]:
    """
    Get statistics for each variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variables : list of str
        List of variable names
        
    Returns
    -------
    dict
        Statistics for each variable
    """
    stats = {}
    
    for var in variables:
        if var not in df.columns:
            logger.warning(f"Variable {var} not found in dataframe")
            continue
            
        var_data = df[var]
        stats[var] = {
            'count': var_data.count(),
            'missing': var_data.isna().sum(),
            'missing_pct': 100 * var_data.isna().sum() / len(var_data),
            'mean': var_data.mean(),
            'std': var_data.std(),
            'min': var_data.min(),
            'max': var_data.max(),
            'median': var_data.median(),
        }
    
    return stats


def print_data_summary(df: pd.DataFrame, variables: List[str]):
    """
    Print a summary of the loaded data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variables : list of str
        List of variables to summarize
    """
    print("\n" + "="*80)
    print("OBSEA DATA SUMMARY")
    print("="*80)
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Total records: {len(df)}")
    print(f"Time span: {(df.index[-1] - df.index[0]).days} days")
    
    # Inferred frequency
    if len(df) > 1:
        freq = pd.infer_freq(df.index)
        if freq:
            print(f"Inferred frequency: {freq}")
    
    print("\n" + "-"*80)
    print("VARIABLE STATISTICS")
    print("-"*80)
    
    stats = get_variable_stats(df, variables)
    
    for var, var_stats in stats.items():
        print(f"\n{var}:")
        print(f"  Valid: {var_stats['count']:,} ({100-var_stats['missing_pct']:.1f}%)")
        print(f"  Missing: {var_stats['missing']:,} ({var_stats['missing_pct']:.1f}%)")
        if var_stats['count'] > 0:
            print(f"  Range: [{var_stats['min']:.3f}, {var_stats['max']:.3f}]")
            print(f"  Mean ± Std: {var_stats['mean']:.3f} ± {var_stats['std']:.3f}")
    
    print("="*80 + "\n")


def filter_by_qc(
    df: pd.DataFrame,
    variable: str,
    qc_column: str,
    good_flags: List[int] = [0],
    bad_flags: List[int] = [9],
) -> pd.DataFrame:
    """
    Filter data based on QC flags.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variable : str
        Variable name to filter
    qc_column : str
        QC flag column name
    good_flags : list of int
        QC values considered good
    bad_flags : list of int
        QC values considered bad
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    if qc_column not in df.columns:
        logger.warning(f"QC column {qc_column} not found, skipping QC filtering")
        return df
    
    # Create mask for good data
    mask = df[qc_column].isin(good_flags)
    
    # Set bad data to NaN
    df_filtered = df.copy()
    df_filtered.loc[~mask, variable] = np.nan
    
    n_flagged = (~mask).sum()
    logger.info(f"Flagged {n_flagged} ({100*n_flagged/len(df):.2f}%) values in {variable} based on {qc_column}")
    
    return df_filtered


def resample_timeseries(
    df: pd.DataFrame,
    freq: str = '30min',
    method: str = 'nearest',
) -> pd.DataFrame:
    """
    Resample time series to a regular frequency.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index
    freq : str
        Target frequency (e.g., '30min', '1H')
    method : str
        Resampling method: 'nearest', 'linear', 'forward', 'backward'
        
    Returns
    -------
    pd.DataFrame
        Resampled dataframe
    """
    logger.info(f"Resampling to {freq} using {method} method")
    
    if method == 'nearest':
        df_resampled = df.resample(freq).nearest()
    elif method == 'linear':
        df_resampled = df.resample(freq).interpolate('linear')
    elif method == 'forward':
        df_resampled = df.resample(freq).ffill()
    elif method == 'backward':
        df_resampled = df.resample(freq).bfill()
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    logger.info(f"Resampled from {len(df)} to {len(df_resampled)} records")
    
    return df_resampled
