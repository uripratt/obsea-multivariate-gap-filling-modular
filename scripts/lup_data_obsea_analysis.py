#!/usr/bin/env python3
"""
OBSEA Multivariate Data Analysis Pipeline (LUP Version)
========================================================

A production-ready, scientifically rigorous pipeline for integrating CTD, 
AWAC currents, AWAC waves, and atmospheric data from OBSEA (Vilanova i la Geltrú, 
NW Mediterranean) into a unified multivariate dataset.

Features:
- QARTOD-style instrumental QC (range, spike, gradient, flatline checks)
- TEOS-10 physical oceanography validation
- Physically-coherent resampling (vectorial for currents, circular for directions)
- Comprehensive gap analysis
- Statistical analysis for gap-filling preparation

Author: Generated for OBSEA PhD Research
Date: 2026-01
"""

import warnings
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, track
from rich.table import Table
import sys
import logging
import datetime

# =============================================================================
# LOGGING SETUP (PRO-CLEAN LOGS)
# =============================================================================
class CleanStreamHandler(logging.StreamHandler):
    """Custom handler to strip Rich formatting for Terminal if needed, 
    but we'll prioritize file logging here."""
    pass

def setup_pipeline_logging(output_dir: Path):
    """Sets up a professional logging system that is clean in text editors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"pipeline_execution_{timestamp}.log"
    
    # File Handler (Clean text) - shared by all loggers
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure ROOT logger so all child loggers inherit handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Also configure our main logger explicitly
    logger = logging.getLogger("OBSEA_Pipeline")
    logger.setLevel(logging.INFO)
    
    return logger, log_file

# Initialize (will be updated in main with correct path)
log = logging.getLogger("OBSEA_Pipeline")
console = Console(force_terminal=True, soft_wrap=True) # Keep for table rendering if needed

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'data_paths': {
        'ctd': 'exported_data/RAW/OBSEA_CTD_30min_nc_RAW.csv',
        'currents': 'exported_data/RAW/OBSEA_AWAC_currents_30min_all_years_MEAN_RAW.csv',
        'waves': 'exported_data/RAW/OBSEA_AWAC_waves_full_nc_RAW.csv',
        'airmar': 'exported_data/RAW/OBSEA_Airmar_30min_nc_RAW.csv',
        'ctvg': 'exported_data/RAW/OBSEA_CTVG_Vantage_Pro2_30min_nc_RAW.csv',
    },
    'target_sampling_rate': '30min',
    'output_dir': 'output_lup',
    
    # Physical ranges for QC (NW Mediterranean - Vilanova i la Geltrú)
    'physical_ranges': {
        # Variable: (fail_min, fail_max, suspect_min, suspect_max)
        'TEMP': (5, 32, 10, 28),           # Temperature (°C)
        'PSAL': (30, 42, 36, 39.5),        # Salinity (PSU)
        'PRES': (0, 30, 15, 22),           # Pressure (dbar)
        'SVEL': (1400, 1600, 1490, 1550),  # Sound velocity (m/s)
        'CNDC': (0, 10, 3.5, 5.5),         # Conductivity (S/m)
        'CSPD': (0, 2.5, 0, 1.5),          # Current speed (m/s)
        'VHM0': (0, 10, 0, 5),             # Significant wave height (m)
        'VTPK': (1, 20, 3, 15),            # Peak wave period (s)
        'WSPD': (0, 50, 0, 25),            # Wind speed (m/s)
        'CAPH': (970, 1050, 990, 1035),    # Atmospheric pressure (hPa)
        'AIRT': (-5, 45, 5, 35),           # Air temperature (°C)
        'RELH': (0, 100, 20, 98),          # Relative humidity (%)
    },
    
    # Spike detection parameters (multiplier of rolling std)
    'spike_threshold_multiplier': 3.0,
    'spike_window': 5,  # number of samples
    
    # Gradient check parameters (max allowed rate of change per 30min)
    'gradient_thresholds': {
        'TEMP': 2.0,    # °C per 30min
        'PSAL': 0.5,    # PSU per 30min
        'PRES': 2.0,    # dbar per 30min
        'CSPD': 0.5,    # m/s per 30min
    },
    
    # Flatline detection
    'flatline_threshold': 5,  # consecutive identical values
    
    # Variables to extract from each instrument
    'variables': {
        'ctd': ['TEMP', 'PSAL', 'PRES', 'SVEL', 'CNDC'],
        'currents': ['CSPD', 'CDIR', 'UCUR', 'VCUR', 'ZCUR'],
        'waves': ['VHM0', 'VTPK', 'VTM02', 'VMDR', 'VPED'],
        'airmar': ['WSPD', 'WDIR', 'AIRT', 'CAPH'],
        'ctvg': ['WSPD', 'WDIR', 'AIRT', 'CAPH', 'RELH'],
    }
}

# Gap classification
GapInfo = namedtuple('GapInfo', ['start', 'end', 'duration_hours', 'category', 'variable'])

GAP_CATEGORIES = {
    'micro': (0, 1),                # < 1 hour
    'short': (1, 6),                # 1-6 hours
    'medium': (6, 72),              # 6h - 3 days
    'long': (72, 720),              # 3-30 days
    'extended': (720, 1440),        # 30-60 days
    'gigant':  (1440, float('inf')) # > 60 days
}

# =============================================================================
# GAP INTERPOLATION CONFIGURATION
# =============================================================================
# Boolean control: True = interpolate this category, False = leave as NaN

# -------------------------------------------------------------------------
# AUTOMATIC vs MANUAL METHOD SELECTION
# -------------------------------------------------------------------------
# True  = Automatically select the best method per gap category based on 
#         benchmark results (lowest RMSE). This is RECOMMENDED for production.
# False = Use manually defined methods in 'manual_methods' below.
USE_AUTOMATIC_METHOD_SELECTION = True

# Path to benchmark results (generated by benchmark_gap_filling)
BENCHMARK_RESULTS_PATH = Path(__file__).parent / 'output_lup' / 'tables' / 'interpolation_comparison.csv'

INTERPOLATION_CONFIG = {
    # Enable/disable interpolation per gap category
    'interpolate_micro': True,      # < 1 hour gaps
    'interpolate_short': True,      # 1-6 hour gaps  
    'interpolate_medium': True,     # 6h - 3 days
    'interpolate_long': True,       # 3-30 days
    'interpolate_extended': True,  # 30-60 days (off by default - too long)
    'interpolate_gigant': False,    # > 60 days (off by default - too long)
    
    # -------------------------------------------------------------------------
    # MANUAL METHOD ASSIGNMENT (used when USE_AUTOMATIC_METHOD_SELECTION = False)
    # -------------------------------------------------------------------------
    # Options: 'linear', 'time', 'splines', 'polynomial', 'var', 'varma', 'bilstm', 'xgboost'
    'manual_methods': {
        'micro': 'time',            # Simple time interpolation for tiny gaps
        'short': 'varma',           # Multivariate VARMA for short gaps
        'medium': 'xgboost',        # XGBoost (Bi-Directional) for medium gaps
        'long': 'xgboost',          # XGBoost (Bi-Directional) for long gaps
        'extended': 'xgboost',      # XGBoost (Bi-Directional) for extended gaps
        'gigant': 'xgboost',        # XGBoost (Bi-Directional) for gigantic gaps
    },
    
    # Variables that will be exported with ALL methods for visual comparison in the webapp
    # (Setting too many variables will increase processing time and file size)
    'comparison_variables': ['TEMP'], 
    
    # Bi-LSTM model configuration
    'bilstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'sequence_length': 720,       # 15 days context for better long-term memory
        'epochs': 100,                # Training with scheduler
        'batch_size': 32,             
        'learning_rate': 0.001,       # Lower LR for stable convergence
        'early_stopping_patience': 20, # More patience  
        'train_ratio': 0.6,           
        'val_ratio': 0.2,             
    }
}

# =============================================================================
# GLOBAL MODEL ACTIVATION (Master Switch)
# =============================================================================
# This dictates which methods are allowed to run (train) for PRODUCTION.
# If a method is disabled here, it will be skipped entirely, even if selected below.
ENABLE_MODELS = {
    'linear': True,
    'time': True,
    'splines': True,
    'polynomial': True,
    'varma': True,      # ENABLED: Good for short-medium multivariate gaps
    'bilstm': True,     # ENABLED: Best for long, complex gaps
    'xgboost': True,    # Recommended: Fast, robust, handles non-linearities
    'missforest': True, # NEW: Robust tabular imputation
    'saits': True,      # NEW: Transformer-based
    'imputeformer': True, # NEW: Low-rank Transformer
    'brits': True,      # NEW: RNN-based with temporal decay
}

# =============================================================================
# BENCHMARK MODEL ACTIVATION (For Scientific Comparison)
# =============================================================================
# All models enabled for benchmarking to provide complete scientific comparison.
# This is separate from production to allow comprehensive thesis evaluation.
BENCHMARK_MODELS = {
    'linear': True,
    'time': True,
    'splines': True,
    'polynomial': True,
    'varma': True,
    'bilstm': True,
    'xgboost': True,
    'missforest': True,
    'saits': True,
    'imputeformer': True,
    'brits': True,
}

# =============================================================================
# PER-MODEL OUTPUT DIRECTORIES
# =============================================================================
# Each model gets its own subdirectory for organized outputs
MODEL_OUTPUT_SUBDIRS = ['xgboost', 'bilstm', 'varma', 'benchmarks', 'gap_analysis']


# =============================================================================
# HIGH QUALITY VARIABLE FILTERING
# =============================================================================
# Variables with ≤25% gaps (identified via analyze_variable_quality.py)
# These are CTD + Meteorological + Derived oceanographic features with sufficient data

HIGH_QUALITY_VARIABLES = [
    # CTD Core Variables (18.93-21.92% gaps)
    'PSAL',           # Salinity
    'PSAL_ANOMALY',   # Salinity anomaly (deseasonalized)
    'TEMP',           # Temperature
    'TEMP_ANOMALY',   # Temperature anomaly (deseasonalized)
    'CNDC',           # Conductivity
    'SVEL',           # Sound velocity
    
    # Meteorological Variables (20.82-21.09% gaps)
    'LAND_RELH',      # Relative humidity (land station)
    'LAND_WSPD',      # Wind speed (land station)
    'LAND_WDIR',      # Wind direction (land station)
    'LAND_CAPH',      # Atmospheric pressure (land station)
    'LAND_AIRT',      # Air temperature (land station)
]

# Configuration flag: Set to True to filter only high-quality variables
USE_HIGH_QUALITY_FILTER = True



# =============================================================================
# DATA LOADING MODULE
# =============================================================================

def load_instrument_data(instrument: str, config: dict = CONFIG) -> pd.DataFrame:
    """
    Load data for a specific instrument from CSV file.
    
    Parameters
    ----------
    instrument : str
        Instrument name: 'ctd', 'currents', 'waves', 'airmar', 'ctvg'
    config : dict
        Configuration dictionary with data paths
        
    Returns
    -------
    pd.DataFrame
        DataFrame with TIME as index and relevant variables
    """
    base_path = Path(__file__).parent
    file_path = base_path / config['data_paths'][instrument]
    
    if not file_path.exists():
        console.print(f"[red]Warning: File not found: {file_path}[/red]")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Parse TIME column
    df['TIME'] = pd.to_datetime(df['TIME'])
    df.set_index('TIME', inplace=True)
    df.sort_index(inplace=True)
    
    # Get relevant variables for this instrument
    vars_to_keep = []
    for var in config['variables'].get(instrument, []):
        if var in df.columns:
            vars_to_keep.append(var)
            # Also keep QC and STD columns if available
            if f'{var}_QC' in df.columns:
                vars_to_keep.append(f'{var}_QC')
            if f'{var}_STD' in df.columns:
                vars_to_keep.append(f'{var}_STD')
    
    return df[vars_to_keep] if vars_to_keep else df


def load_all_data(config: dict = CONFIG) -> Dict[str, pd.DataFrame]:
    """
    Load all instrument data.
    
    Returns
    -------
    dict
        Dictionary with instrument name as key and DataFrame as value
    """
    data = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading data...", total=len(config['data_paths']))
        
        for instrument in config['data_paths'].keys():
            progress.update(task, description=f"Loading {instrument}...")
            df = load_instrument_data(instrument, config)
            if not df.empty:
                data[instrument] = df
                console.print(f"  ✓ {instrument}: {len(df):,} records, {df.index.min()} to {df.index.max()}")
            progress.advance(task)
    
    return data


# =============================================================================
# QC MODULE: INSTRUMENTAL CHECKS
# =============================================================================

def range_check(series: pd.Series, var_name: str, config: dict = CONFIG) -> pd.Series:
    """
    Perform range check on a variable.
    
    Returns QC flags: 1=good, 3=suspect, 4=fail, 9=missing
    """
    qc_flags = pd.Series(index=series.index, dtype=float)
    qc_flags[:] = 9  # Missing by default
    
    valid_mask = series.notna()
    qc_flags[valid_mask] = 1  # Start with good
    
    if var_name in config['physical_ranges']:
        fail_min, fail_max, suspect_min, suspect_max = config['physical_ranges'][var_name]
        
        # Fail flags
        fail_mask = valid_mask & ((series < fail_min) | (series > fail_max))
        qc_flags[fail_mask] = 4
        
        # Suspect flags (only for values not already failed)
        suspect_mask = valid_mask & (qc_flags != 4) & (
            (series < suspect_min) | (series > suspect_max)
        )
        qc_flags[suspect_mask] = 3
    
    return qc_flags


def spike_check(series: pd.Series, config: dict = CONFIG) -> pd.Series:
    """
    Detect spikes using rolling standard deviation.
    
    Returns QC flags: 1=good, 3=suspect (spike detected), 9=missing
    """
    qc_flags = pd.Series(index=series.index, dtype=float)
    qc_flags[:] = 9
    
    valid_mask = series.notna()
    qc_flags[valid_mask] = 1
    
    # Calculate rolling statistics
    window = config['spike_window']
    threshold = config['spike_threshold_multiplier']
    
    rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std()
    
    # Replace zero std with a small value to avoid division issues
    rolling_std = rolling_std.replace(0, np.nan).fillna(series.std() * 0.01)
    
    # Detect spikes
    deviation = np.abs(series - rolling_mean)
    spike_mask = valid_mask & (deviation > threshold * rolling_std)
    qc_flags[spike_mask] = 3
    
    return qc_flags


def gradient_check(series: pd.Series, var_name: str, config: dict = CONFIG) -> pd.Series:
    """
    Check rate of change (gradient) between consecutive values.
    
    Returns QC flags: 1=good, 3=suspect (excessive gradient), 9=missing
    """
    qc_flags = pd.Series(index=series.index, dtype=float)
    qc_flags[:] = 9
    
    valid_mask = series.notna()
    qc_flags[valid_mask] = 1
    
    if var_name in config['gradient_thresholds']:
        threshold = config['gradient_thresholds'][var_name]
        gradient = series.diff().abs()
        
        gradient_mask = valid_mask & (gradient > threshold)
        qc_flags[gradient_mask] = 3
    
    return qc_flags


def flatline_check(series: pd.Series, config: dict = CONFIG) -> pd.Series:
    """
    Detect flatlines (repeated identical values).
    
    Returns QC flags: 1=good, 3=suspect (flatline), 9=missing
    """
    qc_flags = pd.Series(index=series.index, dtype=float)
    qc_flags[:] = 9
    
    valid_mask = series.notna()
    qc_flags[valid_mask] = 1
    
    threshold = config['flatline_threshold']
    
    # Detect consecutive identical values
    diff = series.diff()
    is_same = (diff == 0) | diff.isna()
    
    # Count consecutive identical values
    consecutive_count = is_same.groupby((~is_same).cumsum()).cumsum()
    
    flatline_mask = valid_mask & (consecutive_count >= threshold)
    qc_flags[flatline_mask] = 3
    
    return qc_flags


def apply_instrumental_qc(df: pd.DataFrame, instrument: str, config: dict = CONFIG) -> pd.DataFrame:
    """
    Apply all instrumental QC checks to a DataFrame.
    
    Returns DataFrame with new QC columns: {VAR}_QC_INST
    """
    df_qc = df.copy()
    
    for var in config['variables'].get(instrument, []):
        if var not in df.columns:
            continue
        
        # Apply all checks
        range_qc = range_check(df[var], var, config)
        spike_qc = spike_check(df[var], config)
        gradient_qc = gradient_check(df[var], var, config)
        flatline_qc = flatline_check(df[var], config)
        
        # Combine QC flags (worst flag wins)
        combined_qc = pd.DataFrame({
            'range': range_qc,
            'spike': spike_qc,
            'gradient': gradient_qc,
            'flatline': flatline_qc
        }).max(axis=1)
        
        df_qc[f'{var}_QC_INST'] = combined_qc
    
    return df_qc


# =============================================================================
# PHASE 3: MATHEMATICAL PREPROCESSING MODULE
# =============================================================================
# Robust normalization, stationarization, and oceanographic feature extraction

# -----------------------------------------------------------------------------
# 3.1 Normalization Functions
# -----------------------------------------------------------------------------

def robust_scale(series: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    """
    Robust percentile-based scaling for outlier resistance.
    
    Scales data using IQR-like approach instead of mean/std.
    
    Parameters
    ----------
    series : pd.Series
        Input series to scale
    q_low, q_high : float
        Lower and upper percentiles for scaling
        
    Returns
    -------
    pd.Series
        Scaled series with values roughly in [-1, 1] range
    """
    valid = series.dropna()
    if len(valid) == 0:
        return series
    
    low = valid.quantile(q_low)
    high = valid.quantile(q_high)
    median = valid.median()
    
    # Avoid division by zero
    iqr = high - low
    if iqr == 0:
        return series - median
    
    return (series - median) / iqr


def log_transform(series: pd.Series, offset: float = 1.0) -> pd.Series:
    """
    Log1p transform for positively skewed oceanographic variables.
    
    Suitable for: VHM0 (wave height), PRES (pressure), VTPK (wave period).
    
    Parameters
    ----------
    series : pd.Series
        Input series (must be non-negative)
    offset : float
        Offset to add before log (default 1.0 for log1p)
        
    Returns
    -------
    pd.Series
        Log-transformed series
    """
    # Ensure non-negative
    series_positive = series.clip(lower=0)
    return np.log1p(series_positive + offset - 1)


def compute_anomaly(df: pd.DataFrame, var: str, 
                   groupby: List[str] = ['dayofyear', 'hour']) -> pd.Series:
    """
    Compute climatological anomaly: T' = T - T̄(doy, hr)
    
    Removes seasonal and diurnal cycles to reveal anomalies.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    var : str
        Variable name to compute anomaly for
    groupby : List[str]
        Time components to group by (default: dayofyear + hour)
        
    Returns
    -------
    pd.Series
        Anomaly series (deviations from climatological mean)
    """
    if var not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    series = df[var].copy()
    
    # Create grouping columns based on index
    group_cols = {}
    if 'dayofyear' in groupby:
        group_cols['dayofyear'] = df.index.dayofyear
    if 'hour' in groupby:
        group_cols['hour'] = df.index.hour
    if 'month' in groupby:
        group_cols['month'] = df.index.month
    
    if not group_cols:
        return series - series.mean()
    
    # Create grouping DataFrame
    group_df = pd.DataFrame(group_cols, index=df.index)
    group_df[var] = series
    
    # Compute climatological mean
    climatology = group_df.groupby(list(group_cols.keys()))[var].transform('mean')
    
    # Anomaly = original - climatology
    anomaly = series - climatology
    
    return anomaly


# -----------------------------------------------------------------------------
# 3.2 Stationarization Functions
# -----------------------------------------------------------------------------

def stl_decompose(series: pd.Series, period: int = 48, 
                  robust: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    STL (Seasonal-Trend decomposition using LOESS) decomposition.
    
    Period of 48 = 24 hours at 30-minute resolution.
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    period : int
        Seasonal period (48 for daily at 30min)
    robust : bool
        Use robust fitting (resistant to outliers)
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        trend, seasonal, residual components
    """
    try:
        from statsmodels.tsa.seasonal import STL
        
        # STL requires no missing values - interpolate temporarily
        series_filled = series.interpolate(method='linear').bfill().ffill()
        
        if len(series_filled.dropna()) < period * 2:
            console.print(f"  [yellow]Insufficient data for STL decomposition[/yellow]")
            return series, pd.Series(0, index=series.index), pd.Series(0, index=series.index)
        
        # OOM FIX: Process in chunks (e.g., 2 years) to prevent memory exhaustion
        # 2 years at 30-min resolution = 2 * 365 * 48 = 35040 points
        chunk_size = 35040
        
        trends, seasonals, resids = [], [], []
        
        for i in range(0, len(series_filled), chunk_size):
            chunk = series_filled.iloc[i : i + chunk_size]
            
            # If the last chunk is too small, fallback to rolling mean
            if len(chunk) < period * 2:
                trend_c = chunk.rolling(window=period, center=True, min_periods=1).mean()
                seasonal_c = pd.Series(0, index=chunk.index)
                resid_c = chunk - trend_c
            else:
                stl = STL(chunk, period=period, robust=robust)
                result = stl.fit()
                trend_c = result.trend
                seasonal_c = result.seasonal
                resid_c = result.resid
                
            trends.append(trend_c)
            seasonals.append(seasonal_c)
            resids.append(resid_c)
            
        trend = pd.concat(trends)
        seasonal = pd.concat(seasonals)
        resid = pd.concat(resids)
        
        return trend, seasonal, resid
        
    except ImportError:
        console.print("  [yellow]statsmodels not available for STL decomposition[/yellow]")
        return series, pd.Series(0, index=series.index), pd.Series(0, index=series.index)
    except Exception as e:
        console.print(f"  [yellow]STL failed: {e}[/yellow]")
        return series, pd.Series(0, index=series.index), pd.Series(0, index=series.index)


def apply_differencing(series: pd.Series, order: int = 1) -> pd.Series:
    """
    Apply differencing for stationarity (AR/VAR models).
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    order : int
        Differencing order (1 or 2)
        
    Returns
    -------
    pd.Series
        Differenced series
    """
    result = series.copy()
    for _ in range(order):
        result = result.diff()
    return result


def check_stationarity(series: pd.Series, var_name: str) -> Dict[str, any]:
    """
    Perform Augmented Dickey-Fuller (ADF) test for stationarity.
    
    Returns
    -------
    dict
        Dictionary with p-value, test statistic, and interpretation
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        
        # Drop NaNs for the test
        series_clean = series.dropna()
        if len(series_clean) < 100:
            return {'error': 'Insufficient data'}
            
        # Run ADF test
        result = adfuller(series_clean, autolag='AIC')
        
        p_value = result[1]
        is_stationary = p_value < 0.05
        
        return {
            'p_value': p_value,
            'test_statistic': result[0],
            'is_stationary': is_stationary,
            'interpretation': "Stationary (p < 0.05)" if is_stationary else "Non-Stationary / Trend present (p >= 0.05)"
        }
        
    except Exception as e:
        return {'error': str(e)}


# -----------------------------------------------------------------------------
# 3.3 Oceanographic Feature Extraction
# -----------------------------------------------------------------------------

def compute_density_sigma(temp: pd.Series, sal: pd.Series, 
                          pres: Union[pd.Series, float] = 0) -> pd.Series:
    """
    Compute potential density anomaly σ_θ (sigma-theta) using TEOS-10.
    
    σ_θ = ρ(S, θ, 0) - 1000 kg/m³
    
    Parameters
    ----------
    temp : pd.Series
        In-situ temperature (°C)
    sal : pd.Series
        Practical salinity (PSU)
    pres : Union[pd.Series, float]
        Pressure (dbar), default 0 for surface
        
    Returns
    -------
    pd.Series
        σ_θ potential density anomaly (kg/m³), typically 25-30 for Mediterranean
    """
    try:
        import gsw
        
        # Convert practical salinity to absolute salinity (approximate)
        # For Mediterranean, longitude ~ 1.7°E, latitude ~ 41.2°N
        SA = gsw.SA_from_SP(sal.values, pres, 1.7, 41.2)
        
        # Convert in-situ temperature to conservative temperature
        CT = gsw.CT_from_t(SA, temp.values, pres)
        
        # Compute sigma0 (potential density anomaly referenced to 0 dbar)
        sigma0 = gsw.sigma0(SA, CT)
        
        return pd.Series(sigma0, index=temp.index, name='SIGMA0')
        
    except ImportError:
        console.print("  [yellow]gsw (TEOS-10) not installed, skipping density calculation[/yellow]")
        return pd.Series(np.nan, index=temp.index, name='SIGMA0')
    except Exception as e:
        console.print(f"  [yellow]Density calculation failed: {e}[/yellow]")
        return pd.Series(np.nan, index=temp.index, name='SIGMA0')


def compute_brunt_vaisala(temp: pd.Series, sal: pd.Series, 
                          pres: pd.Series, depth: float = 20.0) -> pd.Series:
    """
    Compute Brunt-Väisälä frequency N² (stratification strength).
    
    For a single-depth CTD, this estimates local stability using
    temporal gradients as proxy for vertical gradients.
    
    N² > 0 indicates stable stratification
    N² ≈ 0 indicates well-mixed conditions
    
    Parameters
    ----------
    temp, sal, pres : pd.Series
        CTD variables
    depth : float
        Approximate sensor depth (m)
        
    Returns
    -------
    pd.Series
        N² (s⁻²), typically 1e-5 to 1e-3 for ocean
    """
    try:
        import gsw
        
        # Compute density
        SA = gsw.SA_from_SP(sal.values, pres.values, 1.7, 41.2)
        CT = gsw.CT_from_t(SA, temp.values, pres.values)
        rho = gsw.rho(SA, CT, pres.values)
        
        # Temporal gradient as proxy for vertical structure
        # dρ/dt scaled by typical vertical mixing time
        drho_dt = pd.Series(rho, index=temp.index).diff()
        
        # Approximate N² using g/ρ * dρ/dz
        # Here we use temporal variation scaled to typical vertical scales
        g = 9.81
        rho_mean = np.nanmean(rho)
        
        # Scale factor: assume temporal variations over ~6h relate to ~10m vertical
        scale_factor = 10.0 / (6 * 3600)  # m/s
        
        N2 = (g / rho_mean) * drho_dt * scale_factor
        
        # Clip to physical range
        N2 = N2.clip(lower=0, upper=1e-2)
        
        return pd.Series(N2.values, index=temp.index, name='N2')
        
    except Exception as e:
        console.print(f"  [yellow]N² calculation failed: {e}[/yellow]")
        return pd.Series(np.nan, index=temp.index, name='N2')


def decompose_wind_uv(wspd: pd.Series, wdir: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Decompose wind speed and direction to zonal (U) and meridional (V) components.
    
    Convention: meteorological (direction FROM which wind blows)
    U positive = westerly (from west)
    V positive = southerly (from south)
    
    Parameters
    ----------
    wspd : pd.Series
        Wind speed (m/s)
    wdir : pd.Series
        Wind direction (degrees, meteorological convention)
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        wind_U (zonal), wind_V (meridional)
    """
    # Convert meteorological to mathematical convention
    # Meteorological: direction wind comes FROM
    # Mathematical: direction wind goes TO
    wdir_rad = np.deg2rad(wdir)
    
    # U = -wspd * sin(dir), V = -wspd * cos(dir)
    # Negative because we want "to" direction
    wind_u = -wspd * np.sin(wdir_rad)
    wind_v = -wspd * np.cos(wdir_rad)
    
    return (pd.Series(wind_u, index=wspd.index, name='WIND_U'),
            pd.Series(wind_v, index=wspd.index, name='WIND_V'))


def compute_wind_stress(wspd: pd.Series, rho_air: float = 1.225, 
                        Cd: float = 1.3e-3) -> pd.Series:
    """
    Compute wind stress magnitude τ = ρ_air * Cd * U².
    
    Parameters
    ----------
    wspd : pd.Series
        Wind speed at 10m (m/s)
    rho_air : float
        Air density (kg/m³), default 1.225
    Cd : float
        Drag coefficient, default 1.3e-3 (typical ocean value)
        
    Returns
    -------
    pd.Series
        Wind stress τ (N/m² = Pa)
    """
    tau = rho_air * Cd * wspd ** 2
    return pd.Series(tau, index=wspd.index, name='WIND_STRESS')


def compute_wave_energy(Hs: pd.Series, Tp: pd.Series, 
                        rho: float = 1025, g: float = 9.81) -> pd.Series:
    """
    Compute wave energy flux (power per unit crest length).
    
    E = (ρ * g² / 64π) * Hs² * Tp
    
    Parameters
    ----------
    Hs : pd.Series
        Significant wave height (m)
    Tp : pd.Series
        Peak wave period (s)
    rho : float
        Water density (kg/m³)
    g : float
        Gravitational acceleration (m/s²)
        
    Returns
    -------
    pd.Series
        Wave energy flux (kW/m)
    """
    # Wave power formula
    # P = (ρ * g² / 64π) * Hs² * Te ≈ 0.5 * Hs² * Te in kW/m
    coefficient = (rho * g**2) / (64 * np.pi)
    energy_flux = coefficient * Hs**2 * Tp / 1000  # Convert to kW/m
    
    return pd.Series(energy_flux, index=Hs.index, name='WAVE_ENERGY')


def compute_rms_currents(ucur: pd.Series, vcur: pd.Series) -> pd.Series:
    """
    Compute RMS (root mean square) current speed from U/V components.
    
    Parameters
    ----------
    ucur, vcur : pd.Series
        Eastward and northward current components (m/s)
        
    Returns
    -------
    pd.Series
        RMS current speed (m/s)
    """
    rms = np.sqrt(ucur**2 + vcur**2)
    return pd.Series(rms, index=ucur.index, name='CUR_RMS')


# -----------------------------------------------------------------------------
# 3.4 Integration Function
# -----------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame, 
                        compute_stl: bool = False) -> pd.DataFrame:
    """
    Add all Phase 3 derived features to the unified dataset.
    
    Features added:
    - SIGMA0: Potential density anomaly (σ_θ)
    - N2: Brunt-Väisälä frequency (stratification)
    - WIND_U, WIND_V: Zonal/meridional wind components
    - WIND_STRESS: Wind stress magnitude (τ)
    - WAVE_ENERGY: Wave energy flux
    - CUR_RMS: RMS current speed
    - *_ANOMALY: Climatological anomalies for key variables
    
    Parameters
    ----------
    df : pd.DataFrame
        Unified multivariate dataset
    compute_stl : bool
        Whether to add STL components (slow, optional)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional derived features
    """
    console.print("\n[bold cyan]Adding Phase 3 derived features...[/bold cyan]")
    df_out = df.copy()
    
    # --- Density and stratification ---
    if 'TEMP' in df.columns and 'PSAL' in df.columns:
        pres = df['PRES'] if 'PRES' in df.columns else 20.0
        
        console.print("  Computing σ_θ (potential density)...")
        df_out['SIGMA0'] = compute_density_sigma(df['TEMP'], df['PSAL'], pres)
        
        console.print("  Computing N² (stratification)...")
        df_out['N2'] = compute_brunt_vaisala(df['TEMP'], df['PSAL'], 
                                              df['PRES'] if 'PRES' in df.columns else pd.Series(20, index=df.index),
                                              depth=20.0)
    
    # --- Wind components and stress ---
    # Check for wind variables (AIR_ prefix from Airmar, or direct)
    wspd_col = 'AIR_WSPD' if 'AIR_WSPD' in df.columns else ('WSPD' if 'WSPD' in df.columns else None)
    wdir_col = 'AIR_WDIR' if 'AIR_WDIR' in df.columns else ('WDIR' if 'WDIR' in df.columns else None)
    
    if wspd_col and wdir_col:
        console.print("  Computing wind U/V components...")
        wind_u, wind_v = decompose_wind_uv(df[wspd_col], df[wdir_col])
        df_out['WIND_U'] = wind_u
        df_out['WIND_V'] = wind_v
        
        console.print("  Computing wind stress...")
        df_out['WIND_STRESS'] = compute_wind_stress(df[wspd_col])
    
    # --- Wave energy ---
    vhm0_col = 'WAV_VHM0' if 'WAV_VHM0' in df.columns else ('VHM0' if 'VHM0' in df.columns else None)
    vtpk_col = 'WAV_VTPK' if 'WAV_VTPK' in df.columns else ('VTPK' if 'VTPK' in df.columns else None)
    
    if vhm0_col and vtpk_col:
        console.print("  Computing wave energy flux...")
        df_out['WAVE_ENERGY'] = compute_wave_energy(df[vhm0_col], df[vtpk_col])
    
    # --- Current RMS ---
    ucur_col = 'CUR_UCUR' if 'CUR_UCUR' in df.columns else ('UCUR' if 'UCUR' in df.columns else None)
    vcur_col = 'CUR_VCUR' if 'CUR_VCUR' in df.columns else ('VCUR' if 'VCUR' in df.columns else None)
    
    if ucur_col and vcur_col:
        console.print("  Computing RMS currents...")
        df_out['CUR_RMS'] = compute_rms_currents(df[ucur_col], df[vcur_col])
    
    # --- Climatological anomalies ---
    console.print("  Computing climatological anomalies...")
    for var in ['TEMP', 'PSAL']:
        if var in df.columns:
            df_out[f'{var}_ANOMALY'] = compute_anomaly(df, var, groupby=['dayofyear', 'hour'])
    
    # --- STL decomposition (optional, slow) ---
    if compute_stl:
        console.print("  Computing STL decomposition and Long-Term Trends...")
        for var in ['TEMP', 'PSAL']:
            if var in df.columns:
                try:
                    trend, seasonal, resid = stl_decompose(df[var], period=48*30) # ~15 days seasonal component
                    df_out[f'{var}_TREND'] = trend
                    df_out[f'{var}_SEASONAL'] = seasonal
                    df_out[f'{var}_RESID'] = resid
                    
                    # Calculate long-term slope (degrees or PSU per year)
                    valid_trend = trend.dropna()
                    if len(valid_trend) > 100:
                        y = valid_trend.values
                        x = np.arange(len(y))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        # Scale to points per year (avg 17520 points per year at 30min)
                        slope_per_year = slope * 17520
                        console.print(f"    [dim]Long-term {var} Trend: {slope_per_year:+.4f} units/year (p={p_value:.4f})[/dim]")
                except Exception as e:
                    console.print(f"    [yellow]STL failed for {var}: {e}[/yellow]")

    # --- Stationarity Checks (ADF Test) ---
    console.print("  Performing stationarity tests (ADF)...")
    for var in ['TEMP', 'PSAL', 'SIGMA0']:
        if var in df_out.columns:
            res = check_stationarity(df_out[var], var)
            if 'error' not in res:
                color = "green" if res['is_stationary'] else "yellow"
                console.print(f"    [{color}]{var}: {res['interpretation']} (p={res['p_value']:.4f})[/{color}]")

    # Count new features
    new_cols = [c for c in df_out.columns if c not in df.columns]
    console.print(f"  [green]✓ Phase 3 Complete: Added {len(new_cols)} derived features[/green]")
    
    return df_out


# =============================================================================
# RESAMPLING MODULE
# =============================================================================


def circular_mean(angles_deg: pd.Series) -> float:
    """
    Calculate circular mean for angular quantities (wind/wave/current direction).
    """
    if angles_deg.isna().all():
        return np.nan
    
    rad = np.deg2rad(angles_deg.dropna())
    mean_angle = np.rad2deg(np.arctan2(np.nanmean(np.sin(rad)), np.nanmean(np.cos(rad))))
    return mean_angle % 360


def resample_variable(series: pd.Series, var_name: str, target_freq: str = '30min') -> pd.Series:
    """
    Resample a variable using physically appropriate method.
    """
    # Directional variables (circular mean)
    if var_name in ['WDIR', 'CDIR', 'VMDR', 'VPED']:
        return series.resample(target_freq).apply(circular_mean)
    
    # Wave period (median - robust to outliers)
    elif var_name in ['VTPK', 'VTM02', 'VTZA']:
        return series.resample(target_freq).median()
    
    # All other variables (arithmetic mean)
    else:
        return series.resample(target_freq).mean()


def resample_dataframe(df: pd.DataFrame, instrument: str, 
                       target_freq: str = '30min', config: dict = CONFIG) -> pd.DataFrame:
    """
    Resample all variables in a DataFrame using appropriate methods.
    """
    resampled_data = {}
    
    for col in df.columns:
        # Extract base variable name (without _QC, _STD suffixes)
        base_var = col.split('_')[0] if '_' in col else col
        
        if '_QC' in col or '_STD' in col:
            # For QC flags, take the worst (max) value
            # For STD, take the mean
            if '_QC' in col:
                resampled_data[col] = df[col].resample(target_freq).max()
            else:
                resampled_data[col] = df[col].resample(target_freq).mean()
        else:
            resampled_data[col] = resample_variable(df[col], base_var, target_freq)
    
    return pd.DataFrame(resampled_data)


# =============================================================================
# MULTIVARIATE INTEGRATION
# =============================================================================

def create_unified_dataset(data: Dict[str, pd.DataFrame], 
                          target_freq: str = '30min') -> pd.DataFrame:
    """
    Create a unified multivariate dataset from all instruments.
    
    Parameters
    ----------
    data : dict
        Dictionary of DataFrames by instrument
    target_freq : str
        Target resampling frequency
        
    Returns
    -------
    pd.DataFrame
        Unified dataset with all variables
    """
    console.print("\n[bold blue]Creating unified dataset...[/bold blue]")
    
    # Determine time range
    all_times = []
    for instrument, df in data.items():
        all_times.extend([df.index.min(), df.index.max()])
    
    start_time = min(all_times)
    end_time = max(all_times)
    
    console.print(f"  Time range: {start_time} to {end_time}")
    
    # Create master time index
    master_index = pd.date_range(start=start_time, end=end_time, freq=target_freq)
    unified_df = pd.DataFrame(index=master_index)
    unified_df.index.name = 'TIME'
    
    # Add prefix to avoid column collisions
    prefixes = {
        'ctd': '',           # Primary variables, no prefix
        'currents': 'CUR_',  # Current-related
        'waves': 'WAV_',     # Wave-related
        'airmar': 'AIR_',    # Offshore atmospheric
        'ctvg': 'LAND_',     # Land-based atmospheric
    }
    
    for instrument, df in data.items():
        prefix = prefixes.get(instrument, f'{instrument.upper()}_')
        
        # Resample to target frequency
        resampled_df = resample_dataframe(df, instrument, target_freq)
        
        # Add prefix to column names
        renamed_cols = {col: f'{prefix}{col}' if prefix else col 
                       for col in resampled_df.columns}
        resampled_df = resampled_df.rename(columns=renamed_cols)
        
        # Merge with unified dataset
        unified_df = unified_df.join(resampled_df, how='left')
        
        n_valid = resampled_df.notna().any(axis=1).sum()
        console.print(f"  ✓ {instrument}: {n_valid:,} timestamps with data")
    
    # Remove completely empty rows
    unified_df = unified_df.dropna(how='all')
    
    console.print(f"\n[green]Unified dataset: {len(unified_df):,} records, {len(unified_df.columns)} variables[/green]")
    
    return unified_df


# =============================================================================
# GAP ANALYSIS MODULE
# =============================================================================

def classify_gap_duration(duration_hours: float) -> str:
    """Classify a gap by its duration."""
    for category, (min_h, max_h) in GAP_CATEGORIES.items():
        if min_h <= duration_hours < max_h:
            return category
    return 'extended'


def detect_gaps(series: pd.Series, var_name: str) -> List[GapInfo]:
    """
    Detect and classify gaps in a time series.
    
    Returns list of GapInfo namedtuples.
    """
    gaps = []
    
    # Find gap boundaries
    is_missing = series.isna()
    gap_starts = is_missing & (~is_missing.shift(1, fill_value=False))
    gap_ends = is_missing & (~is_missing.shift(-1, fill_value=False))
    
    start_times = series.index[gap_starts]
    end_times = series.index[gap_ends]
    
    for start, end in zip(start_times, end_times):
        duration_hours = (end - start).total_seconds() / 3600
        category = classify_gap_duration(duration_hours)
        gaps.append(GapInfo(start, end, duration_hours, category, var_name))
    
    return gaps


def analyze_gaps(df: pd.DataFrame, variables: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Analyze gaps in all specified variables.
    
    Returns DataFrame with gap statistics.
    """
    if variables is None:
        # Select numeric columns without QC/STD suffixes
        variables = [col for col in df.columns 
                    if not any(suffix in col for suffix in ['_QC', '_STD'])]
    
    all_gaps = []
    for var in variables:
        if var in df.columns:
            gaps = detect_gaps(df[var], var)
            all_gaps.extend(gaps)
    
    if not all_gaps:
        return pd.DataFrame()
    
    gaps_df = pd.DataFrame([
        {'variable': g.variable, 'start': g.start, 'end': g.end,
         'duration_hours': g.duration_hours, 'category': g.category}
        for g in all_gaps
    ])
    
    return gaps_df


def create_gap_summary(gaps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics of gaps by variable and category.
    """
    if gaps_df.empty:
        return pd.DataFrame()
    
    summary = gaps_df.groupby(['variable', 'category']).agg(
        count=('duration_hours', 'count'),
        total_hours=('duration_hours', 'sum'),
        mean_hours=('duration_hours', 'mean'),
        max_hours=('duration_hours', 'max')
    ).round(2)
    
    return summary


# =============================================================================
# INTERPOLATION MODULE - SELECTIVE GAP FILLING
# =============================================================================

def create_canonical_index(df: pd.DataFrame, freq: str = '30min') -> pd.DatetimeIndex:
    """
    Create a canonical (regular) time index covering the full data range.
    
    This is essential for proper gap identification and interpolation.
    """
    start = df.index.min()
    end = df.index.max()
    return pd.date_range(start=start, end=end, freq=freq)


def reindex_to_canonical(df: pd.DataFrame, freq: str = '30min') -> pd.DataFrame:
    """
    Reindex DataFrame to a regular time grid.
    
    Missing timestamps will be filled with NaN.
    """
    canonical_idx = create_canonical_index(df, freq)
    return df.reindex(canonical_idx)


def get_gap_mask(gaps_df: pd.DataFrame, variable: str, 
                categories: List[str], full_index: pd.DatetimeIndex,
                freq: str = '30min') -> pd.Series:
    """
    Create a boolean mask for timestamps that fall within specified gap categories.
    
    Parameters
    ----------
    gaps_df : pd.DataFrame
        Gap information DataFrame
    variable : str
        Variable name to filter gaps for
    categories : List[str]
        Gap categories to include (e.g., ['micro', 'short'])
    full_index : pd.DatetimeIndex
        Full time index
    freq : str
        Frequency for timestamp generation
        
    Returns
    -------
    pd.Series
        Boolean mask where True = within specified gap category
    """
    mask = pd.Series(False, index=full_index)
    
    var_gaps = gaps_df[(gaps_df['variable'] == variable) & 
                       (gaps_df['category'].isin(categories))]
    
    for _, gap in var_gaps.iterrows():
        gap_timestamps = pd.date_range(gap['start'], gap['end'], freq=freq)
        mask.loc[mask.index.isin(gap_timestamps)] = True
    
    return mask


def interpolate_linear(series: pd.Series) -> pd.Series:
    """Linear interpolation."""
    return series.interpolate(method='linear')


def interpolate_time(series: pd.Series) -> pd.Series:
    """Time-based interpolation (accounts for irregular spacing)."""
    return series.interpolate(method='time')


def interpolate_spline(series: pd.Series, order: int = 3) -> pd.Series:
    """Cubic spline interpolation."""
    try:
        return series.interpolate(method='spline', order=order)
    except Exception:
        # Fallback to linear if spline fails
        return series.interpolate(method='linear')


def interpolate_polynomial(series: pd.Series, order: int = 2) -> pd.Series:
    """Polynomial interpolation."""
    try:
        return series.interpolate(method='polynomial', order=order)
    except Exception:
        return series.interpolate(method='linear')


def interpolate_var(df: pd.DataFrame, target_var: str, 
                   predictor_vars: Optional[List[str]] = None,
                   maxlags: int = 5) -> pd.Series:
    """
    Vector Autoregression (VAR) based interpolation.
    
    Uses multivariate relationships to impute missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with target and predictor variables
    target_var : str
        Variable to interpolate
    predictor_vars : List[str], optional
        Variables to use as predictors. If None, uses all numeric columns.
    maxlags : int
        Maximum lags for VAR model
        
    Returns
    -------
    pd.Series
        Interpolated series
    """
    try:
        from statsmodels.tsa.api import VAR
        
        # Select variables
        if predictor_vars is None:
            predictor_vars = [col for col in df.columns 
                            if col != target_var and 
                            not any(s in col for s in ['_QC', '_STD'])][:5]  # Limit to 5
        
        vars_to_use = [target_var] + [v for v in predictor_vars if v in df.columns]
        
        if len(vars_to_use) < 2:
            # Not enough variables for VAR, fallback to time
            return df[target_var].interpolate(method='time')
        
        # Prepare data - need complete cases for VAR fitting
        df_var = df[vars_to_use].copy()
        
        # First pass: linear interpolation to get complete data for fitting
        df_complete = df_var.interpolate(method='linear').dropna()
        
        if len(df_complete) < maxlags * 10:
            return df[target_var].interpolate(method='time')
        
        # Fit VAR model
        model = VAR(df_complete)
        results = model.fit(maxlags=maxlags)
        
        # Use fitted values for interpolation where original was NaN
        result_series = df[target_var].copy()
        fitted_values = results.fittedvalues[target_var]
        
        # Fill NaN with fitted values where available
        nan_mask = result_series.isna()
        for idx in result_series.index[nan_mask]:
            if idx in fitted_values.index:
                result_series.loc[idx] = fitted_values.loc[idx]
        
        # Fill remaining with linear interpolation
        result_series = result_series.interpolate(method='linear')
        
        return result_series
        
    except Exception as e:
        console.print(f"  [yellow]VAR failed for {target_var}: {e}, using time interpolation[/yellow]")
        return df[target_var].interpolate(method='time')


def interpolate_varma(df: pd.DataFrame, target_var: str,
                     predictor_vars: Optional[List[str]] = None,
                     order: Tuple[int, int] = (12, 0)) -> pd.Series:
    """
    VARMA (Vector ARMA) based interpolation.
    
    Uses PyTorch-based CudaVARMA for acceleration.
    Automatically selects most correlated variables as predictors.
    
    Parameters
    ----------
    order : Tuple[int, int]
        (p, q) order for VARMA - AR and MA orders
    """
    try:
        # Import custom CudaVARMA
        project_root = Path(__file__).resolve().parent.parent / "gap_project_antigr" 
        if str(project_root) not in sys.path:
             sys.path.append(str(project_root))
        
        from src.models.varma_model import VARMAImputer
        
        # Feature Selection: Tune most correlated variables
        if predictor_vars is None:
            numeric_df = df.select_dtypes(include=[np.number])
            # Filter out QC and STD and target itself
            candidates = [c for c in numeric_df.columns if c != target_var and '_QC' not in c and '_STD' not in c]
            
            if not candidates:
                return df[target_var].interpolate(method='time')

            # Compute correlation with target
            corrs = numeric_df[candidates].corrwith(numeric_df[target_var]).abs().sort_values(ascending=False)
            
            # Select top 5
            predictor_vars = corrs.head(5).index.tolist()
        
        # console.print(f"    [dim]VARMA Predictors for {target_var}: {predictor_vars}[/dim]")
        
        # Initialize Imputer
        imputer = VARMAImputer(
            p=order[0], 
            q=order[1], 
            batch_size=512, 
            epochs=50, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=True 
        )
        
        # Prepare Subset
        df_subset = df[[target_var] + predictor_vars].copy()
        
        # Fit
        # Note: If subset is too large, CUDA handles it, but maybe check mem?
        imputer.fit(df_subset, target_var, predictor_vars)
        
        # Predict
        result_series = imputer.predict_series(df_subset, target_var, predictor_vars)
        
        return result_series
        
    except Exception as e:
        console.print(f"  [yellow]VARMA failed for {target_var}: {e}, using time interpolation[/yellow]")
        # import traceback
        # traceback.print_exc()
        return df[target_var].interpolate(method='time')


def interpolate_bilstm(df: pd.DataFrame, target_var: str,
                       config_bilstm: dict = None,
                       predictor_vars: Optional[List[str]] = None) -> pd.Series:
    """
    Multivariate Bi-directional LSTM with train/val/test split and NaN protection.
    
    IMPROVEMENTS (2026-01-30):
    - Multivariate input (uses correlations between variables)
    - Train/Validation/Test split (60%/20%/20% temporal)
    - NO shuffle (preserves temporal order)
    - Early stopping on validation loss
    - NaN protection at every step
    - Residual learning (anomaly-based)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all variables
    target_var : str
        Variable to interpolate
    config_bilstm : dict, optional
        Configuration (hidden_size, epochs, etc.)
    predictor_vars : List[str], optional
        Predictor variables. If None, auto-selected by correlation.
        
    Returns
    -------
    pd.Series
        Interpolated series (NaN-safe)
    """
    try:
        # Import improved model
        project_root = Path(__file__).resolve().parent.parent / "gap_project_antigr"
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        
        from src.models.multivariate_lstm_model import MultivariateLSTMImputer
        
        # Use config or defaults
        if config_bilstm is None:
            config_bilstm = INTERPOLATION_CONFIG.get('bilstm', {})
        
        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        console.print(f"    [cyan]╔══ Multivariate Bi-LSTM (Improved) for {target_var} ══╗[/cyan]")
        console.print(f"    [dim]│ Device: {device}[/dim]")
        console.print(f"    [dim]│ Mode: Train/Val/Test + Early Stopping + NaN Protection[/dim]")
        console.print(f"    [dim]│ Shuffle: DISABLED (temporal order preserved)[/dim]")
        
        # =====================================================================
        # STEP 1: Auto-select predictors if not provided
        # =====================================================================
        if predictor_vars is None:
            console.print(f"    [dim]│ Auto-selecting predictors by correlation...[/dim]")
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            candidates = [col for col in numeric_cols 
                          if col != target_var and '_QC' not in col and '_STD' not in col]
            
            if candidates:
                # Compute correlations
                correlations = df[candidates].corrwith(df[target_var]).abs()
                correlations = correlations[correlations >= 0.3]  # Min correlation threshold
                correlations = correlations.sort_values(ascending=False)
                
                # Select top 4 predictors + target
                predictor_vars = [target_var] + correlations.head(4).index.tolist()
                log.info(f"    Auto-selected predictors: {predictor_vars}")
            else:
                predictor_vars = [target_var]
        else:
            # Ensure target is in predictors
            if target_var not in predictor_vars:
                predictor_vars = [target_var] + predictor_vars
        
        console.print(f"    [dim]│ Predictors ({len(predictor_vars)}): {', '.join(predictor_vars[:3])}{'...' if len(predictor_vars) > 3 else ''}[/dim]")
        
        # =====================================================================
        # STEP 2: Compute Anomaly for target (residual learning)
        # =====================================================================
        console.print(f"    [dim]│ Computing climatological anomaly for {target_var}...[/dim]")
        
        original_series = df[target_var].copy()
        # Compute climatology directly ensuring it's available for gaps
        group_cols = {
            'dayofyear': df.index.dayofyear,
            'hour': df.index.hour
        }
        group_temp = pd.DataFrame(group_cols, index=df.index)
        group_temp[target_var] = original_series
        
        # Calculate climatology (mean per day/hour) - this fills gaps with historical means
        climatology = group_temp.groupby(list(group_cols.keys()))[target_var].transform('mean')
        
        # Compute anomaly
        anomaly_series = original_series - climatology

        
        console.print(f"    [dim]│ ✓ Anomaly computed[/dim]")
        
        # =====================================================================
        # STEP 3: Prepare multivariate DataFrame
        # =====================================================================
        df_multi = df[predictor_vars].copy()
        df_multi[target_var] = anomaly_series  # Use anomaly for target
        
        # Check for sufficient data
        valid_count = df_multi.notna().all(axis=1).sum()
        min_required = config_bilstm.get('sequence_length', 96) * 50
        
        if valid_count < min_required:
            console.print(f"    [yellow]│ ✗ Insufficient data ({valid_count} rows), falling back[/yellow]")
            console.print(f"    [cyan]╚════════════════════════════════════════════════╝[/cyan]")
            return original_series.interpolate(method='time')
        
        console.print(f"    [dim]│ Complete records: {valid_count:,} / {len(df_multi):,}[/dim]")
        
        # =====================================================================
        # STEP 4: Train Multivariate Bi-LSTM
        # =====================================================================
        console.print(f"    [cyan]│ Training Multivariate Bi-LSTM...[/cyan]")
        
        imputer = MultivariateLSTMImputer(
            target_var=target_var,
            predictor_vars=predictor_vars,
            hidden_size=config_bilstm.get('hidden_size', 128),
            num_layers=config_bilstm.get('num_layers', 2),
            dropout=config_bilstm.get('dropout', 0.2),
            bidirectional=True,
            sequence_length=config_bilstm.get('sequence_length', 96),
            batch_size=config_bilstm.get('batch_size', 32),
            epochs=config_bilstm.get('epochs', 100),
            learning_rate=config_bilstm.get('learning_rate', 0.001),
            early_stopping_patience=config_bilstm.get('early_stopping_patience', 10),
            train_ratio=config_bilstm.get('train_ratio', 0.6),
            val_ratio=config_bilstm.get('val_ratio', 0.2),
            device=device,
        )
        
        # Train model
        metrics = imputer.fit(df_multi, verbose=False)
        
        console.print(f"    [dim]│ ✓ Training complete (stopped at epoch {metrics.best_epoch + 1})[/dim]")
        console.print(f"    [dim]│   Val loss: {metrics.best_val_loss:.6f}, Test loss: {metrics.test_loss:.6f}[/dim]")
        
        # =====================================================================
        # STEP 5: Predict on full series
        # =====================================================================
        console.print(f"    [dim]│ Predicting gaps in anomaly space...[/dim]")
        
        anomaly_filled = imputer.predict(df_multi)
        
        # DEBUG: Log prediction statistics
        log.info(f"    DEBUG: anomaly_filled - Total: {len(anomaly_filled)}, NaN: {anomaly_filled.isna().sum()}, Valid: {anomaly_filled.notna().sum()}")
        if anomaly_filled.notna().any():
            log.info(f"    DEBUG: anomaly_filled range: [{anomaly_filled.min():.4f}, {anomaly_filled.max():.4f}]")
        
        # NaN protection
        nan_count_pred = anomaly_filled.isna().sum()
        nan_count_orig = anomaly_series.isna().sum()
        
        log.info(f"    DEBUG: NaN count - Original anomaly: {nan_count_orig}, Predicted: {nan_count_pred}")
        
        if nan_count_pred > nan_count_orig:
            log.warning(f"    Prediction introduced {nan_count_pred - nan_count_orig} new NaNs, filling with original")
            # Keep original NaNs, don't add new ones
            anomaly_filled = anomaly_filled.fillna(anomaly_series)
        
        # =====================================================================
        # STEP 6: Reconstruct full signal - SMART GAP FILLING
        # =====================================================================
        console.print(f"    [dim]│ Reconstructing full signal (targeted gap filling)...[/dim]")
        
        # Start with original series
        result_series = original_series.copy()
        
        # Only fill where original was NaN AND model made a prediction
        gaps_to_fill = original_series.isna() & anomaly_filled.notna()
        
        if gaps_to_fill.any():
            # Initial Reconstruction: anomaly + climatology
            full_reconstruction = anomaly_filled + climatology
            
            # Apply BLENDING / BIAS CORRECTION
            # Iterate through contiguous gaps and correct them individually
            # We need to find the start and end of each gap block
            
            # 1. Identify gap groups in 'gaps_to_fill' mask
            # True=Gap, False=Observed
            gap_mask = gaps_to_fill.copy()
            gap_groups = (gap_mask != gap_mask.shift()).cumsum()
            gap_groups = gap_groups[gap_mask] # Keep only gap groups
            
            # 2. Iterate through each unique gap block
            for group_id in gap_groups.unique():
                try:
                    # Get indices for this gap block
                    gap_indices = gap_groups[gap_groups == group_id].index
                    
                    if len(gap_indices) == 0: continue
                    
                    gap_start_idx = df.index.get_loc(gap_indices[0])
                    gap_end_idx = df.index.get_loc(gap_indices[-1])
                    
                    # Ensure we have context (previous/next valid points)
                    # Note: We need to use integer location to easily get -1 and +1
                    # But we need timestamp for value lookup
                    
                    # Points just outside the gap
                    prev_valid_idx = gap_start_idx - 1
                    next_valid_idx = gap_end_idx + 1
                    
                    if prev_valid_idx < 0 or next_valid_idx >= len(df):
                        # Cannot blend at very beginning or very end of dataset
                        continue
                        
                    time_prev = df.index[prev_valid_idx]
                    time_next = df.index[next_valid_idx]
                    
                    # Get True Values (Original Series)
                    val_prev = original_series.iloc[prev_valid_idx]
                    val_next = original_series.iloc[next_valid_idx]
                    
                    if pd.isna(val_prev) or pd.isna(val_next):
                        # If neighbors are also NaN (shouldn't happen if we group correctly), skip correction
                        continue
                        
                    # Get Predicted Values (Reconstructed) at the gap edges
                    pred_series_slice = full_reconstruction.iloc[gap_start_idx : gap_end_idx + 1] # slice includes end in iloc? No, Python standard slice
                    # Wait, standard slice logic: [start:end] excludes end. 
                    # gap_indices covers the gap.
                    # slice needs to be gap_start_idx : gap_end_idx + 1
                    
                    pred_first = full_reconstruction.iloc[gap_start_idx]
                    pred_last = full_reconstruction.iloc[gap_end_idx]
                    
                    # Calculate BIAS (Offset)
                    bias_start = val_prev - pred_first
                    bias_end = val_next - pred_last
                    
                    # Create CORRECTION RAMP
                    n_gap = len(gap_indices)
                    correction = np.linspace(bias_start, bias_end, n_gap)
                    
                    # Apply correction to the full reconstruction for this segment
                    # Use .loc with specific timestamps to be safe
                    full_reconstruction.loc[gap_indices] += correction
                    
                except Exception as e:
                    # If correction fails for a specific gap, just ignore and leave raw prediction
                    # log.warning(f"Blending failed for gap group {group_id}: {e}")
                    pass
            
            # Finally assign the (now corrected) values to the result
            result_series.loc[gaps_to_fill] = full_reconstruction.loc[gaps_to_fill]
            
            n_filled = gaps_to_fill.sum()
            console.print(f"    [green]│ ✓ Filled {n_filled:,} gap points (with Blending)[/green]")
        else:
            console.print(f"    [yellow]│ ⚠ No gaps filled (model predictions were all NaN)[/yellow]")
        
        # Final NaN check
        final_nan_count = result_series.isna().sum()
        original_nan_count = original_series.isna().sum()
        filled_count = original_nan_count - final_nan_count
        
        log.info(f"    DEBUG: Final NaN count - Original: {original_nan_count}, Final: {final_nan_count}, Filled: {filled_count}")
        
        console.print(f"    [green]│ ✓ Multivariate residual learning complete ({filled_count:,} gaps filled)[/green]")
        console.print(f"    [cyan]╚════════════════════════════════════════════════╝[/cyan]")
        
        return result_series
        
    except Exception as e:
        console.print(f"    [red]│ ✗ Multivariate Bi-LSTM failed: {e}[/red]")
        console.print(f"    [cyan]╚════════════════════════════════════════════════╝[/cyan]")
        log.error(f"LSTM error: {e}", exc_info=True)
        return df[target_var].interpolate(method='time')


# =============================================================================
# AUTOMATIC METHOD SELECTION FROM BENCHMARK RESULTS
# =============================================================================

def get_best_methods_from_benchmark(benchmark_path: Path = None, 
                                     metric: str = 'RMSE') -> Dict[str, str]:
    """
    Determine the best interpolation method for each gap category based on benchmark results.
    
    This function reads the benchmark comparison CSV and selects the method with the
    lowest RMSE (or other metric) for each gap category. This enables automatic,
    data-driven method selection for production interpolation.
    
    Parameters
    ----------
    benchmark_path : Path, optional
        Path to the benchmark results CSV. Defaults to BENCHMARK_RESULTS_PATH.
    metric : str, optional
        Metric to minimize. Options: 'RMSE', 'MAE'. Default: 'RMSE'.
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping gap category to best method.
        Example: {'micro': 'varma', 'short': 'polynomial', 'medium': 'bilstm', ...}
    
    Notes
    -----
    If benchmark results are unavailable or a category has no results, falls back
    to the manual configuration in INTERPOLATION_CONFIG['manual_methods'].
    """
    if benchmark_path is None:
        benchmark_path = BENCHMARK_RESULTS_PATH
    
    # Default fallback methods
    fallback_methods = INTERPOLATION_CONFIG.get('manual_methods', {
        'micro': 'time',
        'short': 'time',
        'medium': 'time',
        'long': 'time',
        'extended': 'time',
        'gigant': 'time',
    })
    
    # Check if benchmark file exists
    if not benchmark_path.exists():
        log.warning(f"Benchmark results not found at {benchmark_path}")
        log.warning("Using manual method configuration as fallback.")
        console.print(f"[yellow]⚠ Benchmark results not found. Using manual methods.[/yellow]")
        return fallback_methods.copy()
    
    try:
        # Read benchmark results
        benchmark_df = pd.read_csv(benchmark_path)
        
        if benchmark_df.empty or 'Category' not in benchmark_df.columns or metric not in benchmark_df.columns:
            log.warning("Invalid benchmark file format. Using manual methods.")
            return fallback_methods.copy()
        
        # Ensure metric column is numeric
        benchmark_df[metric] = pd.to_numeric(benchmark_df[metric], errors='coerce')
        
        # Find best method per category (minimum RMSE or MAE)
        best_methods = {}
        
        for category in GAP_CATEGORIES.keys():
            category_data = benchmark_df[benchmark_df['Category'] == category]
            
            if category_data.empty:
                # No data for this category, use fallback
                best_methods[category] = fallback_methods.get(category, 'time')
                log.info(f"  {category}: No benchmark data. Using fallback: {best_methods[category]}")
                continue
            
            # Filter out NaN metrics
            valid_data = category_data.dropna(subset=[metric])
            
            if valid_data.empty:
                best_methods[category] = fallback_methods.get(category, 'time')
                log.info(f"  {category}: All metrics are NaN. Using fallback: {best_methods[category]}")
                continue
            
            # Find the row with minimum metric value
            best_row = valid_data.loc[valid_data[metric].idxmin()]
            best_method = best_row['Method']
            best_score = best_row[metric]
            
            best_methods[category] = best_method
            log.info(f"  {category}: Best method = {best_method} ({metric}={best_score:.4f})")
        
        console.print("\n[bold green]✓ Automatic Method Selection (from benchmark results):[/bold green]")
        for cat, method in best_methods.items():
            console.print(f"  {cat}: [cyan]{method}[/cyan]")
        
        return best_methods
        
    except Exception as e:
        log.error(f"Error reading benchmark results: {e}")
        console.print(f"[red]✗ Error reading benchmark results: {e}[/red]")
        console.print("[yellow]  Using manual method configuration as fallback.[/yellow]")
        return fallback_methods.copy()


def selective_interpolation(df: pd.DataFrame, gaps_df: pd.DataFrame,
                           interp_config: dict = INTERPOLATION_CONFIG,
                           freq: str = '30min') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform selective interpolation based on INTERPOLATION_CONFIG.
    
    Each gap category can be enabled/disabled and assigned a specific method.
    Uses boolean flags (interpolate_micro, interpolate_short, etc.) to control
    which categories to process.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potential gaps
    gaps_df : pd.DataFrame
        Gap information from analyze_gaps()
    interp_config : dict
        Interpolation configuration (see INTERPOLATION_CONFIG)
    freq : str
        Data frequency
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Interpolated DataFrame
        - Tracking DataFrame with columns: {var}_SOURCE ('observed', 'interpolated_{method}', 'missing')
    """
    # Determine which categories are enabled
    enabled_categories = []
    for cat in GAP_CATEGORIES.keys():
        flag_key = f'interpolate_{cat}'
        if interp_config.get(flag_key, False):
            enabled_categories.append(cat)
    
    # =========================================================================
    # METHOD SELECTION: AUTOMATIC vs MANUAL
    # =========================================================================
    if USE_AUTOMATIC_METHOD_SELECTION:
        console.print("\n[bold cyan]🔬 AUTOMATIC METHOD SELECTION MODE[/bold cyan]")
        console.print("  Reading benchmark results to select optimal methods per gap category...")
        log.info("Using AUTOMATIC method selection based on benchmark results")
        methods_per_cat = get_best_methods_from_benchmark()
    else:
        console.print("\n[bold yellow]📝 MANUAL METHOD SELECTION MODE[/bold yellow]")
        console.print("  Using manually configured methods from INTERPOLATION_CONFIG['manual_methods']")
        log.info("Using MANUAL method selection from configuration")
        methods_per_cat = interp_config.get('manual_methods', {})
    
    console.print(f"\n  [bold]Interpolation Configuration:[/bold]")
    for cat in GAP_CATEGORIES.keys():
        status = "✓ enabled" if cat in enabled_categories else "✗ disabled"
        method = methods_per_cat.get(cat, 'time')
        source = "auto" if USE_AUTOMATIC_METHOD_SELECTION else "manual"
        console.print(f"    {cat}: {status} (method: {method}) [{source}]")
    
    # Create canonical index
    canonical_idx = create_canonical_index(df, freq)
    df_canonical = df.reindex(canonical_idx)
    
    # Get variables to interpolate (exclude QC/STD columns)
    variables = [col for col in df.columns 
                if not any(suffix in col for suffix in ['_QC', '_STD'])]
    
    # Initialize tracking DataFrame
    tracking = pd.DataFrame(index=canonical_idx)
    
    # Cache for Bi-LSTM models (train once per variable)
    bilstm_cache = {}
    
    # Cache for XGBoost models (train once per variable)
    xgboost_cache = {}
    
    # Process each variable
    for var in track(variables, description="Processing variables..."):
        if var not in df_canonical.columns:
            continue
        
        series = df_canonical[var].copy()
        original_nan = series.isna()
        
        # Create tracking column
        tracking[f'{var}_SOURCE'] = 'observed'
        tracking.loc[original_nan, f'{var}_SOURCE'] = 'missing'
        
        # Process each enabled gap category with its assigned method
        for category in enabled_categories:
            method = methods_per_cat.get(category, 'time')

            # MASTER SWITCH: Enforce ENABLE_MODELS
            # If the selected method is disabled, fallback to 'time' (safest/fastest)
            if method in ENABLE_MODELS and not ENABLE_MODELS[method]:
                console.print(f"    [dim]⚠ Method '{method}' for '{category}' is DISABLED. Falling back to 'time'.[/dim]")
                method = 'time'
            
            # Get mask for this category
            cat_mask = get_gap_mask(gaps_df, var, [category], canonical_idx, freq)
            fill_mask = original_nan & cat_mask
            
            if not fill_mask.any():
                continue  # No gaps of this category for this variable
            
            # Get interpolated values using the assigned method
            if method == 'linear':
                interpolated = interpolate_linear(series)
            elif method == 'time':
                interpolated = interpolate_time(series)
            elif method == 'splines':
                interpolated = interpolate_spline(series)
            elif method == 'polynomial':
                interpolated = interpolate_polynomial(series)
            elif method == 'var':
                interpolated = interpolate_var(df_canonical, var)
            elif method == 'varma':
                interpolated = interpolate_varma(df_canonical, var)
            elif method == 'bilstm':
                # Use cached model if available, else train new
                if var not in bilstm_cache:
                    console.print(f"    [cyan]Training Bi-LSTM for {var}...[/cyan]")
                    bilstm_cache[var] = interpolate_bilstm(
                        df_canonical, var, 
                        config_bilstm=interp_config.get('bilstm', {})
                    )
                interpolated = bilstm_cache[var]
            elif method == 'xgboost':
                # Use cached model if available
                if var not in xgboost_cache:
                    # console.print(f"    [cyan]Training XGBoost (Bi-Directional) for {var}...[/cyan]")
                    # Train and Cache
                    _, trained_imputer = interpolate_xgboost(
                        df_canonical, var, 
                        predictor_vars=None # Auto-select
                    )
                    xgboost_cache[var] = trained_imputer

                # Use cached imputer for prediction
                if xgboost_cache[var] is not None:
                    interpolated, _ = interpolate_xgboost(
                        df_canonical, var,
                        imputer=xgboost_cache[var]
                    )
                else:
                     interpolated = interpolate_time(series)
            else:
                interpolated = interpolate_time(series)
            
            # Apply interpolated values only to this category's gaps
            series.loc[fill_mask] = interpolated.loc[fill_mask]
            
            # Mark as interpolated in tracking
            tracking.loc[fill_mask, f'{var}_SOURCE'] = f'interpolated_{method}'
        
        # Ensure disabled categories remain NaN
        disabled_categories = [c for c in GAP_CATEGORIES.keys() if c not in enabled_categories]
        for category in disabled_categories:
            preserve_mask = get_gap_mask(gaps_df, var, [category], canonical_idx, freq)
            series.loc[preserve_mask & original_nan] = np.nan
        
        df_canonical[var] = series
        
        # SPECIAL FEATURE: Multi-method comparison for research
        # If the variable is in comparison_variables, we generate columns for ALL methods
        if var in interp_config.get('comparison_variables', []):
            console.print(f"    [yellow]Generating multi-method comparison for {var}...[/yellow]")
            # Dynamically select only ENABLED methods for comparison
            comparison_methods = [m for m, active in ENABLE_MODELS.items() if active]
            
            for m_name in comparison_methods:
                col_name = f"{var}_{m_name.upper()}"
                
                # Avoid re-running if already done or cached
                if m_name == 'bilstm' and var in bilstm_cache:
                    df_canonical[col_name] = bilstm_cache[var]
                    continue
                
                try:
                    if m_name == 'linear': interp_val = interpolate_linear(df_canonical[var])
                    elif m_name == 'time': interp_val = interpolate_time(df_canonical[var])
                    elif m_name == 'splines': interp_val = interpolate_spline(df_canonical[var])
                    elif m_name == 'polynomial': interp_val = interpolate_polynomial(df_canonical[var])
                    elif m_name == 'varma': interp_val = interpolate_varma(df_canonical, var)
                    elif m_name == 'bilstm': 
                        interp_val = interpolate_bilstm(df_canonical, var, config_bilstm=interp_config.get('bilstm', {}))
                    elif m_name == 'xgboost':
                        interp_val = interpolate_xgboost(df_canonical, var)
                    else: continue
                    
                    df_canonical[col_name] = interp_val
                    console.print(f"      ✓ Added comparison column: {col_name}")
                except Exception as e:
                    console.print(f"      [red]✗ Method {m_name} failed for comparison: {e}[/red]")
        
        # Clear GPU memory after each variable to prevent CUDA memory exhaustion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Count interpolated points per method
    console.print("\n  [bold]Interpolation Summary:[/bold]")
    for col in tracking.columns:
        if '_SOURCE' in col:
            var = col.replace('_SOURCE', '')
            source_counts = tracking[col].value_counts()
            for source, count in source_counts.items():
                if source != 'observed':
                    console.print(f"    {var}: {count:,} points -> {source}")
    
    return df_canonical, tracking




def simulate_gaps(series: pd.Series, n_gaps: int, 
                  min_duration_pts: int, max_duration_pts: int) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate realistic gaps using block masking.
    
    Parameters
    ----------
    series : pd.Series
        Original data series
    n_gaps : int
        Number of gaps to create
    min_duration_pts : int
        Minimum gap duration in points
    max_duration_pts : int
        Maximum gap duration in points
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        - Series with artificial gaps
        - Boolean mask (True where gaps were created)
    """
    series_gapped = series.copy()
    mask = pd.Series(False, index=series.index)
    
    valid_indices = np.where(series.notna())[0]
    if len(valid_indices) == 0:
        return series_gapped, mask
        
    # Limit maximum attempts to avoid infinite loop
    max_attempts = n_gaps * 50
    attempts = 0
    gaps_created = 0
    
    while gaps_created < n_gaps and attempts < max_attempts:
        attempts += 1
        
        # Choose random duration
        duration = np.random.randint(min_duration_pts, max_duration_pts + 1)
        
        # solid block of data required
        if len(valid_indices) < duration:
            break
            
        start_idx_pos = np.random.randint(0, len(valid_indices) - duration)
        start_idx = valid_indices[start_idx_pos]
        end_idx = min(start_idx + duration, len(series))
        
        # Check if we are overwriting existing data (we should)
        # Verify indices are actual timestamps
        timestamp_start = series.index[start_idx]
        timestamp_end = series.index[min(end_idx, len(series)-1)]
        
        # QUALITY CHECK: Check if the segment is "too perfect" (artificial linear interpolation)
        # We calculate the second difference (acceleration). If it's consistently near zero, it's linear.
        segment_values = series.iloc[start_idx:start_idx+duration]
        
        # 1. Variance Check: If variance is 0, it's a flatline (artificial or stuck sensor) -> SKIP
        if segment_values.var() < 1e-6:
             continue

        if duration >= 3:
            # 2. Linearity Check: If 2nd derivative is ~0 everywhere, it's a linear interpolation -> SKIP
            # We allow some noise, but pure interpolation has very low 2nd diff
            # Calculate diff twice
            diff2 = np.diff(np.diff(segment_values.values))
            is_linear_artificial = np.all(np.abs(diff2) < 1e-4) # Threshold for float precision
            
            if is_linear_artificial:
                # Skip this segment, it's likely fake data
                continue

        # Apply gap
        series_gapped.iloc[start_idx:start_idx+duration] = np.nan
        mask.iloc[start_idx:start_idx+duration] = True
        gaps_created += 1
        
    if gaps_created < n_gaps:
        log.warning(f"Could only generate {gaps_created}/{n_gaps} gaps due to data quality constraints.")
        
    return series_gapped, mask




# =============================================================================
# TRAIN/INTERPOLATION HELPERS
# =============================================================================

def interpolate_xgboost(df: pd.DataFrame, target_var: str, predictor_vars: List[str] = None, imputer=None):
    """Wrapper for XGBoost Imputation with caching support."""
    sys.path.append(str(Path(__file__).parent.parent / "gap_project_antigr"))
    from src.models.xgboost_model import XGBoostImputer
    
    # Use existing imputer if provided
    if imputer is None:
        # Simple config
        imputer = XGBoostImputer(
            xgb_params={'n_estimators': 300, 'max_depth': 6},
            bidirectional=True  # ENABLED BI-DIRECTIONAL
        )
        should_fit = True
    else:
        should_fit = False
    
    try:
        if should_fit:
            imputer.fit(df, target_var, multivariate_vars=predictor_vars)
            
        prediction = imputer.predict(df, multivariate_vars=predictor_vars)
        return prediction, imputer
        
    except Exception as e:
        log.error(f"XGBoost failed: {e}")
        # Return fallback and None for imputer (to indicate failure)
        return df[target_var].interpolate(method='time'), None


def interpolate_missforest(df: pd.DataFrame, target_var: str, predictor_vars: List[str] = None):
    """Wrapper for MissForest Imputation."""
    from src.models.missforest_model import MissForestImputer
    imputer = MissForestImputer()
    try:
        imputer.fit(df, target_var, multivariate_vars=predictor_vars)
        prediction = imputer.predict(df)
        return prediction
    except Exception as e:
        log.error(f"MissForest failed: {e}")
        return df[target_var].interpolate(method='time')


def interpolate_saits(df: pd.DataFrame, target_var: str, predictor_vars: List[str] = None):
    """Wrapper for SAITS Imputation."""
    from src.models.saits_model import SAITSImputer
    n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
    imputer = SAITSImputer(n_steps=128, n_features=n_features, epochs=20)
    try:
        imputer.fit(df, target_var, multivariate_vars=predictor_vars)
        prediction = imputer.predict(df)
        return prediction
    except Exception as e:
        log.error(f"SAITS failed: {e}")
        return df[target_var].interpolate(method='time')


def interpolate_imputeformer(df: pd.DataFrame, target_var: str, predictor_vars: List[str] = None):
    """Wrapper for ImputeFormer Imputation."""
    from src.models.imputeformer_model import ImputeFormerImputer
    n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
    imputer = ImputeFormerImputer(n_steps=128, n_features=n_features, epochs=20)
    try:
        imputer.fit(df, target_var, multivariate_vars=predictor_vars)
        prediction = imputer.predict(df)
        return prediction
    except Exception as e:
        log.error(f"ImputeFormer failed: {e}")
        return df[target_var].interpolate(method='time')


def interpolate_brits(df: pd.DataFrame, target_var: str, predictor_vars: List[str] = None):
    """Wrapper for BRITS Imputation."""
    from src.models.brits_model import BRITSImputer
    n_features = 1 + (len(predictor_vars) if predictor_vars else 0)
    imputer = BRITSImputer(n_steps=128, n_features=n_features, epochs=20)
    try:
        imputer.fit(df, target_var, multivariate_vars=predictor_vars)
        prediction = imputer.predict(df)
        return prediction
    except Exception as e:
        log.error(f"BRITS failed: {e}")
        return df[target_var].interpolate(method='time')



def benchmark_gap_filling(df: pd.DataFrame, 
                         test_variable: str = 'TEMP',
                         gap_categories: List[str] = ['micro', 'short', 'medium', 'long']) -> pd.DataFrame:
    """
    Benchmark interpolation methods across different gap categories using block masking.
    """
    console.print(f"\n  [bold]Benchmarking gap filling on {test_variable}...[/bold]")
    
    if test_variable not in df.columns:
        console.print(f"  [red]Variable {test_variable} not found[/red]")
        return pd.DataFrame()

    # Define durations in points (assuming 30min data)
    cat_params = {
        'micro': {'n_gaps': 50, 'min_pts': 1, 'max_pts': 1}, 
        'short': {'n_gaps': 20, 'min_pts': 2, 'max_pts': 12},
        'medium': {'n_gaps': 5, 'min_pts': 12, 'max_pts': 144}, # 6h - 3d
        'long': {'n_gaps': 2, 'min_pts': 144, 'max_pts': 1440}, # 3d - 30d
        'extended': {'n_gaps': 2, 'min_pts': 1440, 'max_pts': 2880}, # 30-60d
        'gigant': {'n_gaps': 1, 'min_pts': 2880, 'max_pts': 5000}     # > 60d
    }
    
    results = []
    
    # Pre-calculate predictors for VAR/VARMA/XGB to save time
    numeric_df = df.select_dtypes(include=[np.number])
    candidates = [c for c in numeric_df.columns if c != test_variable and '_QC' not in c and '_STD' not in c]
    corrs = numeric_df[candidates].corrwith(numeric_df[test_variable]).abs().sort_values(ascending=False)
    predictor_vars = corrs.head(5).index.tolist()
    
    # Determine active methods - USE BENCHMARK_MODELS for scientific comparison
    methods = [m for m, enabled in BENCHMARK_MODELS.items() if enabled]
    
    for category in gap_categories:
        if category not in cat_params:
            continue
            
        params = cat_params[category]
        log.info(f"  [{category.upper()}] Simulating {params['n_gaps']} gaps ({params['min_pts']}-{params['max_pts']} pts)")
        
        # Create test set with artificial gaps
        df_test = df.copy()
        series_with_gaps, gap_mask = simulate_gaps(
            df[test_variable], 
            n_gaps=params['n_gaps'], 
            min_duration_pts=params['min_pts'], 
            max_duration_pts=params['max_pts']
        )
        
        df_test[test_variable] = series_with_gaps
        true_values = df[test_variable].loc[gap_mask]
        
        if len(true_values) == 0:
            log.warning(f"  [{category.upper()}] No gaps created (insufficient valid data)")
            continue
        
        log.info(f"  [{category.upper()}] Created {len(true_values)} gap points for testing")

        for method in methods:
            try:
                log.info(f"    [{category.upper()}|{method.upper()}] Starting interpolation...")
                
                # Interpolate
                if method == 'linear':
                    interpolated = interpolate_linear(df_test[test_variable])
                elif method == 'time':
                    interpolated = interpolate_time(df_test[test_variable])
                elif method == 'splines':
                    interpolated = interpolate_spline(df_test[test_variable])
                elif method == 'polynomial':
                    interpolated = interpolate_polynomial(df_test[test_variable])
                elif method == 'varma':
                    log.info(f"    [{category.upper()}|{method.upper()}] Training VARMA model...")
                    interpolated = interpolate_varma(df_test, test_variable, predictor_vars=predictor_vars)
                elif method == 'bilstm':
                    log.info(f"    [{category.upper()}|{method.upper()}] Training Bi-LSTM model...")
                    interpolated = interpolate_bilstm(df_test, test_variable, predictor_vars=predictor_vars)
                elif method == 'xgboost':
                    log.info(f"    [{category.upper()}|{method.upper()}] Training XGBoost model...")
                    interpolated, _ = interpolate_xgboost(df_test, test_variable, predictor_vars=predictor_vars)
                elif method == 'missforest':
                    interpolated = interpolate_missforest(df_test, test_variable, predictor_vars=predictor_vars)
                elif method == 'saits':
                    interpolated = interpolate_saits(df_test, test_variable, predictor_vars=predictor_vars)
                elif method == 'imputeformer':
                    interpolated = interpolate_imputeformer(df_test, test_variable, predictor_vars=predictor_vars)
                elif method == 'brits':
                    interpolated = interpolate_brits(df_test, test_variable, predictor_vars=predictor_vars)
                
                # Generate Visualization (Save example of gap filling)
                plot_dir = Path(CONFIG['output_dir']) / 'gap_examples'
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plot_dir / f"gap_{category}_{method}.png"
                
                save_gap_prediction_plot(
                    df_true=df,
                    df_pred=interpolated,
                    gap_mask=gap_mask,
                    variable=test_variable,
                    method=method,
                    category=category,
                    output_path=plot_path
                )
                
                # Evaluate
                predicted = interpolated.loc[gap_mask]
                
                # CRITICAL FIX: Filter NaN values before computing metrics
                valid_mask = predicted.notna() & true_values.notna()
                n_valid = valid_mask.sum()
                n_total = len(predicted)
                
                if n_valid == 0:
                    log.warning(f"    [{category.upper()}|{method.upper()}] No valid predictions for evaluation (all NaN)")
                    results.append({
                        'Category': category,
                        'Method': method,
                        'RMSE': np.nan,
                        'MAE': np.nan,
                        'R2': np.nan,
                        'Precision_%': np.nan
                    })
                    continue
                
                # Filter to valid predictions only
                predicted_valid = predicted[valid_mask]
                true_valid = true_values[valid_mask]
                
                # Log coverage if incomplete
                if n_valid < n_total:
                    coverage_pct = (n_valid / n_total) * 100
                    log.info(f"    [{category.upper()}|{method.upper()}] Prediction coverage: {n_valid}/{n_total} ({coverage_pct:.1f}%)")
                
                # Metrics on valid predictions only
                rmse = np.sqrt(np.mean((predicted_valid - true_valid)**2))
                mae = np.mean(np.abs(predicted_valid - true_valid))
                
                # Precision (Tolerance-based accuracy)
                data_range = df[test_variable].max() - df[test_variable].min()
                tolerance = 0.05 * data_range  # 5% tolerance
                precision = (np.abs(predicted_valid - true_valid) < tolerance).mean() * 100
                
                # R2
                ss_res = np.sum((true_valid - predicted_valid)**2)
                ss_tot = np.sum((true_valid - true_valid.mean())**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

                results.append({
                    'Category': category,
                    'Method': method,
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'R2': round(r2, 4),
                    'Precision_%': round(precision, 2)
                })
                
                log.info(f"    [{category.upper()}|{method.upper()}] RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
                
            except Exception as e:
                log.error(f"    [{category.upper()}|{method.upper()}] FAILED: {e}")
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
         log.info("=" * 70)
         log.info("BENCHMARK RESULTS SUMMARY")
         for _, row in results_df.iterrows():
             log.info(f"  {row['Category']:<10} | {row['Method']:<10} | RMSE={row['RMSE']:<7} | MAE={row['MAE']:<7} | R²={row['R2']:<7}")
         
         # Scientific Recommendation
         try:
             winners = results_df.loc[results_df.groupby('Category')['RMSE'].idxmin()]
             log.info("=" * 70)
             log.info("SCIENTIFIC RECOMMENDATION (Best Method per Category by RMSE)")
             for _, row in winners.iterrows():
                 log.info(f"  {row['Category']:<10} -> {row['Method']:<10} (RMSE: {row['RMSE']})")
             log.info("=" * 70)
         except Exception:
             pass
         
         # Plot results
         plot_benchmark_results(results_df, test_variable)
         
    return results_df

def plot_benchmark_results(results_df: pd.DataFrame, var_name: str):
    """Plot benchmark metrics."""
    if results_df.empty:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE Plot
    sns.barplot(data=results_df, x='Category', y='RMSE', hue='Method', ax=axes[0], palette='viridis')
    axes[0].set_title(f'{var_name} Gap Filling RMSE (Lower is Better)')
    axes[0].grid(True, alpha=0.3)
    
    # Precision Plot
    sns.barplot(data=results_df, x='Category', y='Precision_%', hue='Method', ax=axes[1], palette='viridis')
    axes[1].set_title(f'{var_name} Gap Filling Precision (Higher is Better)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    console.print("  [green]Saved benchmark plot to benchmark_results.png[/green]")


# =============================================================================
# PER-MODEL VISUALIZATION FUNCTIONS
# =============================================================================

def create_model_output_directories(output_dir: Path) -> Dict[str, Path]:
    """
    Create per-model output directories for organized visualization saves.
    
    Returns a dictionary mapping model names to their output paths.
    """
    model_dirs = {}
    models_base = output_dir / 'models'
    
    for model_name in MODEL_OUTPUT_SUBDIRS:
        model_path = models_base / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        model_dirs[model_name] = model_path
    
    log.info(f"Created model output directories in {models_base}")
    return model_dirs


def plot_xgboost_feature_importance(imputer, target_var: str, output_path: Path):
    """
    Plot XGBoost feature importance for interpretability and thesis documentation.
    
    Parameters
    ----------
    imputer : XGBoostImputer
        Trained XGBoost imputer with models
    target_var : str
        Target variable name for title
    output_path : Path
        Where to save the figure
    """
    try:
        # Get the forward model (primary)
        if 'fwd' not in imputer.models:
            log.warning("No forward model found in XGBoost imputer")
            return
        
        model = imputer.models['fwd']
        feature_names = imputer.feature_columns
        
        if feature_names is None or len(feature_names) == 0:
            log.warning("No feature names available for importance plot")
            return
        
        # Get importance scores
        importance = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importance)[::-1][:20]  # Top 20 features
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance[indices], color='steelblue', edgecolor='navy')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()  # Largest on top
        ax.set_xlabel('Feature Importance (Gain)')
        ax.set_title(f'XGBoost Feature Importance for {target_var}')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Saved XGBoost feature importance to {output_path}")
        
    except Exception as e:
        log.error(f"Failed to plot feature importance: {e}")


def plot_gap_example_per_model(df_original: pd.DataFrame, df_filled: pd.DataFrame,
                                target_var: str, gap_category: str, 
                                method_name: str, output_path: Path):
    """
    Plot a single gap example showing original vs interpolated data.
    
    This generates publication-quality figures for thesis documentation.
    """
    try:
        # Find a gap in the original data
        is_gap = df_original[target_var].isna()
        
        if not is_gap.any():
            return
        
        # Get gap boundaries
        gap_starts = is_gap & ~is_gap.shift(1, fill_value=False)
        gap_ends = is_gap & ~is_gap.shift(-1, fill_value=False)
        
        start_indices = df_original.index[gap_starts]
        end_indices = df_original.index[gap_ends]
        
        if len(start_indices) == 0:
            return
        
        # Take the first suitable gap
        gap_start = start_indices[0]
        gap_end = end_indices[0]
        
        # Create context window (3 days before/after)
        context_hours = 72
        plot_start = gap_start - pd.Timedelta(hours=context_hours)
        plot_end = gap_end + pd.Timedelta(hours=context_hours)
        
        # Slice data
        orig_slice = df_original[target_var].loc[plot_start:plot_end]
        filled_slice = df_filled[target_var].loc[plot_start:plot_end]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot original (with gaps)
        ax.plot(orig_slice.index, orig_slice.values, 'b-', linewidth=1.5, 
                label='Observed', alpha=0.8)
        
        # Plot filled values in gap region
        gap_region = filled_slice.loc[gap_start:gap_end]
        ax.plot(gap_region.index, gap_region.values, 'r--', linewidth=2,
                label=f'{method_name} Interpolation', alpha=0.9)
        
        # Highlight gap region
        ax.axvspan(gap_start, gap_end, alpha=0.2, color='yellow', label='Gap Region')
        
        ax.set_xlabel('Time')
        ax.set_ylabel(target_var)
        ax.set_title(f'{gap_category.upper()} Gap Filling Example: {target_var} ({method_name})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Saved gap example to {output_path}")
        
    except Exception as e:
        log.error(f"Failed to plot gap example: {e}")


def save_model_training_metrics(metrics_dict: Dict, model_name: str, output_path: Path):
    """
    Save training metrics to CSV for documentation and reproducibility.
    """
    try:
        metrics_df = pd.DataFrame(metrics_dict)
        csv_path = output_path / f'{model_name}_training_metrics.csv'
        metrics_df.to_csv(csv_path, index=False)
        log.info(f"Saved {model_name} training metrics to {csv_path}")
    except Exception as e:
        log.error(f"Failed to save training metrics: {e}")


def generate_full_series_reconstruction_plot(df_original: pd.DataFrame, 
                                               df_filled: pd.DataFrame,
                                               target_var: str,
                                               model_name: str,
                                               output_path: Path):
    """
    Generate a full time series reconstruction comparison plot.
    Shows original data vs model-interpolated data over the entire period.
    """
    try:
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Plot original
        ax.plot(df_original.index, df_original[target_var], 'b-', 
                linewidth=0.5, label='Original (with gaps)', alpha=0.7)
        
        # Plot filled
        ax.plot(df_filled.index, df_filled[target_var], 'orange', 
                linewidth=0.5, label=f'{model_name} Reconstruction', alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(target_var)
        ax.set_title(f'Full Time Series Reconstruction: {target_var} ({model_name})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Saved full series reconstruction to {output_path}")
        
    except Exception as e:
        log.error(f"Failed to generate reconstruction plot: {e}")


def save_gap_prediction_plot(df_true: pd.DataFrame, 
                             df_pred: pd.DataFrame,
                             gap_mask: pd.Series, 
                             variable: str,
                             method: str, 
                             category: str,
                             output_path: Path):
    """
    Save a zoom-in plot of a specific gap prediction.
    """
    try:
        # Find the first gap segment to plot
        gap_indices = np.where(gap_mask)[0]
        if len(gap_indices) == 0:
            return
            
        # Get start and end of the first gap block
        # We assume simulation creates blocks. We take the largest block or just the first one.
        # Let's simple find the range of the first continuous block in indices
        diffs = np.diff(gap_indices)
        # gaps in indices > 1 mean jumps. 
        # Find split points
        splits = np.where(diffs > 1)[0]
        
        if len(splits) > 0:
            # First block
            start_pos = gap_indices[0]
            end_pos = gap_indices[splits[0]]
        else:
            # One single block
            start_pos = gap_indices[0]
            end_pos = gap_indices[-1]
            
        # Define window around gap (e.g. 2x gap size on each side, separate max 500 pts)
        gap_len = end_pos - start_pos
        window = max(min(gap_len * 2, 500), 50) # at least 50 pts context
        
        plot_start = max(0, start_pos - window)
        plot_end = min(len(df_true), end_pos + window)
        
        # Extract data
        subset_idx = df_true.index[plot_start:plot_end]
        # original indices for gap
        gap_idx = df_true.index[start_pos:end_pos+1]
        
        # Create Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. Context (Surrounding data - True)
        # We plot the full window of True data
        context_vals = df_true.loc[subset_idx, variable]
        ax.plot(context_vals.index, context_vals, color='black', alpha=0.3, label='Context/Truth', linewidth=1)
        
        # 2. Highlight the Hidden Truth (what was removed)
        truth_segment = df_true.loc[gap_idx, variable]
        ax.plot(truth_segment.index, truth_segment, 'k-', linewidth=1.5, label='Ground Truth', alpha=0.8)
        
        # 3. Highlight the Prediction
        pred_segment = df_pred.loc[gap_idx] # df_pred is a Series of the target variable
        ax.plot(pred_segment.index, pred_segment, 'r--', linewidth=2, label=f'Predicted ({method})')
        
        # Highlight gap area
        ax.axvspan(gap_idx[0], gap_idx[-1], color='yellow', alpha=0.1, label='Gap Extent')
        
        ax.set_title(f'Gap Prediction: {variable} | {category.upper()} | {method.upper()}')
        ax.set_ylabel(variable)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
    except Exception as e:
        log.error(f"Failed to plot gap prediction: {e}")


# =============================================================================
# STATISTICAL ANALYSIS MODULE
# =============================================================================

def compute_descriptive_stats(df: pd.DataFrame, 
                             variables: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics.
    """
    if variables is None:
        variables = [col for col in df.columns 
                    if not any(suffix in col for suffix in ['_QC', '_STD'])]
    
    stats_list = []
    for var in variables:
        if var not in df.columns:
            continue
        
        series = df[var].dropna()
        if len(series) == 0:
            continue
        
        stats_dict = {
            'variable': var,
            'count': len(series),
            'missing_pct': (df[var].isna().sum() / len(df)) * 100,
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'p05': series.quantile(0.05),
            'p25': series.quantile(0.25),
            'median': series.median(),
            'p75': series.quantile(0.75),
            'p95': series.quantile(0.95),
            'max': series.max(),
            'skewness': stats.skew(series),
            'kurtosis': stats.kurtosis(series),
        }
        stats_list.append(stats_dict)
    
    return pd.DataFrame(stats_list).set_index('variable')


def compute_correlation_matrix(df: pd.DataFrame, 
                              variables: Optional[List[str]] = None,
                              method: str = 'pearson') -> pd.DataFrame:
    """
    Compute correlation matrix.
    
    Parameters
    ----------
    method : str
        'pearson' or 'spearman'
    """
    if variables is None:
        variables = [col for col in df.columns 
                    if not any(suffix in col for suffix in ['_QC', '_STD'])]
    
    df_subset = df[variables].dropna()
    return df_subset.corr(method=method)


def compute_acf_pacf(series: pd.Series, max_lag: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation and partial autocorrelation functions.
    
    Parameters
    ----------
    max_lag : int
        Maximum lag (default 48 = 24 hours at 30-min resolution)
    """
    from statsmodels.tsa.stattools import acf, pacf
    
    series_clean = series.dropna()
    if len(series_clean) < max_lag * 2:
        return np.array([]), np.array([])
    
    acf_vals = acf(series_clean, nlags=max_lag, fft=True)
    
    try:
        pacf_vals = pacf(series_clean, nlags=max_lag)
    except Exception:
        pacf_vals = np.full(max_lag + 1, np.nan)
    
    return acf_vals, pacf_vals


# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

def plot_gap_heatmap(df: pd.DataFrame, variables: Optional[List[str]] = None,
                     output_path: Optional[Path] = None):
    """
    Create a heatmap visualization of data gaps for ALL variables with yearly x-axis.
    """
    if variables is None:
        # Get ALL variables (no limit)
        variables = [col for col in df.columns 
                    if not any(suffix in col for suffix in ['_QC', '_STD'])]
    
    # Resample to monthly presence (binary: qualified or not)
    # We use 'ME' (Month End) which is the new pandas alias for 'M'
    df_monthly = df[variables].resample('ME').count()
    # Calculate expected observations per month (30 days * 48 half-hours)
    expected_per_month = 30 * 48
    # Normalize to percentage availability (0-100)
    availability_pct = (df_monthly / expected_per_month * 100).clip(upper=100)
    
    # Create figure with appropriate height for all variables
    n_vars = len(variables)
    fig_height = max(8, n_vars * 0.3)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # Create heatmap
    sns.heatmap(availability_pct.T, cmap='RdYlGn', vmin=0, vmax=100,
                yticklabels=True, ax=ax, cbar_kws={'label': 'Data Availability (%)'})
    
    # Set yearly x-axis ticks
    years = df_monthly.index.year.unique()
    year_positions = []
    year_labels = []
    for year in years:
        # Find first occurrence of each year
        year_idx = np.where(df_monthly.index.year == year)[0]
        if len(year_idx) > 0:
            year_positions.append(year_idx[0])
            year_labels.append(str(year))
    
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels, rotation=45, ha='right')
    
    ax.set_title('Data Availability Heatmap - All Variables (Red = Missing, Green = Present)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Variable')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"  ✓ Saved: {output_path}")
    
    plt.close()
    return fig


def plot_correlation_matrix(corr_matrix: pd.DataFrame, 
                           output_path: Optional[Path] = None):
    """
    Create a correlation matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                vmin=-1, vmax=1, fmt='.2f', ax=ax,
                square=True, linewidths=0.5)
    ax.set_title('Correlation Matrix (Pearson)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"  ✓ Saved: {output_path}")
    
    plt.close()
    return fig


def plot_time_series_overview(df: pd.DataFrame, variables: List[str],
                             output_path: Optional[Path] = None):
    """
    Create a multi-panel time series overview.
    """
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 2.5 * n_vars), sharex=True)
    
    if n_vars == 1:
        axes = [axes]
    
    for ax, var in zip(axes, variables):
        if var in df.columns:
            ax.plot(df.index, df[var], lw=0.5, alpha=0.8)
            ax.set_ylabel(var)
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.suptitle('Time Series Overview', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"  ✓ Saved: {output_path}")
    
    plt.close()
    return fig


def plot_instrument_timeseries(df: pd.DataFrame, instrument: str, 
                               output_path: Optional[Path] = None, 
                               config: dict = CONFIG):
    """
    Create a multi-panel time series plot for a specific instrument showing ALL its variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset
    instrument : str
        Instrument name: 'ctd', 'currents', 'waves', 'airmar', 'ctvg'
    """
    # Define prefixes and titles
    instrument_info = {
        'ctd': {'prefix': '', 'title': 'CTD - Oceanographic Variables'},
        'currents': {'prefix': 'CUR_', 'title': 'AWAC Currents'},
        'waves': {'prefix': 'WAV_', 'title': 'AWAC Waves'},
        'airmar': {'prefix': 'AIR_', 'title': 'Airmar - Offshore Atmospheric'},
        'ctvg': {'prefix': 'LAND_', 'title': 'CTVG Vantage Pro2 - Land Atmospheric'},
    }
    
    if instrument not in instrument_info:
        return None
    
    info = instrument_info[instrument]
    prefix = info['prefix']
    title = info['title']
    
    # Get variables for this instrument
    if prefix:
        variables = [col for col in df.columns 
                    if col.startswith(prefix) and 
                    not any(suffix in col for suffix in ['_QC', '_STD'])]
    else:
        # CTD has no prefix - use the config variables
        variables = [col for col in config['variables'].get(instrument, [])
                    if col in df.columns]
    
    if not variables:
        console.print(f"  [yellow]No variables found for {instrument}[/yellow]")
        return None
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(16, 2.5 * n_vars), sharex=True)
    
    if n_vars == 1:
        axes = [axes]
    
    colors = plt.cm.tab10.colors
    
    for i, (ax, var) in enumerate(zip(axes, variables)):
        if var in df.columns:
            # Resample to daily for cleaner visualization
            daily_data = df[var].resample('D').mean()
            ax.plot(daily_data.index, daily_data.values, 
                   lw=0.8, alpha=0.9, color=colors[i % len(colors)])
            
            # Clean label (remove prefix if any)
            label = var.replace(prefix, '') if prefix else var
            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics annotation
            valid_data = df[var].dropna()
            if len(valid_data) > 0:
                stats_text = f"μ={valid_data.mean():.2f}, σ={valid_data.std():.2f}"
                ax.annotate(stats_text, xy=(0.99, 0.95), xycoords='axes fraction',
                           ha='right', va='top', fontsize=8, color='gray')
    
    axes[-1].set_xlabel('Date', fontsize=11)
    plt.suptitle(f'{title} - Time Series (2009-2024)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"  ✓ Saved: {output_path}")
    
    plt.close()
    return fig


def plot_timeseries_with_gaps(df: pd.DataFrame, instrument: str,
                               gaps_df: pd.DataFrame,
                               output_path: Optional[Path] = None,
                               config: dict = CONFIG):
    """
    Create a multi-panel time series plot with classified gap bands (axvspan).
    
    This visualization shows gaps as colored bands overlaid on the time series,
    following scientific oceanographic conventions (similar to preproces_data.py).
    
    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset
    instrument : str
        Instrument name: 'ctd', 'currents', 'waves', 'airmar', 'ctvg'
    gaps_df : pd.DataFrame
        DataFrame with gap information (from analyze_gaps)
    """
    from matplotlib.patches import Patch
    
    # Define prefixes and titles
    instrument_info = {
        'ctd': {'prefix': '', 'title': 'CTD - Oceanographic Variables'},
        'currents': {'prefix': 'CUR_', 'title': 'AWAC Currents'},
        'waves': {'prefix': 'WAV_', 'title': 'AWAC Waves'},
        'airmar': {'prefix': 'AIR_', 'title': 'Airmar - Offshore Atmospheric'},
        'ctvg': {'prefix': 'LAND_', 'title': 'CTVG Vantage Pro2 - Land Atmospheric'},
    }
    
    # Gap colors following oceanographic convention
    gap_colors = {
        'micro': '#2ECC71',      # Green - minimal impact
        'short': '#F1C40F',      # Yellow - minor gaps
        'medium': '#E67E22',     # Orange - moderate gaps
        'long': '#E74C3C',       # Red - significant gaps
        'extended': '#9B59B6',   # Purple - major gaps
        'gigant': '#1C1C1C',     # Dark grey/black - extreme gaps
    }
    
    gap_labels = {
        'micro': '<1h',
        'short': '1-6h',
        'medium': '6h-3d',
        'long': '3-30d',
        'extended': '30-60d',
        'gigant': '>60d',
    }
    
    if instrument not in instrument_info:
        return None
    
    info = instrument_info[instrument]
    prefix = info['prefix']
    title = info['title']
    
    # Get variables for this instrument
    if prefix:
        variables = [col for col in df.columns 
                    if col.startswith(prefix) and 
                    not any(suffix in col for suffix in ['_QC', '_STD'])]
    else:
        variables = [col for col in config['variables'].get(instrument, [])
                    if col in df.columns]
    
    if not variables:
        console.print(f"  [yellow]No variables found for {instrument}[/yellow]")
        return None
    
    # Filter gaps for this instrument's variables
    instrument_gaps = gaps_df[gaps_df['variable'].isin(variables)].copy()
    
    # Use first variable as reference for main gap visualization
    ref_var = variables[0]
    ref_gaps = gaps_df[gaps_df['variable'] == ref_var].copy()
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(18, 3 * n_vars), sharex=True)
    
    if n_vars == 1:
        axes = [axes]
    
    colors = plt.cm.tab10.colors
    
    for i, (ax, var) in enumerate(zip(axes, variables)):
        if var not in df.columns:
            continue
            
        # Get gaps for this specific variable
        var_gaps = gaps_df[gaps_df['variable'] == var]
        
        # Plot the time series (daily mean for clarity)
        daily_data = df[var].resample('D').mean()
        ax.plot(daily_data.index, daily_data.values, 
               lw=0.8, alpha=0.9, color=colors[i % len(colors)], zorder=2)
        
        # Overlay gap bands (axvspan)
        for _, gap in var_gaps.iterrows():
            cat = gap['category']
            if pd.notna(cat) and cat in gap_colors:
                ax.axvspan(gap['start'], gap['end'], 
                          color=gap_colors[cat], alpha=0.3, zorder=1)
        
        # Clean label
        label = var.replace(prefix, '') if prefix else var
        ax.set_ylabel(label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Statistics annotation
        valid_data = df[var].dropna()
        missing_pct = (df[var].isna().sum() / len(df)) * 100
        n_gaps = len(var_gaps)
        
        stats_text = f"μ={valid_data.mean():.2f} | σ={valid_data.std():.2f} | Missing: {missing_pct:.1f}% | Gaps: {n_gaps}"
        ax.annotate(stats_text, xy=(0.99, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=8, color='gray', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    axes[-1].set_xlabel('Date', fontsize=11)
    
    # Calculate gap percentages for legend
    if not instrument_gaps.empty:
        gap_counts = instrument_gaps['category'].value_counts()
        total_gaps = len(instrument_gaps)
        
        # Create legend with all categories
        legend_elements = []
        for cat in gap_colors.keys():
            count = gap_counts.get(cat, 0)
            pct = (count / total_gaps * 100) if total_gaps > 0 else 0
            if count > 0:  # Only show categories that exist
                legend_elements.append(
                    Patch(facecolor=gap_colors[cat], alpha=0.3, 
                          label=f"{cat} ({gap_labels[cat]}): {count} ({pct:.1f}%)")
                )
        
        # Add data line to legend
        legend_elements.append(
            plt.Line2D([0], [0], color=colors[0], lw=1, label='Data (daily mean)')
        )
        
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=min(4, len(legend_elements)),
                  fontsize=9, framealpha=0.9)
    
    plt.suptitle(f'{title} - Time Series with Gap Classification', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"  ✓ Saved: {output_path}")
    
    plt.close()
    return fig


def plot_gap_duration_histogram(gaps_df: pd.DataFrame, 
                                output_path: Optional[Path] = None):
    """
    Create histogram of gap durations for scientific analysis.
    
    Shows distribution of gaps < 24h and log-scale view for all gaps.
    """
    if gaps_df.empty:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Gaps < 24 hours (linear scale)
    gaps_under_24h = gaps_df[gaps_df['duration_hours'] < 24]
    if not gaps_under_24h.empty:
        axes[0].hist(gaps_under_24h['duration_hours'], bins=24, 
                    color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Gap Duration (hours)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Distribution of Gaps < 24 hours', fontsize=12, fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Add statistics
        mean_h = gaps_under_24h['duration_hours'].mean()
        median_h = gaps_under_24h['duration_hours'].median()
        axes[0].axvline(mean_h, color='red', linestyle='--', lw=2, label=f'Mean: {mean_h:.1f}h')
        axes[0].axvline(median_h, color='green', linestyle='--', lw=2, label=f'Median: {median_h:.1f}h')
        axes[0].legend()
    
    # Right: All gaps (log scale)
    axes[1].hist(gaps_df['duration_hours'], bins=50, 
                color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Gap Duration (hours)', fontsize=11)
    axes[1].set_ylabel('Frequency (log scale)', fontsize=11)
    axes[1].set_title('Distribution of All Gaps', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Add category boundaries
    boundaries = [1, 6, 72, 720, 1440]  # hours
    labels = ['micro|short', 'short|medium', 'medium|long', 'long|extended', 'extended|gigant']
    for b, lbl in zip(boundaries, labels):
        axes[1].axvline(b, color='gray', linestyle=':', alpha=0.7)
        axes[1].text(b, axes[1].get_ylim()[1]*0.8, lbl, rotation=90, 
                    fontsize=7, ha='right', va='top')
    
    plt.suptitle('Gap Duration Analysis - OBSEA Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"  ✓ Saved: {output_path}")
    
    plt.close()
    return fig


def plot_ts_diagram(df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Create a T-S diagram with density contours.
    """
    try:
        import gsw
    except ImportError:
        console.print("[yellow]Warning: gsw not installed, skipping T-S diagram[/yellow]")
        return None
    
    temp = df['TEMP'].dropna() if 'TEMP' in df.columns else pd.Series()
    salt = df['PSAL'].dropna() if 'PSAL' in df.columns else pd.Series()
    
    if temp.empty or salt.empty:
        console.print("[yellow]Warning: TEMP or PSAL not available, skipping T-S diagram[/yellow]")
        return None
    
    # Align indices
    common_idx = temp.index.intersection(salt.index)
    temp = temp.loc[common_idx]
    salt = salt.loc[common_idx]
    
    # Create density grid
    t_range = np.linspace(temp.min() - 1, temp.max() + 1, 100)
    s_range = np.linspace(salt.min() - 0.5, salt.max() + 0.5, 100)
    S, T = np.meshgrid(s_range, t_range)
    rho = gsw.rho(S, T, 0) - 1000  # Sigma-theta
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Density contours
    cs = ax.contour(S, T, rho, colors='grey', alpha=0.5, linestyles='dashed')
    ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f')
    
    # Data points colored by day of year
    sc = ax.scatter(salt, temp, c=salt.index.dayofyear, cmap='twilight',
                    s=5, alpha=0.5)
    
    plt.colorbar(sc, label='Day of Year', ax=ax)
    ax.set_xlabel('Salinity (PSU)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('T-S Diagram - OBSEA (with σ₀ isopycnals)')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"  ✓ Saved: {output_path}")
    
    plt.close()
    return fig


# =============================================================================
# VARIABLE QUALITY FILTERING
# =============================================================================

def filter_high_quality_variables(df: pd.DataFrame, 
                                  high_quality_vars: List[str] = HIGH_QUALITY_VARIABLES,
                                  keep_qc_std: bool = True) -> pd.DataFrame:
    """
    Filter DataFrame to keep only high-quality variables (≤25% gaps).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with all variables
    high_quality_vars : List[str]
        List of high-quality variable names
    keep_qc_std : bool
        Whether to also keep associated _QC and _STD columns
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if not USE_HIGH_QUALITY_FILTER:
        log.info("  High-quality filter is DISABLED (USE_HIGH_QUALITY_FILTER=False)")
        return df
    
    log.info(f"  Filtering to {len(high_quality_vars)} high-quality variables (≤25% gaps)")
    
    # Collect columns to keep
    columns_to_keep = []
    
    for var in high_quality_vars:
        if var in df.columns:
            columns_to_keep.append(var)
            
            # Also keep QC and STD columns if they exist
            if keep_qc_std:
                qc_col = f"{var}_QC"
                std_col = f"{var}_STD"
                if qc_col in df.columns:
                    columns_to_keep.append(qc_col)
                if std_col in df.columns:
                    columns_to_keep.append(std_col)
    
    # Filter DataFrame
    df_filtered = df[columns_to_keep].copy()
    
    log.info(f"  ✓ Filtered: {len(df.columns)} → {len(df_filtered.columns)} columns")
    log.info(f"  Variables retained: {', '.join([c for c in columns_to_keep if '_QC' not in c and '_STD' not in c])}")
    
    return df_filtered


# =============================================================================
# OUTPUT MODULE
# =============================================================================

def save_outputs(df: pd.DataFrame, stats_df: pd.DataFrame, 
                corr_matrix: pd.DataFrame, gaps_summary: pd.DataFrame,
                gaps_df: pd.DataFrame,
                output_dir: Path):
    """
    Save all outputs to files.
    
    Parameters
    ----------
    gaps_df : pd.DataFrame
        Raw gaps DataFrame (from analyze_gaps) for generating gap plots
    """
    # Create directories
    data_dir = output_dir / 'data'
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'
    
    for d in [data_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    console.print("\n[bold blue]Saving outputs...[/bold blue]")
    
    # Save data
    df.to_csv(data_dir / 'OBSEA_multivariate_30min.csv')
    console.print(f"  ✓ Data: {data_dir / 'OBSEA_multivariate_30min.csv'}")
    
    # Save tables
    stats_df.to_csv(tables_dir / 'descriptive_statistics.csv')
    corr_matrix.to_csv(tables_dir / 'correlation_matrix.csv')
    if not gaps_summary.empty:
        gaps_summary.to_csv(tables_dir / 'gap_summary.csv')
    if not gaps_df.empty:
        gaps_df.to_csv(tables_dir / 'gaps_detailed.csv', index=False)
    console.print(f"  ✓ Tables: {tables_dir}")
    
    # Generate figures
    console.print("  Generating figures...")
    
    # Gap heatmap with all variables and yearly x-axis
    plot_gap_heatmap(df, output_path=figures_dir / 'gaps_heatmap.png')
    
    # Correlation matrix
    plot_correlation_matrix(corr_matrix, output_path=figures_dir / 'correlation_matrix.png')
    
    # Per-instrument time series plots (without gaps)
    console.print("  Generating per-instrument time series...")
    for instrument in ['ctd', 'currents', 'waves', 'airmar', 'ctvg']:
        plot_instrument_timeseries(df, instrument, 
                                   output_path=figures_dir / f'timeseries_{instrument}.png')
    
    # Per-instrument time series WITH gap classification (axvspan)
    if not gaps_df.empty:
        console.print("  Generating time series with gap classification...")
        for instrument in ['ctd', 'currents', 'waves', 'airmar', 'ctvg']:
            plot_timeseries_with_gaps(df, instrument, gaps_df,
                                      output_path=figures_dir / f'timeseries_{instrument}_gaps.png')
        
        # Gap duration histogram
        console.print("  Generating gap duration histogram...")
        plot_gap_duration_histogram(gaps_df, 
                                    output_path=figures_dir / 'gap_duration_histogram.png')
    
    # T-S diagram
    plot_ts_diagram(df, output_path=figures_dir / 'ts_diagram.png')
    
    console.print(f"  ✓ Figures: {figures_dir}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Main pipeline execution.
    """
    # Setup output directory
    base_path = Path(__file__).parent
    output_dir = base_path / CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize professional logging
    global log
    log, log_file = setup_pipeline_logging(output_dir)
    
    log.info("=" * 70)
    log.info("OBSEA MULTIVARIATE DATA ANALYSIS PIPELINE (LUP VERSION)")
    log.info(f"Execution started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Working Directory: {base_path}")
    log.info(f"Log file: {log_file}")
    log.info("=" * 70)
    
    # Step 1: Load all data
    log.info("STEP 1: LOADING DATA")
    log.info("Loading multiple observational sources (CTD, AWAC, Met)...")
    data = load_all_data(CONFIG)
    
    if not data:
        log.error("Error: No data loaded. Pipeline aborted.")
        return
    
    # Step 2: Apply instrumental QC
    log.info("STEP 2: APPLYING INSTRUMENTAL QC")
    data_qc = {}
    for instrument, df in data.items():
        data_qc[instrument] = apply_instrumental_qc(df, instrument, CONFIG)
        n_flagged = sum(
            (data_qc[instrument][col] > 1).sum() 
            for col in data_qc[instrument].columns if '_QC_INST' in col
        )
        log.info(f"  - {instrument:<10}: {n_flagged:,} values flagged as suspect/fail")
    
    # Step 3: Create unified dataset
    log.info("STEP 3: CREATING UNIFIED MULTIVARIATE DATASET")
    unified_df = create_unified_dataset(data_qc, CONFIG['target_sampling_rate'])
    
    # Step 3.5: Phase 3 - Add derived features (σ_θ, N², wind stress, etc.)
    log.info("STEP 3.5: OCEANOGRAPHIC FEATURE EXTRACTION AND ADVANCED PREPROCESSING")
    unified_df = add_derived_features(unified_df, compute_stl=True) # ENABLED STL
    
    # Step 4: Statistical analysis
    log.info("STEP 4: STATISTICAL ANALYSIS AND CORRELATIONS")
    
    # Get primary variables (no prefixes for main analysis)
    primary_vars = [col for col in unified_df.columns 
                   if not any(suffix in col for suffix in ['_QC', '_STD'])]
    
    stats_df = compute_descriptive_stats(unified_df, primary_vars)
    log.info(f"  - Computed descriptive statistics for {len(stats_df)} variables")
    
    corr_matrix = compute_correlation_matrix(unified_df, primary_vars[:15])  # Limit for readability
    log.info("  - Computed correlation matrix for primary predictors")
    
    # Step 5: Gap analysis
    log.info("STEP 5: GAP ANALYSIS AND CLASSIFICATION")
    gaps_df = analyze_gaps(unified_df, primary_vars)
    gaps_summary = create_gap_summary(gaps_df)
    gaps_summary = create_gap_summary(gaps_df)
    
    if not gaps_summary.empty:
        log.info(f"  - Identified {len(gaps_df):,} gaps across all variables")
        # Detailed summary will be in the gaps_detailed.csv table
    
    
    # Step 5.5: Filter to High-Quality Variables (if enabled)
    log.info("STEP 5.5: VARIABLE QUALITY FILTERING")
    if USE_HIGH_QUALITY_FILTER:
        # Apply filter before interpolation to reduce computational cost
        unified_df_filtered = filter_high_quality_variables(unified_df)
        
        # Also filter gaps_df to match
        filtered_vars = [col for col in unified_df_filtered.columns 
                        if not any(suffix in col for suffix in ['_QC', '_STD'])]
        gaps_df_filtered = gaps_df[gaps_df['variable'].isin(filtered_vars)]
        
        log.info(f"  Working with {len(filtered_vars)} high-quality variables for interpolation")
    else:
        unified_df_filtered = unified_df
        gaps_df_filtered = gaps_df
        log.info("  Using all variables (no quality filter applied)")
    
    # Step 5.6: Benchmark Gap Filling (on filtered data)
    log.info("STEP 5.6: SCIENTIFIC BENCHMARKING OF INTERPOLATION METHODS")
    log.info("Testing: Linear, Time, Splines, Polynomial, VARMA, Bi-LSTM")
    # comparison_df = benchmark_gap_filling(
    #     unified_df_filtered, 
    #     test_variable='TEMP',
    #     gap_categories=['micro', 'short', 'medium', 'long', 'extended', 'gigant']
    # )
    # Enable benchmarking with ALL models for comprehensive scientific comparison
    comparison_df = benchmark_gap_filling(
        unified_df_filtered, 
        test_variable='TEMP',
        gap_categories=['micro', 'short', 'medium', 'long', 'extended', 'gigant']
    )
    
    # Step 5.7: Selective Interpolation using INTERPOLATION_CONFIG (on filtered data)
    log.info("STEP 5.7: SELECTIVE INTERPOLATION (CONFIG-BASED)")
    log.info("Applying optimized methods for production reconstruction")
    
    unified_df_interp, tracking_df = selective_interpolation(
        unified_df_filtered, gaps_df_filtered,
        interp_config=INTERPOLATION_CONFIG,
        freq='30min'
    )
    
    # Step 5.8: Create per-model output directories and save visualizations
    log.info("STEP 5.8: GENERATING PER-MODEL VISUALIZATIONS")
    model_dirs = create_model_output_directories(output_dir)
    
    # Save XGBoost feature importance if model was used
    if hasattr(selective_interpolation, '_model_cache'):
        cache = selective_interpolation._model_cache
        for key, imputer in cache.items():
            if 'xgboost' in key.lower():
                var_name = key.split('_')[0] if '_' in key else 'TEMP'
                plot_xgboost_feature_importance(
                    imputer, var_name,
                    model_dirs['xgboost'] / f'feature_importance_{var_name}.png'
                )
    
    # Generate full series reconstruction plots for primary variables
    for target_var in ['TEMP', 'PSAL']:
        if target_var in unified_df_filtered.columns and target_var in unified_df_interp.columns:
            generate_full_series_reconstruction_plot(
                unified_df_filtered, unified_df_interp, target_var, 'XGBoost',
                model_dirs['xgboost'] / f'full_reconstruction_{target_var}.png'
            )

    # Step 6: Save outputs
    log.info("STEP 6: SAVING FINAL OUTPUTS AND FIGURES")
    save_outputs(unified_df_interp, stats_df, corr_matrix, gaps_summary, gaps_df_filtered, output_dir)

    
    # Save interpolation comparison and tracking to benchmarks folder
    tables_dir = output_dir / 'tables'
    if not comparison_df.empty:
        comparison_df.to_csv(tables_dir / 'interpolation_comparison.csv', index=False)
        comparison_df.to_csv(model_dirs['benchmarks'] / 'interpolation_comparison.csv', index=False)
        log.info(f"  - Saved interpolation benchmark: {tables_dir / 'interpolation_comparison.csv'}")
    
    tracking_df.to_csv(tables_dir / 'interpolation_tracking.csv')
    log.info(f"  - Saved interpolation tracking metadata: {tables_dir / 'interpolation_tracking.csv'}")
    
    # Save interpolated dataset separately
    data_dir = output_dir / 'data'
    unified_df_interp.to_csv(data_dir / 'OBSEA_multivariate_30min_interpolated.csv')
    log.info(f"  - Saved full interpolated dataset: {data_dir / 'OBSEA_multivariate_30min_interpolated.csv'}")
    
    # Final summary
    log.info("=" * 70)
    log.info("PIPELINE COMPLETED SUCCESSFULLY")
    log.info(f"  Final shape: {unified_df_interp.shape}")
    log.info(f"  Time range:  {unified_df_interp.index.min()} to {unified_df_interp.index.max()}")
    log.info(f"  Results in: {output_dir}")
    log.info(f"  Model outputs in: {output_dir / 'models'}")
    log.info("=" * 70)
    
    return unified_df_interp


if __name__ == '__main__':
    main()
