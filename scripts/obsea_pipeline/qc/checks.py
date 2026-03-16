import numpy as np
import pandas as pd
import logging

from obsea_pipeline.config.settings import CONFIG

logger = logging.getLogger(__name__)

def range_check(series, var_name):
    """
    QARTOD Test 11: Gross Range Test
    """
    if var_name not in CONFIG:
        return series
    vmin, vmax = CONFIG[var_name][0], CONFIG[var_name][1]
    filtered = series.copy()
    filtered[(series < vmin) | (series > vmax)] = np.nan
    nan_count = filtered.isna().sum() - series.isna().sum()
    if nan_count > 0:
         logger.info(f"    QC [Range]: {var_name} rejected {nan_count} values outside [{vmin}, {vmax}]")
    return filtered

def spike_check(series, var_name):
    """
    QARTOD Test 10: Spike Test
    """
    if var_name not in CONFIG:
        return series
    
    threshold = CONFIG[var_name][4]  # Spike_Threshold
    if threshold is None:
        return series
        
    filtered = series.copy()
    # Identificar picos comparando con el valor anterior y posterior
    diff_prev = np.abs(series - series.shift(1))
    diff_next = np.abs(series - series.shift(-1))
    
    mask = (diff_prev > threshold) & (diff_next > threshold)
    filtered[mask] = np.nan
    
    nan_count = filtered.isna().sum() - series.isna().sum()
    if nan_count > 0:
        logger.info(f"    QC [Spike]: {var_name} rejected {nan_count} spikes (> {threshold})")
    return filtered

def gradient_check(series, var_name):
    """
    QARTOD Test 15: Rate of Change Test
    """
    if var_name not in CONFIG:
         return series
         
    threshold = CONFIG[var_name][5]  # Gradient_Threshold
    if threshold is None:
         return series
         
    filtered = series.copy()
    diff = np.abs(series.diff())
    
    mask = diff > threshold
    filtered[mask] = np.nan
    
    nan_count = filtered.isna().sum() - series.isna().sum()
    if nan_count > 0:
        logger.info(f"    QC [Grad]:  {var_name} rejected {nan_count} high gradients (> {threshold})")
    
    return filtered

def flatline_check(series, var_name):
    """
    QARTOD Test 6: Flat Line Test
    Valores idénticos consecutivos indicando fallo de sensor.
    """
    if var_name not in CONFIG:
         return series
         
    min_points = CONFIG[var_name][6]  # Flatline_Points limit
    if min_points is None:
         return series
         
    filtered = series.copy()
    
    # Encontrar secuencias de valores repetidos
    # diff() es 0 si es igual al anterior
    is_same = series.diff() == 0
    # Agrupar las secuencias consecutivas de verdaderos
    groups = (~is_same).cumsum()
    # Contar tamaño de cada grupo
    counts = is_same.groupby(groups).transform('size')
    
    # Anular grupos mayores que el límite
    mask = counts >= min_points
    # La máscara solo marca desde el 2o elemento repetido, hay que marcar también el 1o
    # para eliminar toda la secuencia flatline
    mask = mask | mask.shift(-1).fillna(False) 
    # Asegurarnos de no borrar NaNs que ya estaban, solo valores reales repetidos
    mask = mask & series.notna()

    filtered[mask] = np.nan
    
    nan_count = filtered.isna().sum() - series.isna().sum()
    if nan_count > 0:
        logger.info(f"    QC [Flat]:  {var_name} rejected {nan_count} flatline values (> {min_points} seq)")
        
    return filtered

def apply_instrumental_qc(df):
    """
    Aplica todos los tests QARTOD al dataframe completo, variable por variable.
    """
    qc_df = df.copy()
    logger.info("Applying QARTOD Quality Control...")
    
    for col in df.columns:
        if col in CONFIG:
            qc_df[col] = range_check(qc_df[col], col)
            qc_df[col] = spike_check(qc_df[col], col)
            qc_df[col] = gradient_check(qc_df[col], col)
            qc_df[col] = flatline_check(qc_df[col], col)
            
    return qc_df
