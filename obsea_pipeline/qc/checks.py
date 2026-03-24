import numpy as np
import pandas as pd
import logging

from obsea_pipeline.config.settings import CONFIG

logger = logging.getLogger(__name__)

def range_check(series, var_name):
    """
    QARTOD Test 11: Gross Range Test
    Out of bounds values are considered FAILS (4).
    """
    mask_fail = pd.Series(False, index=series.index)
    mask_suspect = pd.Series(False, index=series.index)
    
    if var_name not in CONFIG:
        return mask_fail, mask_suspect
        
    vmin, vmax = CONFIG[var_name][0], CONFIG[var_name][1]
    mask_fail = (series < vmin) | (series > vmax)
    
    return mask_fail, mask_suspect

def spike_check(series, var_name):
    """
    QARTOD Test 10: Spike Test
    Spikes represent likely anomalies, marked as SUSPECT (3).
    """
    mask_fail = pd.Series(False, index=series.index)
    mask_suspect = pd.Series(False, index=series.index)
    
    if var_name not in CONFIG:
        return mask_fail, mask_suspect
    
    threshold = CONFIG[var_name][4]  # Spike_Threshold
    if threshold is None:
        return mask_fail, mask_suspect
        
    diff_prev = np.abs(series - series.shift(1))
    diff_next = np.abs(series - series.shift(-1))
    
    mask_suspect = (diff_prev > threshold) & (diff_next > threshold)
    return mask_fail, mask_suspect

def gradient_check(series, var_name):
    """
    QARTOD Test 15: Rate of Change Test
    High gradients marked as SUSPECT (3).
    """
    mask_fail = pd.Series(False, index=series.index)
    mask_suspect = pd.Series(False, index=series.index)
    
    if var_name not in CONFIG:
         return mask_fail, mask_suspect
         
    threshold = CONFIG[var_name][5]  # Gradient_Threshold
    if threshold is None:
         return mask_fail, mask_suspect
         
    diff = np.abs(series.diff())
    mask_suspect = diff > threshold
    return mask_fail, mask_suspect

def flatline_check(series, var_name):
    """
    QARTOD Test 6: Flat Line Test
    Valores idénticos consecutivos indicando fallo de sensor (stuck sensor).
    Marked as FAIL (4).
    """
    mask_fail = pd.Series(False, index=series.index)
    mask_suspect = pd.Series(False, index=series.index)
    
    if var_name not in CONFIG:
         return mask_fail, mask_suspect
         
    min_points = CONFIG[var_name][6]  # Flatline_Points limit
    if min_points is None:
         return mask_fail, mask_suspect
         
    # Encontrar secuencias de valores repetidos
    # diff() es 0 si es igual al anterior
    is_same = series.diff() == 0
    # Agrupar las secuencias consecutivas de verdaderos
    groups = (~is_same).cumsum()
    # Contar tamaño de cada grupo
    counts = is_same.groupby(groups).transform('size')
    
    # Anular grupos mayores que el límite
    mask_fail = counts >= min_points
    # La máscara solo marca desde el 2o elemento repetido, hay que marcar también el 1o
    # para eliminar toda la secuencia flatline
    mask_fail = mask_fail | mask_fail.shift(-1).fillna(False) 
    # Asegurarnos de no borrar NaNs que ya estaban, solo valores reales repetidos
    mask_fail = mask_fail & series.notna()
    
    return mask_fail, mask_suspect

def rolling_mad_check(series, var_name, window=24, threshold_factor=4):
    """
    Robust Outlier Detection using Rolling Median Absolute Deviation (MAD).
    Excellent for oceanographic spikes that are not isolated points.
    Marked as SUSPECT (3) following QARTOD guidelines.
    """
    mask_fail = pd.Series(False, index=series.index)
    mask_suspect = pd.Series(False, index=series.index)
    
    if var_name not in CONFIG:
        return mask_fail, mask_suspect
        
    rolling_median = series.rolling(window=window, center=True).median()
    rolling_mad = (series - rolling_median).abs().rolling(window=window, center=True).median()
    
    # Avoid division by zero
    rolling_mad = rolling_mad.replace(0, np.nan).bfill().ffill()
    
    mask_suspect = (series - rolling_median).abs() > (threshold_factor * rolling_mad)
    return mask_fail, mask_suspect

def apply_instrumental_qc(df):
    """
    Aplica todos los tests QARTOD al dataframe completo, variable por variable.
    Construye una columna _QC con flags oficiales:
      1 = Pass
      3 = Suspect
      4 = Fail
    Solo se eliminan (convierten a np.nan) los valores FAIL.
    """
    qc_df = df.copy()
    logger.info("Applying Advanced QARTOD + MAD Quality Control (Soft Flagging)...")
    
    for col in df.columns:
        if col in CONFIG:
            flags = pd.Series(1, index=df.index, dtype=int)
            
            fail1, susp1 = range_check(df[col], col)
            fail2, susp2 = rolling_mad_check(df[col], col)
            fail3, susp3 = spike_check(df[col], col)
            fail4, susp4 = gradient_check(df[col], col)
            fail5, susp5 = flatline_check(df[col], col)
            
            is_susp = susp1 | susp2 | susp3 | susp4 | susp5
            is_fail = fail1 | fail2 | fail3 | fail4 | fail5
            
            # Apply flags in hierarchical order
            flags[is_susp] = 3
            flags[is_fail] = 4
            
            # Add QC column
            qc_col_name = f"{col}_QC"
            qc_df[qc_col_name] = flags
            
            # HARD DELETION only for fails
            qc_df.loc[flags == 4, col] = np.nan
            
            rejected = (flags == 4).sum()
            suspects = (flags == 3).sum()
            if rejected > 0 or suspects > 0:
                logger.info(f"    QC [{col}]: {rejected} FAILED (Removed), {suspects} SUSPECT (Included with Flag)")
            
    return qc_df
