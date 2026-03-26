import numpy as np
import pandas as pd
import logging
from scipy.stats import circmean

logger = logging.getLogger(__name__)

def circular_mean(series):
    """
    Promedio direccional circular para datos angulares (0-360) usando scipy.stats.circmean.
    Evita que el promedio de 359º y 1º sea 180º, devolviendo correctamente 0º.
    """
    valid_data = series.dropna()
    if len(valid_data) == 0:
        return np.nan
        
    # Convertir a radianes, calcular media y volver a grados
    rad_data = np.deg2rad(valid_data)
    mean_rad = circmean(rad_data)
    mean_deg = np.rad2deg(mean_rad)
    
    # Asegurar que está entre 0 y 360
    if mean_deg < 0:
        mean_deg += 360
        
    return mean_deg

# Sufijos de variables direccionales que requieren media circular
_CIRCULAR_SUFFIXES = ('WDIR', 'CDIR', 'VMDR')

def _is_circular_variable(var_name: str) -> bool:
    """Detecta si una variable es direccional (0-360°) usando matching por sufijo."""
    return var_name.endswith(_CIRCULAR_SUFFIXES)

def resample_variable(series, var_name, freq='30min'):
    """
    Remuestrea una serie temporal específica usando la técnica adecuada según la variable.
    Variables angulares usan media circular. El resto usan media aritmética estandar.
    """
    if _is_circular_variable(var_name):
        return series.resample(freq).apply(circular_mean)
    else:
        # Standard arithmetic mean for scalar variables (TEMP, PSAL, WSPD...)
        return series.resample(freq).mean()

def resample_with_qc(df, var_name, freq='30min'):
    """
    QC-Aware Resampling: excluye valores FAIL del promedio y propaga el peor flag
    del bin temporal como el QC del punto remuestreado.
    
    Returns: (resampled_values, resampled_qc) o (resampled_values, None) si no hay QC.
    """
    qc_col = f"{var_name}_QC"
    if qc_col in df.columns:
        # Solo promediar datos con flag <= 3 (Pass + Suspect)
        clean = df[var_name].where(df[qc_col] <= 3)
        resampled_val = resample_variable(clean, var_name, freq)
        resampled_qc = df[qc_col].resample(freq).max()  # Peor flag del bin
        return resampled_val, resampled_qc
    else:
        return resample_variable(df[var_name], var_name, freq), None

def resample_dataframe(df, freq='30min'):
    """
    Alinea y remuestrea un dataframe asíncrono entero a una frecuencia fija.
    Usa resample_with_qc si hay columnas _QC disponibles.
    """
    logger.info(f"Resampling complete dataset to {freq} intervals...")
    
    # Check if index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
         raise ValueError("El DataFrame debe tener un DatetimeIndex para resamplear.")
         
    resampled_cols = {}
    for col in df.columns:
        if df[col].dtype == object:  # saltar columnas de texto si las hay
            continue
        if col.endswith('_QC') or col.endswith('_STD'):
            continue  # QC/STD columns are handled inside resample_with_qc
            
        resampled_val, resampled_qc = resample_with_qc(df, col, freq)
        resampled_cols[col] = resampled_val
        if resampled_qc is not None:
            resampled_cols[f"{col}_QC"] = resampled_qc
        
    return pd.DataFrame(resampled_cols)

def create_unified_dataset(dict_of_dfs, freq='30min'):
    """
    Une y resamplea multiples diccionarios de DataFrames de diversos instrumentos en un
    único DataFrame cronológico ordenado maestrado por freq.
    """
    logger.info(f"Unifying all data sources into a single DataFrame at {freq} frequency")
    
    resampled_dfs = []
    
    for source_name, df in dict_of_dfs.items():
        if df is None or df.empty:
            logger.warning(f"  Source {source_name} is empty. Skipping.")
            continue
            
        logger.info(f"  Processing {source_name}")
        resampled_df = resample_dataframe(df, freq)
        resampled_dfs.append(resampled_df)
        
    if not resampled_dfs:
        raise ValueError("Todos los datasets de entrada están vacíos. Imposible unificar.")
        
    # Join (outer) them all aligning the time index gracefully
    unified_df = resampled_dfs[0]
    for i in range(1, len(resampled_dfs)):
         # Combine_first o join
         unified_df = unified_df.join(resampled_dfs[i], how='outer')
         
    # Eliminar posibles columnas duplicadas si un mismo nombre venía en 2 sources
    unified_df = unified_df.loc[:,~unified_df.columns.duplicated()]
    return unified_df
