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

def resample_variable(series, var_name, freq='30min'):
    """
    Remuestrea una serie temporal específica usando la técnica adecuada según la variable.
    Variables angulares usan media circular. El resto usan media aritmética estandar.
    """
    # Variables direccionales que necesitan media circular
    circular_vars = ['WDIR', 'CDIR', 'VMDR'] 
    
    if var_name in circular_vars:
        # custom resampling with circular mean
        # .apply calls the function on the groups
        return series.resample(freq).apply(circular_mean)
    else:
        # Standard arithmetic mean for scalar variables (TEMP, PSAL, WSPD...)
        return series.resample(freq).mean()

def resample_dataframe(df, freq='30min'):
    """
    Alinea y remuestrea un dataframe asíncrono entero a una frecuencia fija.
    """
    logger.info(f"Resampling complete dataset to {freq} intervals...")
    
    # Check if index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
         raise ValueError("El DataFrame debe tener un DatetimeIndex para resamplear.")
         
    resampled_cols = {}
    for col in df.columns:
        if df[col].dtype == object: # saltar columnas de texto si las hay
            continue
        resampled_cols[col] = resample_variable(df[col], col, freq)
        
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
