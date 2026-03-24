import numpy as np
import pandas as pd
import logging
import random

from obsea_pipeline.config.settings import GAP_CATEGORIES, GapInfo

logger = logging.getLogger(__name__)

def classify_gap_duration(length_points, freq_minutes=30):
    """
    Convierte la logitud de puntos nulos consecutivos a horas físicas 
    y cruza contra el diccionario de `GAP_CATEGORIES` para asignarle una clase (Micro, Long...).
    """
    hours = (length_points * freq_minutes) / 60.0
    
    # Evaluar en orden de menor a mayor
    for category, limits in GAP_CATEGORIES.items():
        if hours <= limits['max_hours']:
            return category
    
    # Catch-all (debería entrar en 'gigant' teóricamente)
    return 'gigant'

def detect_gaps(series):
    """
    Identifica todos los intervalos contiguos de NaNs en una serie unidimensional.
    Devuelve la máscara booleana de NaNs y la lista de objetos GapInfo evaluados.
    """
    is_nan = series.isna()
    
    # Si no hay huecos, atajo rápido
    if not is_nan.any():
        return is_nan, []
        
    # Cambio de True a False o viceversa indica bordes de gaps
    diffs = np.diff(is_nan.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0]
    
    # Manejar bordes de la serie (empieza o termina con gap)
    if is_nan.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if is_nan.iloc[-1]:
        ends = np.append(ends, len(is_nan) - 1)
        
    gaps = []
    for s, e in zip(starts, ends):
        length = e - s + 1
        category = classify_gap_duration(length)
        # s y e son índices numéricos relativos a la posición en el dataframe
        gaps.append(GapInfo(s, e, length, category))
        
    return is_nan, gaps

def analyze_dataset_gaps(df):
    """
    Ejecuta detect_gaps sobre todas las columnas para elaborar un cuadro 
    resumen global del estado de degradación del dataset multi-variable.
    """
    logger.info("Analyzing Multi-Variate Gap patterns (Scale-Aware classification)...")
    gap_summary = {}
    
    for col in df.columns:
        if df[col].dtype == object or col == 'Timestamp':
             continue
             
        mask, gaps = detect_gaps(df[col])
        series_len = len(df[col])
        total_missing = mask.sum()
        pct_missing = (total_missing / series_len) * 100 if series_len > 0 else 0
        
        # Conteo por categorías
        cats = {k: 0 for k in GAP_CATEGORIES.keys()}
        for g in gaps:
            cats[g.category] += 1
            
        gap_summary[col] = {
            'total_missing': total_missing,
            'pct_missing': pct_missing,
            'total_gaps_count': len(gaps),
            'categories': cats
        }
        
        # Logear variable crítica
        if pct_missing > 0:
            logger.info(f"  [{col}] Missing: {pct_missing:.2f}% | Gaps: {len(gaps)} | Gigant: {cats['gigant']}")
            
    return gap_summary

def create_canonical_index(start_date, end_date, freq='30min'):
    """
    Genera un index contínuo, aséptico de fecha a fecha para forzar la reindexación
    de instrumentos que hayan saltado lecturas, exponiendo así el salto como un NaN.
    """
    # Start on ceiling to freq and end on floor
    start = pd.Timestamp(start_date).ceil(freq)
    end = pd.Timestamp(end_date).floor(freq)
    return pd.date_range(start=start, end=end, freq=freq)

def get_gap_mask(df):
    """
    Devuelve un DataFrame booleano (misma shape) donde True = Data faltante artificial 
    o instrumental. Usado para enmascarar entrenamientos o aislar cálculos.
    """
    return df.isna()

def simulate_gaps(df, columns, missing_ratio=0.1, pattern='random', lengths=None):
    """
    Creador artificial de Gaps (MCAR/MAR). Usado en la fase de Benchmarking
    para reventar a propósito un dataset limpio y validarlo contra Truth.
    """
    df_simulated = df.copy()
    truth_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    if lengths is None:
        lengths = [10, 50, 200, 1000] # micro, short, long, extended simulado
    
    for col in columns:
        valid_indices = df[df[col].notna()].index
        num_missing = int(len(valid_indices) * missing_ratio)
        current_missing = 0
        
        while current_missing < num_missing:
            # Drop a consecutive chunk
            chunk_size = random.choice(lengths)
            chunk_size = min(chunk_size, num_missing - current_missing)
            
            # Pick valid starting point
            try:
                 start_idx = random.choice(valid_indices)
                 start_loc = df.index.get_loc(start_idx)
                 end_loc = min(start_loc + chunk_size, len(df))
                 
                 # slice en index position
                 target_idx = df.index[start_loc:end_loc]
                 
                 df_simulated.loc[target_idx, col] = np.nan
                 truth_mask.loc[target_idx, col] = True
                 
                 current_missing += chunk_size
                 # Refresh valid indices for next loop to avoid overlaps artificially
                 valid_indices = valid_indices.difference(target_idx)
            except Exception:
                 break
                 
    return df_simulated, truth_mask
