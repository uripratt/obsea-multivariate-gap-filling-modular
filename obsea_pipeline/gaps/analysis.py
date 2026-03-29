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
    [LEGACY] Creador artificial de Gaps dispersos (MCAR/MAR). 
    ADVERTENCIA: Esta función dispersa muchos fragmentos pequeños en vez de crear
    bloques contiguos. Usar simulate_contiguous_gaps() para benchmarking riguroso.
    """
    df_simulated = df.copy()
    truth_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    if lengths is None:
        lengths = [10, 50, 200, 1000]
    
    for col in columns:
        valid_indices = df[df[col].notna()].index
        num_missing = int(len(valid_indices) * missing_ratio)
        current_missing = 0
        
        while current_missing < num_missing:
            chunk_size = random.choice(lengths)
            chunk_size = min(chunk_size, num_missing - current_missing)
            
            try:
                 start_idx = random.choice(valid_indices)
                 start_loc = df.index.get_loc(start_idx)
                 end_loc = min(start_loc + chunk_size, len(df))
                 target_idx = df.index[start_loc:end_loc]
                 
                 df_simulated.loc[target_idx, col] = np.nan
                 truth_mask.loc[target_idx, col] = True
                 current_missing += chunk_size
                 valid_indices = valid_indices.difference(target_idx)
            except Exception:
                 break
                 
    return df_simulated, truth_mask


def simulate_contiguous_gaps(df, column, n_gaps, min_pts, max_pts, context_margin=96, extreme_mode=False):
    """
    Creador de Gaps CONTIGUOS para benchmarking científico riguroso.
    
    A diferencia de simulate_gaps(), esta función crea exactamente `n_gaps` bloques 
    contiguos de tamaño aleatorio entre [min_pts, max_pts], cada uno separado por 
    al menos `context_margin` puntos de datos observados para permitir el aprendizaje 
    de contexto por parte de los modelos.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset completo con índice temporal.
    column : str
        Variable objetivo a enmascarar.
    n_gaps : int
        Número de gaps contiguos a crear.
    min_pts : int
        Longitud mínima del gap en puntos (30min cada uno).
    max_pts : int
        Longitud máxima del gap en puntos.
    context_margin : int
        Puntos mínimos de separación entre gaps y entre gaps y bordes del dataset.
        Garantiza que los modelos tengan suficiente contexto observado.
    
    Returns:
    --------
    df_simulated : pd.DataFrame (copia con NaNs inyectados)
    gap_mask : pd.Series booleana (True = punto enmascarado artificialmente)
    gap_blocks : list[dict] con {start_loc, end_loc, length} de cada gap
    """
    df_simulated = df.copy()
    gap_mask = pd.Series(False, index=df.index, name=column)
    gap_blocks = []
    
    series = df[column]
    n = len(series)
    
    # Solo trabajar con regiones ya observadas (no pre-existentes NaN)
    observed_mask = series.notna()
    
    # Zona segura: evitar bordes del dataset para garantizar contexto bilateral
    safe_start = context_margin
    safe_end = n - context_margin - max_pts
    
    if safe_end <= safe_start:
        logger.warning(f"  [simulate_contiguous_gaps] Dataset too short for {n_gaps} gaps of size {max_pts}")
        return df_simulated, gap_mask, gap_blocks
        
    extreme_candidates = []
    if extreme_mode:
        logger.info(f"  [simulate_contiguous_gaps] EXTREME MODE ON: Calculating top 5% variance anomalies for {column}...")
        try:
            roll_var = series.rolling(window=48, min_periods=24).var()
            threshold = roll_var.quantile(0.95)
            extreme_mask = roll_var >= threshold
            extreme_candidates = np.where(extreme_mask.iloc[safe_start:safe_end])[0] + safe_start
        except Exception as e:
            logger.warning(f"  Failed to calculate extreme bounds: {e}. Falling back to random.")
    
    # Intentar colocar n_gaps gaps sin solapamiento
    placed = 0
    max_attempts = n_gaps * 50  # Evitar loop infinito
    attempts = 0
    
    # Zonas reservadas (para evitar solapamientos)
    reserved = np.zeros(n, dtype=bool)
    
    while placed < n_gaps and attempts < max_attempts:
        attempts += 1
        
        # Tamaño aleatorio dentro del rango de la categoría
        gap_length = random.randint(min_pts, min(max_pts, n - 2 * context_margin))
        
        # Posición aleatoria dentro de la zona segura, sesgada hacia extremos si está activado
        if extreme_mode and len(extreme_candidates) > 0:
            peak_loc = random.choice(extreme_candidates)
            # Center the gap around the storm peak
            start_loc = max(safe_start, peak_loc - int(gap_length / 2))
        else:
            start_loc = random.randint(safe_start, max(safe_start, safe_end))
            
        end_loc = start_loc + gap_length
        
        if end_loc >= n:
            continue
            
        # Verificar que la zona candidata:
        # 1. No solape con gaps ya colocados (incluyendo márgenes de contexto)
        margin_start = max(0, start_loc - context_margin)
        margin_end = min(n, end_loc + context_margin)
        
        if reserved[margin_start:margin_end].any():
            continue
            
        # 2. Tenga el 100% de los datos observados (Rigurosidad científica para Ground Truth)
        candidate_region = observed_mask.iloc[start_loc:end_loc]
        if candidate_region.mean() < 1.0:
            continue
        
        # 3. Tenga contexto observado 100% completo en ambos lados (bilateralidad perfecta)
        left_context = observed_mask.iloc[max(0, start_loc - context_margin):start_loc]
        right_context = observed_mask.iloc[end_loc:min(n, end_loc + context_margin)]
        
        if left_context.mean() < 1.0 or right_context.mean() < 1.0:
            continue
        
        # ¡Colocar el gap!
        target_idx = df.index[start_loc:end_loc]
        df_simulated.loc[target_idx, column] = np.nan
        gap_mask.loc[target_idx] = True
        reserved[margin_start:margin_end] = True
        
        gap_blocks.append({
            'start_loc': start_loc,
            'end_loc': end_loc,
            'length': gap_length,
            'start_time': df.index[start_loc],
            'end_time': df.index[end_loc - 1],
        })
        placed += 1
    
    if placed < n_gaps:
        logger.warning(f"  [simulate_contiguous_gaps] Only placed {placed}/{n_gaps} gaps (dataset constraints)")
    else:
        logger.info(f"  [simulate_contiguous_gaps] Placed {placed} contiguous gaps [{min_pts}-{max_pts} pts each]")
    
    return df_simulated, gap_mask, gap_blocks
