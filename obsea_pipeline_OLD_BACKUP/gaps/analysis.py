import numpy as np
import pandas as pd
import logging
import random

from obsea_pipeline.config.settings import GAP_CATEGORIES, GapInfo

logger = logging.getLogger(__name__)

def classify_gap_duration(length_points, freq_minutes=30):
    """
    Converts the length of consecutive null points to physical hours
    and matches against the `GAP_CATEGORIES` dictionary to assign a class (Micro, Long...).
    """
    hours = (length_points * freq_minutes) / 60.0
    
    # Evaluate in order from smallest to largest
    for category, limits in GAP_CATEGORIES.items():
        if hours <= limits['max_hours']:
            return category
    
    # Catch-all (should theoretically fall into 'gigant')
    return 'gigant'

def detect_gaps(series):
    """
    Identifies all contiguous intervals of NaNs in a 1D series.
    Returns the boolean NaN mask and the list of evaluated GapInfo objects.
    """
    is_nan = series.isna()
    
    # Quick bypass if there are no gaps
    if not is_nan.any():
        return is_nan, []
        
    # A change from True to False or vice versa indicates gap edges
    diffs = np.diff(is_nan.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0]
    
    # Handle series edges (starts or ends with a gap)
    if is_nan.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if is_nan.iloc[-1]:
        ends = np.append(ends, len(is_nan) - 1)
        
    gaps = []
    for s, e in zip(starts, ends):
        length = e - s + 1
        category = classify_gap_duration(length)
        # s and e are relative numerical indices to the dataframe position
        gaps.append(GapInfo(s, e, length, category))
        
    return is_nan, gaps

def analyze_dataset_gaps(df):
    """
    Runs detect_gaps on all columns to build a global summary
    of the degradation state of the multi-variable dataset.
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
        
        # Count by category
        cats = {k: 0 for k in GAP_CATEGORIES.keys()}
        for g in gaps:
            cats[g.category] += 1
            
        gap_summary[col] = {
            'total_missing': total_missing,
            'pct_missing': pct_missing,
            'total_gaps_count': len(gaps),
            'categories': cats
        }
        
        # Log critical variables
        if pct_missing > 0:
            logger.info(f"  [{col}] Missing: {pct_missing:.2f}% | Gaps: {len(gaps)} | Gigant: {cats['gigant']}")
            
    return gap_summary

def create_canonical_index(start_date, end_date, freq='30min'):
    """
    Generates a continuous, clean index from date to date to force reindexing
    of instruments that skipped readings, exposing the skip as a NaN.
    """
    # Start on ceiling to freq and end on floor
    start = pd.Timestamp(start_date).ceil(freq)
    end = pd.Timestamp(end_date).floor(freq)
    return pd.date_range(start=start, end=end, freq=freq)

def get_gap_mask(df):
    """
    Returns a boolean DataFrame (same shape) where True = Artificial or
    instrumental missing data. Used to mask training or isolate calculations.
    """
    return df.isna()

def simulate_gaps(df, columns, missing_ratio=0.1, pattern='random', lengths=None):
    """
    Artificial Gap Creator (MCAR/MAR). Used in the Benchmark phase
    to purposefully corrupt a clean dataset and validate it against Ground Truth.
    """
    df_simulated = df.copy()
    truth_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    if lengths is None:
        lengths = [10, 50, 200, 1000] # Simulated micro, short, long, extended
    
    for col in columns:
        valid_indices = df[df[col].notna()].index
        if len(valid_indices) == 0: continue
        
        num_missing = int(len(df) * missing_ratio)
        current_missing = 0
        
        # Limit attempts to avoid infinite loops if data is too sparse
        max_total_attempts = 200
        total_attempts = 0
        
        while current_missing < num_missing and total_attempts < max_total_attempts:
            total_attempts += 1
            # Drop a consecutive chunk
            chunk_size = random.choice(lengths)
            
            if chunk_size >= len(df):
                continue
            
            # Pick a random starting position and check if the ENTIRE chunk is valid
            start_loc = random.randint(0, len(df) - chunk_size)
            target_idx = df.index[start_loc : start_loc + chunk_size]
            
            # Check for truth availability in the entire chunk
            if df.loc[target_idx, col].isna().any():
                continue
            
            # Check if we already masked this (avoid double masking/overlapping)
            if truth_mask.loc[target_idx, col].any():
                continue
                
            # Apply mask
            df_simulated.loc[target_idx, col] = np.nan
            truth_mask.loc[target_idx, col] = True
            
            current_missing += chunk_size
                 
    return df_simulated, truth_mask
