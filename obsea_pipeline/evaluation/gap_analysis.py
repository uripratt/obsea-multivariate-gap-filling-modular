"""
Gap-specific analysis: error stratified by gap characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def identify_gaps(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Identify gap positions and lengths.
    
    Parameters
    ----------
    mask : np.ndarray
        Boolean mask where True indicates a gap
        
    Returns
    -------
    list of tuples
        List of (start_idx, end_idx) for each gap
    """
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i, is_gap in enumerate(mask):
        if is_gap and not in_gap:
            # Starting a gap
            in_gap = True
            gap_start = i
        elif not is_gap and in_gap:
            # Ending a gap
            gaps.append((gap_start, i))
            in_gap = False
    
    # Handle case where series ends in a gap
    if in_gap:
        gaps.append((gap_start, len(mask)))
    
    return gaps


def calculate_error_by_gap_length(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gap_mask: np.ndarray,
    length_bins: List[int] = [1, 6, 24, 48, 168, np.inf],
) -> pd.DataFrame:
    """
    Calculate error metrics stratified by gap length.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth
    y_pred : np.ndarray
        Predictions
    gap_mask : np.ndarray
        Boolean mask indicating gap positions
    length_bins : list of int
        Bin edges for gap lengths (in time steps)
        
    Returns
    -------
    pd.DataFrame
        Error metrics by gap length bin
    """
    gaps = identify_gaps(gap_mask)
    
    # Create bin labels
    bin_labels = []
    for i in range(len(length_bins) - 1):
        start = length_bins[i]
        end = length_bins[i+1]
        if np.isinf(end):
            bin_labels.append(f">{start}")
        else:
            bin_labels.append(f"{start}-{end}")
    
    results = []
    
    for i, (bin_start, bin_end) in enumerate(zip(length_bins[:-1], length_bins[1:])):
        # Select gaps in this length range
        gap_indices = []
        for start, end in gaps:
            gap_len = end - start
            if bin_start <= gap_len < bin_end:
                gap_indices.extend(range(start, end))
        
        if not gap_indices:
            continue
        
        # Calculate metrics for this bin
        y_true_bin = y_true[gap_indices]
        y_pred_bin = y_pred[gap_indices]
        
        # Remove NaN
        valid = ~(np.isnan(y_true_bin) | np.isnan(y_pred_bin))
        y_true_bin = y_true_bin[valid]
        y_pred_bin = y_pred_bin[valid]
        
        if len(y_true_bin) == 0:
            continue
        
        rmse = np.sqrt(np.mean((y_pred_bin - y_true_bin)**2))
        mae = np.mean(np.abs(y_pred_bin - y_true_bin))
        bias = np.mean(y_pred_bin - y_true_bin)
        
        results.append({
            'gap_length_bin': bin_labels[i],
            'bin_start': bin_start,
            'bin_end': bin_end if not np.isinf(bin_end) else np.nan,
            'n_samples': len(y_true_bin),
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
        })
    
    return pd.DataFrame(results)


def calculate_error_by_gap_position(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gap_mask: np.ndarray,
    n_segments: int = 3,
) -> pd.DataFrame:
    """
    Calculate error by position within gaps (start, middle, end).
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth
    y_pred : np.ndarray
        Predictions
    gap_mask : np.ndarray
        Boolean mask indicating gaps
    n_segments : int
        Number of segments to divide each gap into
        
    Returns
    -------
    pd.DataFrame
        Error metrics by gap position
    """
    gaps = identify_gaps(gap_mask)
    
    # Collect errors by segment
    segment_errors = {i: [] for i in range(n_segments)}
    
    for start, end in gaps:
        gap_len = end - start
        
        if gap_len < n_segments:
            # Gap too short, put all in first segment
            segment = 0
            for idx in range(start, end):
                if not (np.isnan(y_true[idx]) or np.isnan(y_pred[idx])):
                    segment_errors[segment].append(np.abs(y_pred[idx] - y_true[idx]))
        else:
            # Divide gap into segments
            segment_size = gap_len / n_segments
            
            for idx in range(start, end):
                segment = int((idx - start) / segment_size)
                segment = min(segment, n_segments - 1)  # Ensure last index in last segment
                
                if not (np.isnan(y_true[idx]) or np.isnan(y_pred[idx])):
                    segment_errors[segment].append(np.abs(y_pred[idx] - y_true[idx]))
    
    # Calculate metrics
    segment_names = ['start', 'middle', 'end'] if n_segments == 3 else [f'segment_{i}' for i in range(n_segments)]
    
    results = []
    for seg_idx in range(n_segments):
        errors = segment_errors[seg_idx]
        
        if errors:
            results.append({
                'position': segment_names[seg_idx],
                'n_samples': len(errors),
                'mae': np.mean(errors),
                'median_ae': np.median(errors),
                'rmse': np.sqrt(np.mean(np.array(errors)**2)),
            })
    
    return pd.DataFrame(results)


def print_gap_analysis(
    error_by_length: pd.DataFrame,
    error_by_position: pd.DataFrame,
):
    """
    Print gap analysis results.
    
    Parameters
    ----------
    error_by_length : pd.DataFrame
        Error by gap length
    error_by_position : pd.DataFrame
        Error by gap position
    """
    print("\n" + "="*70)
    print("GAP-SPECIFIC ERROR ANALYSIS")
    print("="*70)
    
    print("\nError by Gap Length:")
    print("-"*70)
    print(error_by_length.to_string(index=False))
    
    print("\n\nError by Gap Position:")
    print("-"*70)
    print(error_by_position.to_string(index=False))
    
    print("="*70 + "\n")
