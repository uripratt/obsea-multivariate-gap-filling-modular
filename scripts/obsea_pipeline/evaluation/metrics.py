"""
Evaluation metrics for gap filling models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger =logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    mask : np.ndarray, optional
        Boolean mask for valid comparisons (True = valid)
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    # Apply mask
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    # Remove NaN pairs
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    
    if len(y_true) == 0:
        logger.warning("No valid data points for metric calculation")
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'r2': np.nan,
            'n_samples': 0,
        }
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = np.nan
    
    # Additional metrics
    median_ae = np.median(np.abs(y_pred - y_true))
    percentile_95_ae = np.percentile(np.abs(y_pred - y_true), 95)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'median_ae': median_ae,
        'percentile_95_ae': percentile_95_ae,
        'bias': bias,
        'r2': r2,
        'n_samples': len(y_true),
    }


def calculate_skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    metric: str = 'rmse',
) -> float:
    """
    Calculate skillscore relative to baseline.
    
    SS = 1 - (Error_model / Error_baseline)
    
    SS > 0: Model better than baseline
    SS = 0: Model same as baseline
    SS < 0: Model worse than baseline
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth
    y_pred : np.ndarray
        Model predictions
    y_baseline : np.ndarray
        Baseline predictions
    metric : str
        Metric to use: 'rmse' or 'mae'
        
    Returns
    -------
    float
        Skill score
    """
    # Remove NaN
    valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_baseline))
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    y_baseline = y_baseline[valid]
    
    if len(y_true) == 0:
        return np.nan
    
    if metric == 'rmse':
        error_model = np.sqrt(mean_squared_error(y_true, y_pred))
        error_baseline = np.sqrt(mean_squared_error(y_true, y_baseline))
    elif metric == 'mae':
        error_model = mean_absolute_error(y_true, y_pred)
        error_baseline = mean_absolute_error(y_true, y_baseline)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if error_baseline == 0:
        return np.nan
    
    skill_score = 1 - (error_model / error_baseline)
    
    return skill_score


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print metrics in a formatted table.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics
    model_name : str
        Name of the model
    """
    print(f"\n{model_name} Performance:")
    print("-" * 50)
    print(f"  RMSE:            {metrics['rmse']:.4f}")
    print(f"  MAE:             {metrics['mae']:.4f}")
    print(f"  Median AE:       {metrics['median_ae']:.4f}")
    print(f"  95th %ile AE:    {metrics['percentile_95_ae']:.4f}")
    print(f"  Bias:            {metrics['bias']:.4f}")
    print(f"  R²:              {metrics['r2']:.4f}")
    print(f"  N samples:       {metrics['n_samples']}")
    print("-" * 50)


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = 'rmse',
) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model names to metric dictionaries
    metric : str
        Primary metric for ranking
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    comparison = pd.DataFrame(results).T
    
    # Sort by primary metric (lower is better for RMSE/MAE)
    if metric in ['rmse', 'mae', 'bias']:
        comparison = comparison.sort_values(metric)
    else:  # R² - higher is better
        comparison = comparison.sort_values(metric, ascending=False)
    
    return comparison


def save_metrics(
    metrics: Dict[str, float],
    output_path: str,
    model_name: str,
):
    """
    Save metrics to file.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary
    output_path : str
        Output file path
    model_name : str
        Model name
    """
    import json
    from pathlib import Path
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add model name
    metrics_with_name = {'model': model_name, **metrics}
    
    # Save as JSON
    with open(output_file, 'w') as f:
        json.dump(metrics_with_name, f, indent=2)
    
    logger.info(f"Saved metrics to {output_path}")
