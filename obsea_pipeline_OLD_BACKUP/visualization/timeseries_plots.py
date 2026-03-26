"""
Visualization utilities for time series and model comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_error_by_gap_length(
    error_by_length_dict: Dict[str, pd.DataFrame],
    metric: str = 'rmse',
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot error metrics by gap length for multiple models.
    
    Parameters
    ----------
    error_by_length_dict : dict
        Dictionary mapping model names to error_by_length DataFrames
    metric : str
        Metric to plot ('rmse', 'mae', 'bias')
    output_path : str, optional
        Path to save figure
    title : str, optional
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = sns.color_palette("husl", len(error_by_length_dict))
    
    for (model_name, error_df), color in zip(error_by_length_dict.items(), colors):
        if error_df.empty or metric not in error_df.columns:
            continue
        
        # Use bin_start for x-axis
        x = error_df['bin_start'].values
        y = error_df[metric].values
        
        ax.plot(x, y, marker='o', linewidth=2, markersize=8, 
                label=model_name, color=color)
    
    ax.set_xlabel('Gap Length (time steps)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{metric.upper()} vs Gap Length', fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Log scale for x-axis if large range
    if error_df['bin_start'].max() > 100:
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    
    return fig, ax


def plot_model_comparison_bars(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['rmse', 'mae', 'bias'],
    output_path: Optional[str] = None,
    title: str = "Model Comparison",
):
    """
    Create bar plots comparing models across metrics.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model names to metric dictionaries
    metrics : list of str
        Metrics to plot
    output_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    df = pd.DataFrame(results).T
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = sns.color_palette("Set2", len(df))
    
    for ax, metric in zip(axes, metrics):
        if metric not in df.columns:
            continue
        
        df_sorted = df.sort_values(metric)
        
        bars = ax.barh(range(len(df_sorted)), df_sorted[metric], color=colors)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted.index, fontsize=11)
        ax.set_xlabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            value = row[metric]
            ax.text(value, i, f'  {value:.4f}', va='center', fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    
    return fig, axes


def plot_gap_pattern_comparison(
    pattern_results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = 'rmse',
    output_path: Optional[str] = None,
):
    """
    Plot model performance across different gap patterns.
    
    Parameters
    ----------
    pattern_results : dict
        Nested dict: {pattern_name: {model_name: {metrics}}}
    metric : str
        Metric to plot
    output_path : str, optional
        Path to save figure
    """
    # Reorganize data
    data = []
    for pattern, models in pattern_results.items():
        for model, metrics in models.items():
            if metric in metrics:
                data.append({
                    'Pattern': pattern,
                    'Model': model,
                    metric: metrics[metric]
                })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Grouped bar plot
    patterns = df['Pattern'].unique()
    models = df['Model'].unique()
    
    x = np.arange(len(patterns))
    width = 0.8 / len(models)
    
    colors = sns.color_palette("husl", len(models))
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        values = [model_data[model_data['Pattern'] == p][metric].values[0] 
                  if len(model_data[model_data['Pattern'] == p]) > 0 else 0
                  for p in patterns]
        
        ax.bar(x + i * width, values, width, label=model, color=colors[i])
    
    ax.set_xlabel('Gap Pattern', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
    ax.set_title(f'{metric.upper()} by Gap Pattern and Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([p.replace('_', ' ').title() for p in patterns], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    
    return fig, ax


def plot_timeseries_with_gaps(
    df_original: pd.DataFrame,
    df_imputed: pd.DataFrame,
    gap_mask: pd.Series,
    variable: str,
    model_name: str,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    output_path: Optional[str] = None,
):
    """
    Plot original vs imputed time series highlighting gaps.
    
    Parameters
    ----------
    df_original : pd.DataFrame
        Original data (ground truth)
    df_imputed : pd.DataFrame
        Imputed data
    gap_mask : pd.Series
        Boolean mask indicating gap positions
    variable : str
        Variable name to plot
    model_name : str
        Model name for title
    start_idx : int, optional
        Start index for zooming
    end_idx : int, optional
        End index for zooming
    output_path : str, optional
        Path to save figure
    """
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(df_original)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Slice data
    time = df_original.index[start_idx:end_idx]
    original = df_original[variable].iloc[start_idx:end_idx]
    imputed = df_imputed[variable].iloc[start_idx:end_idx]
    gaps = gap_mask.iloc[start_idx:end_idx]
    
    # Plot original data
    ax.plot(time, original, 'o-', label='Original', color='black', 
            linewidth=1.5, markersize=3, alpha=0.7)
    
    # Plot imputed data in gaps
    ax.plot(time[gaps], imputed[gaps], 's-', label=f'{model_name} (imputed)', 
            color='red', linewidth=2, markersize=5, alpha=0.8)
    
    # Highlight gap regions
    gap_starts = np.where(np.diff(np.concatenate([[False], gaps.values, [False]])) == 1)[0]
    gap_ends = np.where(np.diff(np.concatenate([[False], gaps.values, [False]])) == -1)[0]
    
    for start, end in zip(gap_starts, gap_ends):
        if end > start:
            ax.axvspan(time[start], time[min(end, len(time)-1)], 
                      alpha=0.2, color='red', label='Gap' if start == gap_starts[0] else '')
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel(variable, fontsize=12, fontweight='bold')
    ax.set_title(f'{variable} - {model_name} Imputation', fontsize=14, fontweight='bold')
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=11, loc='best')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    
    return fig, ax
