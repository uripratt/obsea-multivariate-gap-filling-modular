import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 1. BENCHMARK SUMMARY BAR CHART (RMSE + Precision)
# ═══════════════════════════════════════════════════════════

def plot_benchmark_results(results_df: pd.DataFrame, var_name: str, output_path: str = None):
    """
    Gráfica comparativa mejorada: RMSE + Coverage + Physical Violations.
    """
    if results_df.empty: return
    
    n_metrics = 2
    has_coverage = 'Coverage_%' in results_df.columns
    has_violations = 'Physical_Violations_%' in results_df.columns
    if has_coverage: n_metrics += 1
    if has_violations: n_metrics += 1
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 7))
    if n_metrics == 1: axes = [axes]
    
    palette = 'viridis'
    idx = 0
    
    # RMSE Plot
    sns.barplot(data=results_df, x='Category', y='RMSE', hue='Method', ax=axes[idx], palette=palette)
    axes[idx].set_title(f'{var_name} RMSE (Lower is Better)', fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('RMSE', fontsize=12)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].tick_params(axis='x', rotation=30)
    idx += 1
    
    # Precision Plot
    if 'Precision_%' in results_df.columns:
        sns.barplot(data=results_df, x='Category', y='Precision_%', hue='Method', ax=axes[idx], palette=palette)
        axes[idx].set_title(f'{var_name} Precision (Higher is Better)', fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('Precision (%)', fontsize=12)
        axes[idx].set_ylim(0, 105)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=30)
    else:
        sns.barplot(data=results_df, x='Category', y='MAE', hue='Method', ax=axes[idx], palette='magma')
        axes[idx].set_title(f'{var_name} MAE', fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('MAE', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=30)
    idx += 1
    
    # Coverage Plot (new in v2)
    if has_coverage:
        sns.barplot(data=results_df, x='Category', y='Coverage_%', hue='Method', ax=axes[idx], palette='crest')
        axes[idx].set_title(f'{var_name} Coverage (Higher is Better)', fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('Coverage (%)', fontsize=12)
        axes[idx].set_ylim(0, 105)
        axes[idx].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=30)
        idx += 1
    
    # Physical Violations Plot (new in v2)
    if has_violations:
        sns.barplot(data=results_df, x='Category', y='Physical_Violations_%', hue='Method', ax=axes[idx], palette='flare')
        axes[idx].set_title(f'{var_name} Physical Violations (Lower is Better)', fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('Violations (%)', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=30)
    
    # Remove duplicate legends except first
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved Benchmark Summary Plot: {output_path}")
    
    plt.close()


# ═══════════════════════════════════════════════════════════
# 2. GAP EXAMPLE (single model vs ground truth)
# ═══════════════════════════════════════════════════════════

def plot_gap_example_per_model(df_original: pd.DataFrame, df_filled: pd.DataFrame, 
                               gap_mask: pd.Series, method: str, variable: str = "VAR", 
                               category: str = "Unknown", output_path: str = None):
    """
    Comparación visual: Ground Truth vs reconstrucción por UN modelo.
    Mejorado: zoom al gap con contexto bilateral de ±80 pts.
    """
    plt.figure(figsize=(14, 6))
    
    gap_indices = gap_mask[gap_mask].index
    if len(gap_indices) == 0: return
    
    start_idx = gap_indices.min()
    end_idx = gap_indices.max()
    
    try:
        pos_start = max(0, df_original.index.get_loc(start_idx) - 80)
        pos_end = min(len(df_original)-1, df_original.index.get_loc(end_idx) + 80)
        window_idx = df_original.index[pos_start:pos_end]
    except (KeyError, ValueError):
        window_idx = df_original.index
        
    # Ground truth
    plt.plot(df_original.loc[window_idx], color='#27AE60', label='Ground Truth (Original)', alpha=0.7, linewidth=1.5)
    
    # Reconstruction
    plt.plot(df_filled.loc[window_idx], color='#2980B9', linestyle='--', label=f'{method} Reconstruction', alpha=0.8, linewidth=2)
    
    # Gap highlight
    plt.axvspan(start_idx, end_idx, color='red', alpha=0.1, label='Target Gap Area')
    
    # Physical bounds annotation
    observed = df_original.dropna()
    if not observed.empty:
        obs_min, obs_max = observed.min(), observed.max()
        plt.axhline(y=obs_min, color='gray', linestyle=':', alpha=0.3)
        plt.axhline(y=obs_max, color='gray', linestyle=':', alpha=0.3)
    
    plt.title(f'Imputation Fidelity Check | Var: {variable} | Cat: {category} | Method: {method.upper()}', 
              fontsize=12, fontweight='bold')
    plt.ylabel(variable)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════
# 3. SUPERIMPOSED MULTI-MODEL COMPARISON (NEW - Phase 4)
# ═══════════════════════════════════════════════════════════

def plot_multi_model_comparison(df_original: pd.Series, model_outputs: dict, 
                                 gap_mask: pd.Series, variable: str, category: str,
                                 output_path: str = None):
    """
    Plot ALL models superimposed on the same ground truth for direct comparison.
    
    model_outputs: dict of {method_name: pd.Series} with predictions.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True)
    
    gap_indices = gap_mask[gap_mask].index
    if len(gap_indices) == 0: return
    
    start_idx = gap_indices.min()
    end_idx = gap_indices.max()
    
    try:
        pos_start = max(0, df_original.index.get_loc(start_idx) - 100)
        pos_end = min(len(df_original)-1, df_original.index.get_loc(end_idx) + 100)
        window_idx = df_original.index[pos_start:pos_end]
    except (KeyError, ValueError):
        window_idx = df_original.index
    
    # Color palette for models
    colors = {
        'linear': '#95A5A6', 'time': '#BDC3C7', 'splines': '#AAB7B8',
        'xgboost': '#E67E22', 'xgboost_pro': '#D35400',
        'bilstm': '#9B59B6', 'saits': '#E74C3C', 'imputeformer': '#3498DB',
        'brits': '#1ABC9C', 'brits_pro': '#16A085', 
        'varma': '#F39C12', 'missforest': '#2ECC71'
    }
    
    # TOP: Time series comparison
    ax_top = axes[0]
    ax_top.axvspan(start_idx, end_idx, color='red', alpha=0.08, label='Gap Area')
    ax_top.plot(df_original.loc[window_idx], color='#2C3E50', linewidth=2.5, alpha=0.9, label='Ground Truth', zorder=10)
    
    for method, prediction in model_outputs.items():
        color = colors.get(method, '#7F8C8D')
        ax_top.plot(prediction.loc[window_idx], linestyle='--', linewidth=1.5, alpha=0.7, 
                    color=color, label=method.upper())
    
    ax_top.set_title(f'Multi-Model Comparison | {variable} | {category.upper()}', fontsize=14, fontweight='bold')
    ax_top.set_ylabel(variable, fontsize=12)
    ax_top.legend(loc='upper right', fontsize=8, ncol=2, frameon=True, shadow=True)
    ax_top.grid(True, alpha=0.2)
    
    # BOTTOM: Residuals (prediction - truth)
    ax_bot = axes[1]
    ax_bot.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    ax_bot.axvspan(start_idx, end_idx, color='red', alpha=0.08)
    
    gt_window = df_original.loc[window_idx]
    for method, prediction in model_outputs.items():
        color = colors.get(method, '#7F8C8D')
        residual = prediction.loc[window_idx] - gt_window
        ax_bot.plot(residual, linewidth=1.2, alpha=0.6, color=color, label=method)
    
    ax_bot.set_title('Residuals (Prediction − Ground Truth)', fontsize=11, fontweight='bold')
    ax_bot.set_ylabel('Error', fontsize=12)
    ax_bot.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"  ✓ Saved multi-model comparison: {output_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# 4. RESIDUAL DISTRIBUTION BOXPLOT (NEW - Phase 4)
# ═══════════════════════════════════════════════════════════

def plot_residual_distributions(df_original: pd.Series, model_outputs: dict,
                                 gap_mask: pd.Series, variable: str, category: str,
                                 output_path: str = None):
    """
    Box+Violin plots of residual distributions per model for a given gap category.
    """
    residuals_data = []
    gt = df_original.loc[gap_mask]
    
    for method, prediction in model_outputs.items():
        pred_gap = prediction.loc[gap_mask]
        valid = pred_gap.notna() & gt.notna()
        if valid.sum() == 0:
            continue
        residual = (pred_gap[valid] - gt[valid]).values
        for r in residual:
            residuals_data.append({'Method': method.upper(), 'Residual': r})
    
    if not residuals_data:
        return
    
    rdf = pd.DataFrame(residuals_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.violinplot(data=rdf, x='Method', y='Residual', inner=None, alpha=0.3, ax=ax, palette='viridis')
    sns.boxplot(data=rdf, x='Method', y='Residual', width=0.3, ax=ax, palette='viridis',
                flierprops=dict(markersize=2, alpha=0.5))
    
    ax.axhline(y=0, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.set_title(f'Residual Distribution (Pred − Truth) | {variable} | {category.upper()}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{variable} Residual', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        logger.info(f"  ✓ Saved residual distributions: {output_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# 5. EXISTING UTILS (Gap Heatmap, Correlation Matrix)
# ═══════════════════════════════════════════════════════════

def plot_gap_heatmap(df: pd.DataFrame, output_path: str = None):
    """Mapeo visual de la integridad del histórico de datos."""
    plt.figure(figsize=(16, 10))
    sns.heatmap(df.isna().T, cbar=False, cmap='binary', yticklabels=True)
    plt.title("Sensor Missing Data Landscape (Black=Missing, White=Valid)")
    plt.xlabel('Time Index')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame, output_path: str = None):
    """Matriz de calor de correlaciones de Pearson entre variables."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Multivariate Parameter Pearson Correlation")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()


# ═══════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════

def save_gap_prediction_plot(*args, **kwargs):
    """Legacy alias for backward compatibility."""
    return plot_gap_example_per_model(*args, **kwargs)
