import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Configurar matplotlib para ser 'headless' (no interactiva) en servidores
import matplotlib
matplotlib.use('Agg')

def plot_benchmark_results(results_df: pd.DataFrame, var_name: str, output_path: str = None):
    """
    Dibuja grafica comparativa del RMSE entre todos los algoritmos 
    para las distintas categorias de Gap.
    """
    if results_df.empty: return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # RMSE Plot
    sns.barplot(data=results_df, x='Category', y='RMSE', hue='Method', ax=axes[0], palette='viridis')
    axes[0].set_title(f'{var_name} Gap Filling RMSE (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Precision Plot (if available in results)
    if 'Precision_%' in results_df.columns:
        sns.barplot(data=results_df, x='Category', y='Precision_%', hue='Method', ax=axes[1], palette='viridis')
        axes[1].set_title(f'{var_name} Gap Filling Precision (Higher is Better)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Precision (%)', fontsize=12)
        axes[1].set_ylim(0, 105)
        axes[1].grid(True, alpha=0.3)
    else:
        # Fallback to MAE if Precision is not there
        sns.barplot(data=results_df, x='Category', y='MAE', hue='Method', ax=axes[1], palette='magma')
        axes[1].set_title(f'{var_name} Gap Filling MAE', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved Benchmark Summary Plot: {output_path}")
    
    plt.close()

def plot_gap_heatmap(df: pd.DataFrame, output_path: str = None):
    """
    Mapeo visual de la integridad del histórico de datos (Dónde faltan valores).
    """
    plt.figure(figsize=(16, 10))
    # Usar una muestra de colores simple (Negro = Data, Blanco = NaN)
    sns.heatmap(df.isna().T, cbar=False, cmap='binary', yticklabels=True)
    plt.title("Sensor Missing Data Landscape (Black=Missing, White=Valid)")
    plt.xlabel('Time Index')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame, output_path: str = None):
    """
    Matriz de calor de correlaciones de Pearson entre variables. Muy útil
    para entender qué predictores le sirven más a XGBoost/VARMA.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    # Diagonal mask to hide upper triangle cleanly
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Multivariate Parameter Pearson Correlation")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()

def plot_gap_example_per_model(df_original: pd.DataFrame, df_filled: pd.DataFrame, 
                               gap_mask: pd.Series, method: str, variable: str = "VAR", 
                               category: str = "Unknown", output_path: str = None):
    """
    Pinta el valor original artificialmente eliminado en rojo/naranja,
    vs el valor re-creado por la red neuronal/estadística en Azul, para comparar visualmente
    qué tan fiel es el dibujo de la curva.
    """
    plt.figure(figsize=(12, 6))
    
    # Identify the gap indices
    # INSTEAD of all gaps, find the largest contiguous gap block to show as example
    if not gap_mask.any(): return
    
    # Find contiguous blocks of True in gap_mask
    mask_values = gap_mask.values
    is_gap = mask_values.astype(int)
    # Detect starts and ends
    diff = np.diff(np.concatenate([[0], is_gap, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    
    if len(starts) == 0: return
    
    # Find the largest gap to show as a representative example
    lengths = ends - starts + 1
    largest_idx = np.argmax(lengths)
    
    start_pos = starts[largest_idx]
    end_pos = ends[largest_idx]
    
    start_idx = gap_mask.index[start_pos]
    end_idx = gap_mask.index[end_pos]
    
    # Context window: +- 3 * gap_length or at least 100 points
    gap_len = end_pos - start_pos + 1
    context = max(100, int(gap_len * 1.5))
    
    pos_start = max(0, start_pos - context)
    pos_end = min(len(df_original)-1, end_pos + context)
    window_idx = df_original.index[pos_start:pos_end]
    
    # Plotting
    # We plot the ORIGINAL truth (which might be missing in some parts if the sensor failed)
    plt.plot(df_original.loc[window_idx], color='#27AE60', label='Ground Truth (Original)', alpha=0.6, linewidth=1.5)
    
    # Plot the reconstruction
    plt.plot(df_filled.loc[window_idx], color='#2980B9', linestyle='--', label=f'{method} Reconstruction', alpha=0.8, linewidth=2)
    
    # Highlight the specific gap being evaluated
    plt.axvspan(start_idx, end_idx, color='red', alpha=0.1, label=f'Target Gap ({gap_len} pts)')
    
    plt.title(f'Imputation Fidelity Check | Var: {variable} | Cat: {category} | Method: {method.upper()}', fontsize=12, fontweight='bold')
    plt.ylabel(variable)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()

def save_gap_prediction_plot(*args, **kwargs):
    """Legacy alias for backward compatibility."""
    return plot_gap_example_per_model(*args, **kwargs)

