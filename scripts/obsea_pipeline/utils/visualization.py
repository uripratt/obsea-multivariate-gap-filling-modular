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
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=results_df, x='Category', y='RMSE', hue='Method')
    plt.title(f'Benchmark Results for {var_name} (Lower is Better)')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
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

def plot_gap_example_per_model(df_original: pd.DataFrame, df_filled: pd.DataFrame, gap_mask: pd.Series, method: str, output_path: str = None):
    """
    Pinta el valor original artificialmente eliminado en rojo/naranja,
    vs el valor re-creado por la red neuronal/estadística en Azul, para comparar visualmente
    qué tan fiel es el dibujo de la curva.
    """
    plt.figure(figsize=(12, 6))
    
    # Slice around the gap context to not plot 10 years
    start_idx = gap_mask[gap_mask].index.min()
    end_idx = gap_mask[gap_mask].index.max()
    
    if pd.isna(start_idx) or pd.isna(end_idx): return
    
    # Agregar márgenes al plot (ej. +- 50 puntos a izquierda y derecha)
    # Extraemos por position
    pos_start = max(0, df_original.index.get_loc(start_idx) - 50)
    pos_end = min(len(df_original)-1, df_original.index.get_loc(end_idx) + 50)
    
    window_idx = df_original.index[pos_start:pos_end]
    
    plt.plot(df_original.loc[window_idx], color='green', label='Original Truth', alpha=0.5, linewidth=2)
    plt.plot(df_filled.loc[window_idx].loc[gap_mask], color='red', linestyle='--', label=f'{method} Imputation', linewidth=2)
    
    plt.title(f'Prediction Fidelity vs Truth: {method}')
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()

def save_gap_prediction_plot(*args, **kwargs):
    """Legacy alias to avoid modifying old benchmarking codes."""
    return plot_gap_example_per_model(*args, **kwargs)
