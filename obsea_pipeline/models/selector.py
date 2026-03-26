import logging
import numpy as np

from obsea_pipeline.config.settings import GAP_CATEGORIES, SMART_SELECTION, SINGLE_BEST_MODEL
from obsea_pipeline.models.interpolation import (
    interpolate_pchip, interpolate_linear, interpolate_time,
    interpolate_spline, interpolate_polynomial
)
from obsea_pipeline.models.varma_wrapper import interpolate_varma
from obsea_pipeline.models.xgboost_wrapper import interpolate_xgboost_pro, interpolate_xgboost
# Deep Learning & Multivariate Imports (Graceful fail if Torch/CUDA is broken)
try:
    from obsea_pipeline.models.deep_learning import (
        interpolate_saits, interpolate_brits, interpolate_imputeformer, interpolate_brits_pro
    )
    from obsea_pipeline.models.bilstm_wrapper import interpolate_bilstm
    DL_AVAILABLE = True
except (ImportError, Exception) as e:
    # Error fatal de Torch/CUDA (suele pasar en entornos con NCCL roto o CUDA mal configurado)
    logger.warning(f"  [IMPORT WARNING] Deep Learning / Torch models could not be loaded: {e}.")
    logger.warning("  The pipeline will fallback to XGBoost/VARMA/Linear for these gaps.")
    DL_AVAILABLE = False
    # Definir stubs para evitar NameError
    interpolate_saits = interpolate_brits = interpolate_imputeformer = interpolate_brits_pro = None
    interpolate_bilstm = None

from obsea_pipeline.gaps.analysis import detect_gaps

logger = logging.getLogger(__name__)

def apply_selected_model(df, col, method, gap=None):
    """
    Ruter interno de métodos a funciones con soporte de Fallback por dependencias rotas.
    """
    max_gap = gap.length if gap else None
    
    if method == 'linear':
        df[col] = interpolate_linear(df[col], max_gap)
    elif method == 'time':
        df[col] = interpolate_time(df[col], max_gap)
    elif method == 'pchip':
        df[col] = interpolate_pchip(df[col], max_gap)
    elif method == 'spline_linear':
        df[col] = interpolate_spline(df[col], order=1, max_gap=max_gap)
    elif method == 'varma':
        df[col] = interpolate_varma(df, col, max_gap_size=max_gap)
    elif method == 'bilstm':
        if DL_AVAILABLE and interpolate_bilstm:
            df[col] = interpolate_bilstm(df, col, max_gap_size=max_gap)
        else:
            logger.warning(f"  [FALLBACK] BiLSTM no disponible para {col}. Usando XGBoost Pro.")
            prediction, _ = interpolate_xgboost_pro(df, col, max_gap_size=max_gap)
            df[col] = prediction
    elif method == 'saits':
        if DL_AVAILABLE and interpolate_saits:
            df[col] = interpolate_saits(df, col, max_gap_size=max_gap)
        else:
            logger.warning(f"  [FALLBACK] SAITS no disponible para {col}. Usando XGBoost Pro.")
            prediction, _ = interpolate_xgboost_pro(df, col, max_gap_size=max_gap)
            df[col] = prediction
    elif method == 'imputeformer':
        if DL_AVAILABLE and interpolate_imputeformer:
            df[col] = interpolate_imputeformer(df, col, max_gap_size=max_gap)
        else:
            logger.warning(f"  [FALLBACK] ImputeFormer no disponible para {col}. Usando XGBoost Pro.")
            prediction, _ = interpolate_xgboost_pro(df, col, max_gap_size=max_gap)
            df[col] = prediction
    elif method == 'brits':
        if DL_AVAILABLE and interpolate_brits:
            df[col] = interpolate_brits(df, col, max_gap_size=max_gap)
        else:
            logger.warning(f"  [FALLBACK] BRITS no disponible para {col}. Usando XGBoost Pro.")
            prediction, _ = interpolate_xgboost_pro(df, col, max_gap_size=max_gap)
            df[col] = prediction
    elif method == 'brits_pro':
        if DL_AVAILABLE and interpolate_brits_pro:
            df[col] = interpolate_brits_pro(df, col)
        else:
            logger.warning(f"  [FALLBACK] BRITS Pro no disponible para {col}. Usando XGBoost Pro.")
            prediction, _ = interpolate_xgboost_pro(df, col, max_gap_size=max_gap)
            df[col] = prediction
    elif method == 'xgboost':
        prediction, _ = interpolate_xgboost(df, col, max_gap_size=max_gap)
        df[col] = prediction
    elif method == 'xgboost_pro':
        prediction, _ = interpolate_xgboost_pro(df, col, max_gap_size=max_gap)
        df[col] = prediction
    else:
        logger.warning(f"Metodo '{method}' desconocido. Usando 'time' por defecto para {col}.")
        df[col] = interpolate_time(df[col], max_gap)
        
    return df

def selective_interpolation(df, method=None):
    """
    Motor central Scale-Aware.
    Si method is None y SMART_SELECTION es True: iterará cada gap del DataFrame 
    físicamente e inyectará el mejor modelo según es Micro, Short o Gigant.
    """
    logger.info("Applying Predictive Interpoaltion...")
    df_imputed_final = df.copy()
    df_masked_base = df.copy() # Dataset inmutable con NaNs originales (Ground Truth para los predictores)

    # Opción 1: Aplicar un único modelo a TODO el dataset a "fuego"
    if not SMART_SELECTION or method is not None:
        target_model = method if method else SINGLE_BEST_MODEL
        logger.info(f"  Modo Monolítico: Aplicando {target_model} a todos los gaps.")
        
        for col in df_imputed_final.columns:
            if df_imputed_final[col].isna().any():
                # Apply model on the base masked dataframe
                df_temp = apply_selected_model(df_masked_base.copy(), col, target_model)
                df_imputed_final[col] = df_imputed_final[col].fillna(df_temp[col])
                
        return df_imputed_final

    # Opción 2: The PhD Scale-Aware Strategy (por Variable y Gap)
    logger.info("  Modo Scale-Aware: Clasificando gaps y distribuyendo algoritmos...")
    
    # Pre-filtrar columnas para el progreso
    cols_to_process = [c for c in df_masked_base.columns if df_masked_base[c].dtype != object and c != 'Timestamp']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        transient=False
    ) as progress:
        
        overall_task = progress.add_task("[yellow]Interpolating Variables...", total=len(cols_to_process))
        
        for col in cols_to_process:
            progress.update(overall_task, description=f"[bold cyan]Processing {col}...")
             
            mask, gaps = detect_gaps(df_masked_base[col])
            if not gaps: 
                progress.advance(overall_task)
                continue
                
            # Agrupamos los gaps por algoritmo
            gaps_by_method = {m: [] for m in ['linear', 'pchip', 'time', 'varma', 'xgboost_pro', 'bilstm', 'saits', 'brits', 'imputeformer']}
            
            for gap in gaps:
                if gap.category == 'micro': best_method = 'linear'
                elif gap.category == 'short': best_method = 'xgboost_pro'
                elif gap.category == 'medium': best_method = 'bilstm'
                elif gap.category in ['long', 'extended']: best_method = 'imputeformer'
                else: best_method = 'brits'
                gaps_by_method[best_method].append(gap)
                
            # Ejecutar modelos
            for current_method, assigned_gaps in gaps_by_method.items():
                if not assigned_gaps: continue
                
                max_gap_for_method = max(g.length for g in assigned_gaps)
                dummy_gap = type('DummyGap', (), {'length': max_gap_for_method})()
                
                logger.info(f"      -> Running {current_method} for {len(assigned_gaps)} gaps (max size: {max_gap_for_method})")
                df_temp_inference = apply_selected_model(df_masked_base.copy(), col, current_method, dummy_gap)
                
                for gap in assigned_gaps:
                    gap_slice = slice(gap.start_idx, gap.end_idx + 1)
                    df_imputed_final[col].iloc[gap_slice] = df_temp_inference[col].iloc[gap_slice]
            
            progress.advance(overall_task)
            
    return df_imputed_final

def process_variable_cpu(series, model_name):
    """
    Wrapper para paralelización local de modelos de Interpolación Base.
    """
    # En desarrollo. Solo usado por benchmark/runner.py para spawnear pools de procesos.
    pass
