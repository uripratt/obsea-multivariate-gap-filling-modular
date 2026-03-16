import logging
import numpy as np

from obsea_pipeline.config.settings import GAP_CATEGORIES, SMART_SELECTION, SINGLE_BEST_MODEL
from obsea_pipeline.models.interpolation import (
    interpolate_pchip, interpolate_linear, interpolate_time,
    interpolate_spline, interpolate_polynomial
)
from obsea_pipeline.models.varma_wrapper import interpolate_varma
from obsea_pipeline.models.xgboost_wrapper import interpolate_xgboost_pro
from obsea_pipeline.models.bilstm_wrapper import interpolate_bilstm
from obsea_pipeline.gaps.analysis import detect_gaps

logger = logging.getLogger(__name__)

def apply_selected_model(df, col, method, gap=None):
    """
    Ruter interno de métodos a funciones.
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
        df[col] = interpolate_bilstm(df, col, max_gap_size=max_gap)
    elif method in ['xgboost', 'xgboost_pro']:
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
    df_result = df.copy()

    # Opción 1: Aplicar un único modelo a TODO el dataset a "fuego"
    if not SMART_SELECTION or method is not None:
        target_model = method if method else SINGLE_BEST_MODEL
        logger.info(f"  Modo Monolítico: Aplicando {target_model} a todos los gaps.")
        
        for col in df_result.columns:
            if df_result[col].isna().any():
                df_result = apply_selected_model(df_result, col, target_model)
                
        return df_result

    # Opción 2: The PhD Scale-Aware Strategy (por Variable y Gap)
    logger.info("  Modo Scale-Aware: Clasificando gaps y distribuyendo algoritmos...")
    
    for col in df_result.columns:
        if df_result[col].dtype == object or col == 'Timestamp':
             continue
             
        mask, gaps = detect_gaps(df_result[col])
        if not gaps: continue
            
        logger.debug(f"    Col {col}: Encontrados {len(gaps)} gaps para imputación selectiva.")
        
        # En una arquitectura matricial 100% aislada, extraeríamos las "view" del 
        # entorno del Gap para evitar que el ML pise los huecos procesados por PCHIP,
        # pero es computacionalmente ineficiente. 
        # Ordenamos los Gaps de "fáciles" a "díficiles" para estuchar primero los 
        # micro y dar pista limpia a XGBoost de los gigantes.
        
        gaps_sorted = sorted(gaps, key=lambda g: g.length)
        
        for gap in gaps_sorted:
            # Seleccionar algoritmo óptimo (Reglas de Tesis Doctoral)
            if gap.category == 'micro':
                # <= 6h: PCHIP es rey indiscutible
                best_method = 'pchip'
            elif gap.category == 'short':
                # 6h - 24h: Time o PCHIP
                best_method = 'time'
            elif gap.category == 'medium':
                # 1 - 3 días: Dependencia multivariante local 
                best_method = 'varma'
            elif gap.category in ['long', 'extended']:
                # 3 - 10 días: Secuencias puras
                best_method = 'bilstm'
            else: # 'gigant'
                # > 10 días: XGBoost Pro con decay para evitar rotura histórica
                best_method = 'xgboost_pro'
                
            # Extraer entorno del target y aplicar (En este wrapper lo aplicamos al Df)
            # Para refinar en Producción, limitar el df.slice(start-margin : end+margin)
            df_result = apply_selected_model(df_result, col, best_method, gap)
            
    return df_result

def process_variable_cpu(series, model_name):
    """
    Wrapper para paralelización local de modelos de Interpolación Base.
    """
    # En desarrollo. Solo usado por benchmark/runner.py para spawnear pools de procesos.
    pass
