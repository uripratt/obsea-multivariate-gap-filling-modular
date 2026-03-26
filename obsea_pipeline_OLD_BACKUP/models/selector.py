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
from typing import List

logger = logging.getLogger(__name__)

def apply_selected_model(df, col, method, gap=None, predictor_vars: List[str] = None):
    """
    Internal router of methods to interpolation functions.
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
        df[col] = interpolate_bilstm(df, col, predictor_vars=predictor_vars, max_gap_size=max_gap)
    elif method in ['xgboost', 'xgboost_pro']:
        prediction, _ = interpolate_xgboost_pro(df, col, predictor_vars=predictor_vars, max_gap_size=max_gap)
        df[col] = prediction
    else:
        logger.warning(f"Unknown method '{method}'. Defaulting to 'time' for {col}.")
        df[col] = interpolate_time(df[col], max_gap)
        
    return df

def selective_interpolation(df, method=None):
    """
    Scale-Aware Core Engine.
    If method is None and SMART_SELECTION is True: it iterates through each gap in the DataFrame
    physically and injects the best model based on its size (Micro, Short or Gigant).
    """
    logger.info("Applying Predictive Interpoaltion...")
    df_imputed_final = df.copy()
    df_masked_base = df.copy() # Dataset inmutable con NaNs originales (Ground Truth para los predictores)

    # Pre-calculate global correlation matrix for predictor selection
    numeric_df = df.select_dtypes(include=[np.number])
    full_corr_matrix = numeric_df.corr().abs()

    # Option 1: Apply a single model to the ENTIRE dataset
    if not SMART_SELECTION or method is not None:
        target_model = method if method else SINGLE_BEST_MODEL
        logger.info(f"  Monolithic Mode: Applying {target_model} to all gaps.")
        
        for col in df_imputed_final.columns:
            if df_imputed_final[col].isna().any():
                # Identify Top 5 physical predictors for this specific column
                if col in full_corr_matrix.columns:
                    predictor_vars = full_corr_matrix[col].sort_values(ascending=False)[1:6].index.tolist()
                else:
                    predictor_vars = []
                
                # Apply model on the base masked dataframe
                df_temp = apply_selected_model(df_masked_base.copy(), col, target_model, predictor_vars=predictor_vars)
                df_imputed_final[col] = df_imputed_final[col].fillna(df_temp[col])
                
        return df_imputed_final

    # Option 2: The PhD Scale-Aware Strategy (by Variable and Gap)
    logger.info("  Scale-Aware Mode: Classifying gaps and distributing models free of Temporal Leakage...")
    
    for col in df_masked_base.columns:
        if df_masked_base[col].dtype == object or col == 'Timestamp':
             continue
             
        mask, gaps = detect_gaps(df_masked_base[col])
        if not gaps: continue
            
        logger.debug(f"    Col {col}: Found {len(gaps)} gaps for selective imputation.")
        
        # Group gaps by algorithm to avoid instantiating the same DL model multiple times per variable
        gaps_by_method = {
            'pchip': [],
            'time': [],
            'varma': [],
            'bilstm': [],
            'xgboost_pro': []
        }
        
        for gap in gaps:
            # Select optimal algorithm (PhD Thesis Rules)
            if gap.category == 'micro':
                best_method = 'pchip'
            elif gap.category == 'short':
                best_method = 'time'
            elif gap.category == 'medium':
                best_method = 'varma'
            elif gap.category in ['long', 'extended']:
                best_method = 'bilstm'
            else: # 'gigant'
                best_method = 'xgboost_pro'
                
            gaps_by_method[best_method].append(gap)
            
        # Execute models by group and stitch results back into the final dataframe
        for current_method, assigned_gaps in gaps_by_method.items():
            if not assigned_gaps: continue
            
            # Get the max gap for this group to dimension the model (e.g. XGBoost Lags)
            max_gap_for_method = max(g.length for g in assigned_gaps)
            # Create a dummy object compatible with original signature (expects object with length attribute)
            dummy_gap = type('DummyGap', (), {'length': max_gap_for_method})()
            
            logger.info(f"      -> Executing {current_method} for {len(assigned_gaps)} gaps (max size: {max_gap_for_method})")
            
            # Identify Top 5 physical predictors for this specific column
            if col in full_corr_matrix.columns:
                predictor_vars = full_corr_matrix[col].sort_values(ascending=False)[1:6].index.tolist()
            else:
                predictor_vars = []

            # ELIMINATE TEMPORAL LEAKAGE: Model inference is ALWAYS on raw NaNs
            df_temp_inference = apply_selected_model(df_masked_base.copy(), col, current_method, dummy_gap, predictor_vars=predictor_vars)
            
            # Paste only the exact segments assigned to this algorithm into the final dataframe
            for gap in assigned_gaps:
                gap_slice = slice(gap.start, gap.end + 1)
                df_imputed_final[col].iloc[gap_slice] = df_temp_inference[col].iloc[gap_slice]
            
    return df_imputed_final

def process_variable_cpu(series, model_name):
    """
    Wrapper para paralelización local de modelos de Interpolación Base.
    """
    # En desarrollo. Solo usado por benchmark/runner.py para spawnear pools de procesos.
    pass
