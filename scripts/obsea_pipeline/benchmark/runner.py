import logging
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

try:
    import torch
except ImportError:
    pass

from obsea_pipeline.config.settings import CONFIG, BENCHMARK_MODELS, HARDWARE_CONFIG, GAP_CATEGORIES
from obsea_pipeline.gaps.analysis import simulate_gaps
from obsea_pipeline.models.interpolation import interpolate_linear, interpolate_time, interpolate_spline, interpolate_polynomial
from obsea_pipeline.models.xgboost_wrapper import interpolate_xgboost, interpolate_xgboost_pro
from obsea_pipeline.models.bilstm_wrapper import interpolate_bilstm
from obsea_pipeline.models.varma_wrapper import interpolate_varma
from obsea_pipeline.models.deep_learning import (
    interpolate_missforest, interpolate_saits, interpolate_imputeformer, interpolate_brits, interpolate_brits_pro
)

logger = logging.getLogger(__name__)

def benchmark_gap_filling(df: pd.DataFrame, test_variable: str = 'TEMP', gap_categories: list = None) -> pd.DataFrame:
    """
    Benchmark interpolation methods across different gap categories using block masking.
    """
    logger.info(f"Benchmarking gap filling on {test_variable}...")
    
    if gap_categories is None:
        gap_categories = ['micro', 'short', 'medium', 'long']
        
    if test_variable not in df.columns:
        logger.error(f"Variable {test_variable} not found")
        return pd.DataFrame()

    # Define durations in points (assuming 30min data)
    cat_params = {
        'micro': {'n_gaps': 50, 'min_pts': 1, 'max_pts': 1}, 
        'short': {'n_gaps': 20, 'min_pts': 2, 'max_pts': 12},
        'medium': {'n_gaps': 5, 'min_pts': 12, 'max_pts': 144}, 
        'long': {'n_gaps': 2, 'min_pts': 144, 'max_pts': 1440}, 
        'extended': {'n_gaps': 2, 'min_pts': 1440, 'max_pts': 2880}, 
        'gigant': {'n_gaps': 1, 'min_pts': 2880, 'max_pts': 5000}    
    }
    
    results = []
    numeric_df = df.select_dtypes(include=[np.number])
    candidates = [c for c in numeric_df.columns if c != test_variable and '_QC' not in c and '_STD' not in c]
    corrs = numeric_df[candidates].corrwith(numeric_df[test_variable]).abs().sort_values(ascending=False)
    predictor_vars = corrs.head(5).index.tolist()
    
    methods = [m for m, enabled in BENCHMARK_MODELS.items() if enabled]
    gpu_methods = ['bilstm', 'xgboost', 'xgboost_pro', 'saits', 'imputeformer', 'brits', 'brits_pro']
    cpu_methods = [m for m in methods if m not in gpu_methods]
    
    def evaluate_result(interpolated, gap_mask, true_values, cat, met):
        predicted = interpolated.loc[gap_mask]
        valid_mask = predicted.notna() & true_values.notna()
        n_valid = valid_mask.sum()
        
        if n_valid == 0: return None
        
        predicted_valid = predicted[valid_mask]
        true_valid = true_values[valid_mask]
        
        rmse = np.sqrt(np.mean((predicted_valid - true_valid)**2))
        mae = np.mean(np.abs(predicted_valid - true_valid))
        data_range = df[test_variable].max() - df[test_variable].min()
        tolerance = 0.05 * data_range
        precision = (np.abs(predicted_valid - true_valid) < tolerance).mean() * 100
        
        ss_res = np.sum((true_valid - predicted_valid)**2)
        ss_tot = np.sum((true_valid - true_valid.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        return {
            'Category': cat, 'Method': met, 'RMSE': round(rmse, 4), 'MAE': round(mae, 4),
            'R2': round(r2, 4), 'Precision_%': round(precision, 2)
        }
    
    def run_model(method, df_test, test_var, pred_vars, max_gap_size=None):
        try:
            logger.info(f"    [GPU/CPU|{method.upper()}] Starting interpolation...")
            if method == 'linear': interp = interpolate_linear(df_test[test_var])
            elif method == 'time': interp = interpolate_time(df_test[test_var])
            elif method == 'splines': interp = interpolate_spline(df_test[test_var])
            elif method == 'polynomial': interp = interpolate_polynomial(df_test[test_var])
            elif method == 'varma': interp = interpolate_varma(df_test) # Using simplified
            elif method == 'bilstm': interp = interpolate_bilstm(df_test, test_var, max_gap=max_gap_size)
            elif method == 'xgboost': interp = interpolate_xgboost(df_test, test_var, max_gap=max_gap_size)
            elif method == 'xgboost_pro': interp = interpolate_xgboost_pro(df_test, test_var, max_gap=max_gap_size)
            elif method == 'missforest': interp = interpolate_missforest(df_test, test_var, predictor_vars=pred_vars)
            elif method == 'saits': interp = interpolate_saits(df_test, test_var, predictor_vars=pred_vars)
            elif method == 'imputeformer': interp = interpolate_imputeformer(df_test, test_var, predictor_vars=pred_vars, max_gap_size=max_gap_size)
            elif method == 'brits': interp = interpolate_brits(df_test, test_var, predictor_vars=pred_vars, max_gap_size=max_gap_size)
            elif method == 'brits_pro': interp = interpolate_brits_pro(df_test, test_var, predictor_vars=pred_vars)
            else: interp = None
            return method, interp
        except Exception as e:
            logger.error(f"    [ERROR|{method.upper()}] FAILED: {e}")
            return method, None

    for category in gap_categories:
        if category not in cat_params: continue
        params = cat_params[category]
        logger.info(f"  [{category.upper()}] Simulating {params['n_gaps']} gaps")
        
        df_test = df.copy()
        
        # Here we extract an inline implementation of simulate_gaps since the original
        # script's simulate_gaps may have differed slightly from our analysis.py. 
        # But we will trust our own gaps.analysis.simulate_gaps which we ported earlier.
        df_simulated, truth_mask = simulate_gaps(
            df, [test_variable], 
            missing_ratio=0.1, 
            lengths=[params['max_pts']]
        )
        
        df_test[test_variable] = df_simulated[test_variable]
        gap_mask = truth_mask[test_variable]
        true_values = df[test_variable].loc[gap_mask]
        
        category_interps = {}
        
        for method in gpu_methods:
            if method not in methods: continue
            m_name, interp_absolute = run_model(method, df_test.copy(), test_variable, predictor_vars, max_gap_size=params['max_pts'])
            if interp_absolute is not None:
                category_interps[m_name] = interp_absolute
                
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except: pass
            
        n_jobs_cpu = HARDWARE_CONFIG.get('joblib_n_jobs', 4) if HARDWARE_CONFIG.get('use_parallel_processing', True) else 1
        active_cpu_methods = [m for m in cpu_methods if m in methods]
        
        if active_cpu_methods:
            cpu_outputs = Parallel(n_jobs=n_jobs_cpu, prefer="threads")(
                delayed(run_model)(method, df_test.copy(), test_variable, predictor_vars, max_gap_size=params['max_pts'])
                for method in active_cpu_methods
            )
            for m_name, interp_absolute in cpu_outputs:
                if interp_absolute is not None:
                    category_interps[m_name] = interp_absolute
                    
        for method, interpolated in category_interps.items():
            res = evaluate_result(interpolated, gap_mask, true_values, category, method)
            if res:
                results.append(res)

    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
         try:
             winners = results_df.loc[results_df.groupby('Category')['RMSE'].idxmin()]
             logger.info("SCIENTIFIC RECOMMENDATION (Best Method per Category by RMSE)")
             for _, row in winners.iterrows():
                 logger.info(f"{row['Category']} -> {row['Method']} (RMSE: {row['RMSE']})")
         except Exception:
             pass
             
    return results_df
