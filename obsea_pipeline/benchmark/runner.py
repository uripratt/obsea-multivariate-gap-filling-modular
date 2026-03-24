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
from obsea_pipeline.models.varma_wrapper import interpolate_varma

# Deep Learning Wrappers (Graceful fail if Torch/CUDA is broken)
try:
    from obsea_pipeline.models.bilstm_wrapper import interpolate_bilstm
    from obsea_pipeline.models.deep_learning import (
        interpolate_missforest, interpolate_saits, interpolate_imputeformer, interpolate_brits, interpolate_brits_pro
    )
    DL_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"Deep Learning benchmarking components unavailable: {e}")
    DL_AVAILABLE = False

from obsea_pipeline.utils.visualization import plot_gap_example_per_model, plot_benchmark_results

logger = logging.getLogger(__name__)

def benchmark_gap_filling(df: pd.DataFrame, test_variable: str = 'TEMP', gap_categories: list = None) -> pd.DataFrame:
    """
    Benchmark interpolation methods across different gap categories using block masking.
    """
    logger.info(f"Benchmarking gap filling on {test_variable}...")
    
    # Setup Output Directories
    output_dir = Path(CONFIG.get('output_dir', 'output_lup')) / 'benchmarks'
    gap_examples_dir = output_dir / 'gap_examples'
    gap_examples_dir.mkdir(parents=True, exist_ok=True)
    
    if gap_categories is None:
        gap_categories = ['micro', 'short', 'medium', 'long']
        
    if test_variable not in df.columns:
        logger.error(f"Variable {test_variable} not found")
        return pd.DataFrame()

    # Define durations in points (assuming 30min data)
    cat_params = {
        'micro': {'n_gaps': 50, 'min_pts': 1, 'max_pts': 12}, 
        'short': {'n_gaps': 20, 'min_pts': 13, 'max_pts': 48},
        'medium': {'n_gaps': 5, 'min_pts': 49, 'max_pts': 144}, 
        'long': {'n_gaps': 2, 'min_pts': 145, 'max_pts': 336}, 
        'extended': {'n_gaps': 2, 'min_pts': 337, 'max_pts': 1440}, 
        'gigant': {'n_gaps': 1, 'min_pts': 1441, 'max_pts': 5000}    
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
        
        # Ensure we are working with a Series to avoid DataFrame sum ambiguity
        if isinstance(predicted, pd.DataFrame):
            predicted = predicted.iloc[:, 0]
        if isinstance(true_values, pd.DataFrame):
            true_values = true_values.iloc[:, 0]
            
        valid_mask = predicted.notna() & true_values.notna()
        n_valid = int(np.sum(valid_mask.values))
        
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
        
        # New: Plot individual gap example
        try:
            plot_path = gap_examples_dir / f"gap_{cat}_{met}.png"
            # We pass df as df_original to show the Ground Truth
            plot_gap_example_per_model(
                df_original=df[test_variable], 
                df_filled=interpolated, 
                gap_mask=gap_mask, 
                method=met, 
                variable=test_variable,
                category=cat,
                output_path=str(plot_path)
            )
        except Exception as e:
            logger.warning(f"Failed to generate plot for {met} in cat {cat}: {e}")

        try:
            from scipy.stats import wasserstein_distance
            # Wasserstein Distance (Earth Mover's Distance) measures distributional similarity
            w_dist = wasserstein_distance(true_valid, predicted_valid)
        except Exception:
            w_dist = np.nan

        return {
            'Category': cat, 'Method': met, 'RMSE': round(rmse, 4), 'MAE': round(mae, 4),
            'R2': round(r2, 4), 'Precision_%': round(precision, 2),
            'Wasserstein': round(w_dist, 4) if not np.isnan(w_dist) else None
        }
    
    def run_model(method, df_test, test_var, pred_vars, max_gap_size=None):
        try:
            logger.info(f"    [GPU/CPU|{method.upper()}] Starting interpolation...")
            if method == 'linear': interp = interpolate_linear(df_test[test_var])
            elif method == 'time': interp = interpolate_time(df_test[test_var])
            elif method == 'splines': interp = interpolate_spline(df_test[test_var])
            elif method == 'polynomial': interp = interpolate_polynomial(df_test[test_var])
            elif method == 'varma': interp = interpolate_varma(df_test, target_var=test_var, max_gap=max_gap_size)
            elif method == 'bilstm': interp = interpolate_bilstm(df_test, test_var, max_gap_size=max_gap_size)
            elif method == 'xgboost': 
                interp, _ = interpolate_xgboost(df_test, test_var, max_gap_size=max_gap_size)
            elif method == 'xgboost_pro': 
                interp, _ = interpolate_xgboost_pro(df_test, test_var, max_gap_size=max_gap_size)
            elif method == 'missforest': interp = interpolate_missforest(df_test, test_var, predictor_vars=pred_vars)
            elif method == 'saits': interp = interpolate_saits(df_test, test_var, predictor_vars=pred_vars, max_gap_size=max_gap_size)
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
        
        # Simulate varying lengths natively to test model resilience inside its own class
        length_range = list(range(params['min_pts'], params['max_pts'] + 1))
        
        # Simulate gaps
        df_simulated, truth_mask = simulate_gaps(
            df, [test_variable], 
            missing_ratio=0.05, # Adjusted to avoid saturating context
            lengths=length_range
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
         # Export Benchmark Results as CSV
         results_df.to_csv(output_dir / 'interpolation_comparison.csv', index=False)
         
         # Generate Benchmark Comparison Plot
         plot_benchmark_results(
             results_df, 
             test_variable, 
             output_path=str(output_dir / 'benchmark_results.png')
         )
         
         try:
             winners = results_df.loc[results_df.groupby('Category')['RMSE'].idxmin()]
             logger.info("SCIENTIFIC RECOMMENDATION (Best Method per Category by RMSE)")
             for _, row in winners.iterrows():
                 logger.info(f"{row['Category']} -> {row['Method']} (RMSE: {row['RMSE']})")
         except Exception:
             pass
             
    return results_df
