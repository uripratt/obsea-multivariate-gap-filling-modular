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
from obsea_pipeline.gaps.analysis import simulate_contiguous_gaps
from obsea_pipeline.models.interpolation import interpolate_linear, interpolate_time, interpolate_spline, interpolate_polynomial
from obsea_pipeline.models.xgboost_wrapper import interpolate_xgboost, interpolate_xgboost_pro
from obsea_pipeline.models.varma_wrapper import interpolate_varma

# Deep Learning Wrappers (Graceful fail if Torch/CUDA is broken)
logger = logging.getLogger(__name__)

try:
    from obsea_pipeline.models.bilstm_wrapper import interpolate_bilstm
    from obsea_pipeline.models.deep_learning import (
        interpolate_missforest, interpolate_saits, interpolate_imputeformer, interpolate_brits, interpolate_brits_pro
    )
    DL_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"Deep Learning benchmarking components unavailable: {e}")
    DL_AVAILABLE = False

from obsea_pipeline.utils.visualization import (
    plot_gap_example_per_model, plot_benchmark_results,
    plot_multi_model_comparison, plot_residual_distributions
)

def benchmark_gap_filling(df: pd.DataFrame, test_variable: str = 'TEMP', gap_categories: list = None) -> pd.DataFrame:
    """
    Benchmark de interpolación con gaps CONTIGUOS por categoría.
    
    Mejoras vs versión anterior:
    1. Usa simulate_contiguous_gaps() para crear bloques únicos contiguos
    2. Evalúa SOLO los puntos dentro del gap (no puntos parcialmente rellenados)
    3. Añade validación de constraints físicos
    4. Métricas mejoradas: Coverage%, Physical_Violations%, NRMSE
    """
    logger.info(f"═══════════════════════════════════════════════════════")
    logger.info(f"  BENCHMARK v2 — Contiguous Block Evaluation on {test_variable}")
    logger.info(f"═══════════════════════════════════════════════════════")
    
    # Setup Output Directories
    output_dir = Path(CONFIG.get('output_dir', 'output_lup')) / 'benchmarks'
    gap_examples_dir = output_dir / 'gap_examples'
    gap_examples_dir.mkdir(parents=True, exist_ok=True)
    
    if gap_categories is None:
        gap_categories = ['micro', 'short', 'medium', 'long', 'extended', 'gigant']
        
    if test_variable not in df.columns:
        logger.error(f"Variable {test_variable} not found")
        return pd.DataFrame()

    # Gap category parameters: n_gaps × contiguous blocks of [min_pts, max_pts]
    cat_params = {
        'micro':    {'n_gaps': 10, 'min_pts': 3,    'max_pts': 12},    # 1.5h - 6h
        'short':    {'n_gaps': 5,  'min_pts': 24,   'max_pts': 48},    # 12h - 24h
        'medium':   {'n_gaps': 3,  'min_pts': 72,   'max_pts': 144},   # 1.5d - 3d
        'long':     {'n_gaps': 2,  'min_pts': 192,  'max_pts': 336},   # 4d - 7d
        'extended': {'n_gaps': 1,  'min_pts': 480,  'max_pts': 1440},  # 10d - 30d
        'gigant':   {'n_gaps': 1,  'min_pts': 1441, 'max_pts': 4000}   # 30d+
    }
    
    # Physical constraints for the test variable
    obs_min = df[test_variable].min()
    obs_max = df[test_variable].max()
    obs_std = df[test_variable].std()
    phys_lo = obs_min - 2 * obs_std
    phys_hi = obs_max + 2 * obs_std
    logger.info(f"  Physical range: [{obs_min:.2f}, {obs_max:.2f}] | Tolerance: [{phys_lo:.2f}, {phys_hi:.2f}]")
    
    results = []
    numeric_df = df.select_dtypes(include=[np.number])
    candidates = [c for c in numeric_df.columns if c != test_variable and '_QC' not in c and '_STD' not in c]
    corrs = numeric_df[candidates].corrwith(numeric_df[test_variable]).abs().sort_values(ascending=False)
    predictor_vars = corrs.head(5).index.tolist()
    logger.info(f"  Top predictors: {predictor_vars}")
    
    methods = [m for m, enabled in BENCHMARK_MODELS.items() if enabled]
    gpu_methods = ['bilstm', 'xgboost', 'xgboost_pro', 'saits', 'imputeformer', 'brits', 'brits_pro']
    cpu_methods = [m for m in methods if m not in gpu_methods]
    
    def evaluate_result(interpolated, gap_mask, true_values, cat, met, gap_blocks=None):
        """Evaluación rigurosa: solo puntos dentro del gap, con constraints físicos."""
        predicted = interpolated.loc[gap_mask]
        
        if isinstance(predicted, pd.DataFrame):
            predicted = predicted.iloc[:, 0]
        if isinstance(true_values, pd.DataFrame):
            true_values = true_values.iloc[:, 0]
            
        valid_mask = predicted.notna() & true_values.notna()
        n_total_gap = int(gap_mask.sum())
        n_valid = int(np.sum(valid_mask.values))
        
        if n_valid == 0:
            return {
                'Category': cat, 'Method': met, 'RMSE': np.nan, 'MAE': np.nan,
                'R2': np.nan, 'Precision_%': 0.0, 'Coverage_%': 0.0,
                'Wasserstein': np.nan, 'Physical_Violations_%': 100.0,
                'NRMSE': np.nan, 'N_Gap_Points': n_total_gap, 'N_Filled': 0
            }
        
        predicted_valid = predicted[valid_mask].values.astype(float)
        true_valid = true_values[valid_mask].values.astype(float)
        
        # Standard metrics
        rmse = np.sqrt(np.mean((predicted_valid - true_valid)**2))
        mae = np.mean(np.abs(predicted_valid - true_valid))
        data_range = obs_max - obs_min
        nrmse = rmse / data_range if data_range > 0 else np.nan
        
        tolerance = 0.05 * data_range
        precision = (np.abs(predicted_valid - true_valid) < tolerance).mean() * 100
        
        ss_res = np.sum((true_valid - predicted_valid)**2)
        ss_tot = np.sum((true_valid - true_valid.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        # Coverage: % of gap points actually filled
        coverage = (n_valid / n_total_gap) * 100 if n_total_gap > 0 else 0.0
        
        # Physical constraint violations
        violations = np.sum((predicted_valid < phys_lo) | (predicted_valid > phys_hi))
        phys_viol_pct = (violations / n_valid) * 100 if n_valid > 0 else 0.0
        
        # Plot per-model gap example
        try:
            plot_path = gap_examples_dir / f"gap_{cat}_{met}.png"
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
            w_dist = wasserstein_distance(true_valid, predicted_valid)
        except Exception:
            w_dist = np.nan

        return {
            'Category': cat, 'Method': met, 
            'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'NRMSE': round(nrmse, 4),
            'R2': round(r2, 4), 'Precision_%': round(precision, 2),
            'Coverage_%': round(coverage, 2),
            'Wasserstein': round(w_dist, 4) if not np.isnan(w_dist) else None,
            'Physical_Violations_%': round(phys_viol_pct, 2),
            'N_Gap_Points': n_total_gap, 'N_Filled': n_valid
        }
    
    def run_model(method, df_test, test_var, pred_vars, max_gap_size=None):
        try:
            t0 = time.time()
            logger.info(f"    [{method.upper()}] Starting interpolation...")
            if method == 'linear': interp = interpolate_linear(df_test[test_var], max_gap=max_gap_size)
            elif method == 'time': interp = interpolate_time(df_test[test_var], max_gap=max_gap_size)
            elif method == 'splines': interp = interpolate_spline(df_test[test_var], max_gap=max_gap_size)
            elif method == 'polynomial': interp = interpolate_polynomial(df_test[test_var], max_gap=max_gap_size)
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
            elapsed = time.time() - t0
            logger.info(f"    [{method.upper()}] Done in {elapsed:.1f}s")
            return method, interp
        except Exception as e:
            logger.error(f"    [ERROR|{method.upper()}] FAILED: {e}")
            return method, None

    for category in gap_categories:
        if category not in cat_params: continue
        params = cat_params[category]
        
        logger.info(f"")
        logger.info(f"  ┌──────────────────────────────────────────────────")
        logger.info(f"  │ [{category.upper()}] Creating {params['n_gaps']} contiguous gap(s) [{params['min_pts']}-{params['max_pts']} pts]")
        logger.info(f"  └──────────────────────────────────────────────────")
        
        # Create contiguous gaps using the new function
        df_simulated, gap_mask, gap_blocks = simulate_contiguous_gaps(
            df, test_variable, 
            n_gaps=params['n_gaps'],
            min_pts=params['min_pts'],
            max_pts=params['max_pts'],
            context_margin=96  # 48h of context on each side
        )
        
        if not gap_blocks:
            logger.warning(f"  [{category.upper()}] No gaps could be placed, skipping")
            continue
        
        # Log the placed gaps
        for i, block in enumerate(gap_blocks):
            logger.info(f"    Gap {i+1}: {block['start_time']} → {block['end_time']} ({block['length']} pts)")
        
        df_test = df.copy()
        df_test[test_variable] = df_simulated[test_variable]
        true_values = df[test_variable].loc[gap_mask]
        
        n_masked = gap_mask.sum()
        logger.info(f"  Total masked points: {n_masked}")
        
        category_interps = {}
        
        # GPU methods (sequential to avoid VRAM conflicts)
        for method in gpu_methods:
            if method not in methods: continue
            m_name, interp_absolute = run_model(method, df_test.copy(), test_variable, predictor_vars, max_gap_size=params['max_pts'])
            if interp_absolute is not None:
                category_interps[m_name] = interp_absolute
                
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except: pass
            
        # CPU methods (parallel)
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
            res = evaluate_result(interpolated, gap_mask, true_values, category, method, gap_blocks)
            if res:
                results.append(res)
                # Log key metrics immediately
                logger.info(f"    {method}: RMSE={res['RMSE']} | R²={res['R2']} | "
                           f"Coverage={res['Coverage_%']}% | Phys.Violations={res['Physical_Violations_%']}%")
        
        # Generate multi-model comparison and residual plots per category
        if len(category_interps) >= 2:
            try:
                plot_multi_model_comparison(
                    df[test_variable], category_interps, gap_mask,
                    variable=test_variable, category=category,
                    output_path=str(gap_examples_dir / f'comparison_{category}_all_models.png')
                )
                plot_residual_distributions(
                    df[test_variable], category_interps, gap_mask,
                    variable=test_variable, category=category,
                    output_path=str(gap_examples_dir / f'residuals_{category}_boxplot.png')
                )
            except Exception as e:
                logger.warning(f"  Multi-model plot generation failed for {category}: {e}")

    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
         # Export Benchmark Results as CSV
         results_df.to_csv(output_dir / 'interpolation_comparison.csv', index=False)
         logger.info(f"  ✓ Saved benchmark CSV: {output_dir / 'interpolation_comparison.csv'}")
         
         # Generate Benchmark Comparison Plot
         plot_benchmark_results(
             results_df, 
             test_variable, 
             output_path=str(output_dir / 'benchmark_results.png')
         )
         
         # Scientific recommendations
         try:
             logger.info(f"")
             logger.info(f"  ╔═══════════════════════════════════════════════════╗")
             logger.info(f"  ║  SCIENTIFIC RECOMMENDATION (Best per Category)   ║")
             logger.info(f"  ╚═══════════════════════════════════════════════════╝")
             
             for cat in results_df['Category'].unique():
                 cat_data = results_df[results_df['Category'] == cat]
                 # Filter only methods with >50% coverage (fair comparison)
                 fair = cat_data[cat_data['Coverage_%'] >= 50]
                 if fair.empty:
                     logger.info(f"  {cat}: NO METHOD achieved >50% coverage")
                     continue
                     
                 best = fair.loc[fair['RMSE'].idxmin()]
                 logger.info(f"  {cat:>10s} → {best['Method']:>15s} | RMSE={best['RMSE']:.4f} | "
                            f"R²={best['R2']:.4f} | Coverage={best['Coverage_%']:.0f}%")
         except Exception:
             pass
              
    return results_df
