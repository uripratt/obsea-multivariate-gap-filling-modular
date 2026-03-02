#!/usr/bin/env python
"""
Comprehensive Baseline Evaluation Script
- Evaluates Linear, PCHIP, and Spline methods
- Tests on all gap patterns
- Generates comparison figures
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger
from src.data import load_obsea_data
from src.models import BaselineImputer
from src.evaluation import (
    calculate_metrics,
    calculate_error_by_gap_length,
    print_metrics,
    save_metrics,
)
from src.visualization import (
    plot_error_by_gap_length,
    plot_model_comparison_bars,
    plot_gap_pattern_comparison,
)
import pandas as pd
import numpy as np
import logging
import json
from collections import defaultdict

logger = setup_logger(
    name="baseline_eval",
    log_file="results/logs/04_baseline_evaluation.log",
    level=logging.INFO
)


def evaluate_model_on_pattern(
    model,
    gap_pattern: str,
    dataset: str,
    target_var: str,
    gap_dir: Path,
):
    """Evaluate a single model on a gap pattern."""
    # Load gapped data and ground truth
    df_gapped = load_obsea_data(str(gap_dir / f"{dataset}_{gap_pattern}_gapped.csv"))
    ground_truth = load_obsea_data(str(gap_dir / f"{dataset}_{gap_pattern}_truth.csv"))
    
    # Impute
    imputed = model.impute(df_gapped[target_var])
    
    # Evaluate on gaps only
    gap_mask = ground_truth[target_var].notna()
    
    if gap_mask.sum() == 0:
        logger.warning(f"No gaps found in {gap_pattern}")
        return None, None, None
    
    y_true = ground_truth.loc[gap_mask, target_var].values
    y_pred = imputed.loc[gap_mask].values
    
    # Calculate overall metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Calculate error by gap length
    error_by_length = calculate_error_by_gap_length(
        ground_truth[target_var].values,
        imputed.values,
        gap_mask.values,
        length_bins=[1, 6, 24, 48, 168, 336, np.inf]
    )
    
    return metrics, error_by_length, imputed


def main():
    """Main evaluation routine."""
    logger.info("="*80)
    logger.info("COMPREHENSIVE BASELINE EVALUATION")
    logger.info("="*80)
    
    # Load configurations
    data_config = load_config("configs/data_config.yaml")
    gap_config = load_config("configs/gap_simulation.yaml")
    
    # Settings
    methods = ['linear', 'pchip', 'spline']
    target_var = data_config['variables']['target']
    gap_dir = Path(gap_config['simulation']['output_dir'])
    dataset = 'test'  # Evaluate on test set
    
    # Patterns to evaluate (only single-variable patterns for now)
    patterns_to_eval = [p['name'] for p in gap_config['gap_patterns'] 
                        if target_var in p.get('target_vars', [])]
    
    logger.info(f"\nEvaluating {len(methods)} methods on {len(patterns_to_eval)} patterns")
    logger.info(f"Methods: {methods}")
    logger.info(f"Patterns: {patterns_to_eval}")
    
    # Storage for results
    all_results = defaultdict(dict)  # {pattern: {method: metrics}}
    all_errors_by_length = defaultdict(dict)  # {pattern: {method: error_df}}
    
    # Evaluate each method on each pattern
    for pattern in patterns_to_eval:
        logger.info(f"\n{'='*60}")
        logger.info(f"Pattern: {pattern}")
        logger.info(f"{'='*60}")
        
        for method in methods:
            logger.info(f"\n  Method: {method.upper()}")
            
            # Create imputer
            imputer = BaselineImputer(method=method)
            
            # Evaluate
            metrics, error_by_length, imputed = evaluate_model_on_pattern(
                imputer, pattern, dataset, target_var, gap_dir
            )
            
            if metrics is None:
                continue
            
            # Store results
            all_results[pattern][method] = metrics
            all_errors_by_length[pattern][method] = error_by_length
            
            # Print metrics
            logger.info(f"    RMSE: {metrics['rmse']:.4f}")
            logger.info(f"    MAE:  {metrics['mae']:.4f}")
            logger.info(f"    Bias: {metrics['bias']:.4f}")
            logger.info(f"    R²:   {metrics['r2']:.4f}")
            
            # Save detailed metrics
            metrics_dir = Path(f"results/metrics/baseline/{pattern}")
            metrics_dir.mkdir(parents=True, exist_ok=True)
            save_metrics(metrics, str(metrics_dir / f"{method}.json"), f"{method}_{pattern}")
    
    # ========== VISUALIZATIONS ==========
    logger.info("\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    figures_dir = Path("results/figures/baseline_comparison")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Error by gap length for each pattern
    logger.info("\n1. Creating error vs gap length plots...")
    for pattern in patterns_to_eval:
        if pattern not in all_errors_by_length:
            continue
        
        plot_error_by_gap_length(
            all_errors_by_length[pattern],
            metric='rmse',
            output_path=str(figures_dir / f"{pattern}_rmse_by_length.png"),
            title=f"RMSE vs Gap Length - {pattern.replace('_', ' ').title()}"
        )
        
        plot_error_by_gap_length(
            all_errors_by_length[pattern],
            metric='mae',
            output_path=str(figures_dir / f"{pattern}_mae_by_length.png"),
            title=f"MAE vs Gap Length - {pattern.replace('_', ' ').title()}"
        )
    
    # 2. Model comparison bars for each pattern
    logger.info("\n2. Creating model comparison bar charts...")
    for pattern in patterns_to_eval:
        if pattern not in all_results:
            continue
        
        plot_model_comparison_bars(
            all_results[pattern],
            metrics=['rmse', 'mae', 'bias'],
            output_path=str(figures_dir / f"{pattern}_comparison.png"),
            title=f"Model Comparison - {pattern.replace('_', ' ').title()}"
        )
    
    # 3. Gap pattern comparison (all patterns together)
    logger.info("\n3. Creating gap pattern comparison...")
    plot_gap_pattern_comparison(
        all_results,
        metric='rmse',
        output_path=str(figures_dir / "all_patterns_rmse.png")
    )
    
    plot_gap_pattern_comparison(
        all_results,
        metric='mae',
        output_path=str(figures_dir / "all_patterns_mae.png")
    )
    
    # ========== SUMMARY TABLES ==========
    logger.info("\n" + "="*80)
    logger.info("SUMMARY TABLES")
    logger.info("="*80)
    
    # Create summary table
    summary_data = []
    for pattern in patterns_to_eval:
        for method in methods:
            if pattern in all_results and method in all_results[pattern]:
                metrics = all_results[pattern][method]
                summary_data.append({
                    'Pattern': pattern,
                    'Method': method,
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'Bias': metrics['bias'],
                    'R²': metrics['r2'],
                    'N_samples': metrics['n_samples']
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    summary_path = Path("results/metrics/baseline_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSaved summary table to {summary_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE METHODS SUMMARY")
    print("="*80)
    print("\nOverall Results:")
    print(summary_df.to_string(index=False))
    
    # Find best method for each pattern
    print("\n" + "-"*80)
    print("BEST METHOD PER PATTERN (by RMSE):")
    print("-"*80)
    for pattern in patterns_to_eval:
        pattern_results = summary_df[summary_df['Pattern'] == pattern]
        if len(pattern_results) > 0:
            best = pattern_results.loc[pattern_results['RMSE'].idxmin()]
            print(f"\n{pattern.replace('_', ' ').title()}:")
            print(f"  Best: {best['Method'].upper()}")
            print(f"  RMSE: {best['RMSE']:.4f}")
            print(f"  MAE:  {best['MAE']:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*80)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  - Summary: {summary_path}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Total patterns evaluated: {len(patterns_to_eval)}")
    print(f"  - Total methods compared: {len(methods)}")


if __name__ == "__main__":
    main()
