#!/usr/bin/env python
"""
Script 3: Train and evaluate baseline interpolation methods
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger
from src.data import load_obsea_data
from src.models import BaselineImputer
from src.evaluation import calculate_metrics, print_metrics, save_metrics
import pandas as pd
import numpy as np
import logging

logger = setup_logger(
    name="baseline",
    log_file="results/logs/03_baseline.log",
    level=logging.INFO
)

def main():
    """Train and evaluate baseline models."""
    logger.info("="*80)
    logger.info("BASELINE INTERPOLATION METHODS")
    logger.info("="*80)
    
    # Load configuration
    data_config = load_config("configs/data_config.yaml")
    gap_config = load_config("configs/gap_simulation.yaml")
    
    # Baseline methods to test
    methods = ['linear', 'pchip', 'spline']
    
    # Gap patterns to evaluate
    gap_pattern = 'random_medium'  # Can be changed
    
    # Results storage
    all_results = {}
    
    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Method: {method.upper()}")
        logger.info(f"{'='*60}")
        
        # Create imputer
        imputer = BaselineImputer(method=method)
        
        # Load test data with gaps
        gap_dir = Path(gap_config['simulation']['output_dir'])
        df_gapped = load_obsea_data(str(gap_dir / f"test_{gap_pattern}_gapped.csv"))
        ground_truth = load_obsea_data(str(gap_dir / f"test_{gap_pattern}_truth.csv"))
        
        # Get target variable
        target_var = data_config['variables']['target']
        
        # Impute
        logger.info(f"Imputing {target_var}...")
        imputed = imputer.impute(df_gapped[target_var])
        
        # Evaluate on gaps only
        gap_mask = ground_truth[target_var].notna()
        
        y_true = ground_truth.loc[gap_mask, target_var].values
        y_pred = imputed.loc[gap_mask].values
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Print and save
        print_metrics(metrics, model_name=f"Baseline {method}")
        
        # Save metrics
        metrics_dir = Path("results/metrics/baseline")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(
            metrics,
            str(metrics_dir / f"{method}_{gap_pattern}.json"),
            model_name=f"Baseline_{method}"
        )
        
        # Save predictions
        pred_dir = Path("results/predictions/baseline")
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        pred_df = pd.DataFrame({
            'timestamp': df_gapped.index,
            'ground_truth': ground_truth[target_var],
            'imputed': imputed,
            'is_gap': gap_mask
        })
        pred_df.to_csv(pred_dir / f"{method}_{gap_pattern}_predictions.csv", index=False)
        
        all_results[method] = metrics
    
    # Compare methods
    logger.info("\n" + "="*80)
    logger.info("BASELINE COMPARISON")
    logger.info("="*80)
    
    comparison = pd.DataFrame(all_results).T
    comparison = comparison.sort_values('rmse')
    
    print("\nBaseline Method Comparison:")
    print(comparison[['rmse', 'mae', 'bias', 'r2']].to_string())
    
    logger.info("\nBest method (by RMSE): " + comparison.index[0])
    
    print(f"\n✓ Baseline evaluation complete!")
    print(f"  Best method: {comparison.index[0]}")
    print(f"  RMSE: {comparison.iloc[0]['rmse']:.4f}")

if __name__ == "__main__":
    main()
