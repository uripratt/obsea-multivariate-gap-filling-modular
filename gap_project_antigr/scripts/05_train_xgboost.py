#!/usr/bin/env python
"""
Script 5: Train XGBoost model
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger
from src.data import load_obsea_data
from src.models import XGBoostImputer
from src.evaluation import calculate_metrics, save_metrics
import logging

logger = setup_logger(
    name="xgboost_train",
    log_file="results/logs/05_xgboost_train.log",
    level=logging.INFO
)


def main():
    """Train XGBoost model."""
    logger.info("="*80)
    logger.info("XGBOOST TRAINING")
    logger.info("="*80)
    
    # Load configurations
    data_config = load_config("configs/data_config.yaml")
    xgb_config = load_config("configs/xgboost_config.yaml")
    gap_config = load_config("configs/gap_simulation.yaml")
    
    # Settings
    target_var = data_config['variables']['target']
    multivariate_vars = xgb_config['features'].get('multivariate_vars', [])
    gap_dir = Path(gap_config['simulation']['output_dir'])
    
    # Load training data
    logger.info("\nLoading training data...")
    train_df = load_obsea_data(f"data/processed/{data_config['data']['train_file']}")
    val_df = load_obsea_data(f"data/processed/{data_config['data']['val_file']}")
    
    logger.info(f"Train set: {len(train_df)} records")
    logger.info(f"Val set: {len(val_df)} records")
    
    # Initialize model
    logger.info("\nInitializing XGBoost model...")
    model = XGBoostImputer(
        xgb_params=xgb_config['model'],
        feature_config=xgb_config['features'],
    )
    
    # Train
    logger.info("\nTraining model...")
    model.fit(
        train_df,
        target_var=target_var,
        multivariate_vars=multivariate_vars,
        validation_data=val_df,
    )
    
    # Save model
    model_path = Path(xgb_config['output']['model_dir']) / "xgboost_model.pkl"
    model.save(str(model_path))
    
    # Feature importance
    logger.info("\nFeature Importance (Top 20):")
    importance_df = model.get_feature_importance(top_n=20)
    print(importance_df.to_string(index=False))
    
    # Save feature importance
    importance_path = Path(xgb_config['output']['model_dir']) / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    
    # Evaluate on a gap pattern
    logger.info("\nEvaluating on test set (random_medium pattern)...")
    
    df_gapped = load_obsea_data(str(gap_dir / "test_random_medium_gapped.csv"))
    ground_truth = load_obsea_data(str(gap_dir / "test_random_medium_truth.csv"))
    
    # Predict
    imputed = model.predict(df_gapped, multivariate_vars=multivariate_vars)
    
    # Evaluate
    gap_mask = ground_truth[target_var].notna()
    metrics = calculate_metrics(
        ground_truth.loc[gap_mask, target_var].values,
        imputed.loc[gap_mask].values,
    )
    
    logger.info("\nTest Metrics:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")
    logger.info(f"  Bias: {metrics['bias']:.4f}")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    
    # Save metrics
    metrics_dir = Path(xgb_config['output']['model_dir'])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, str(metrics_dir / "test_metrics.json"), "XGBoost")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    
    print(f"\n✓ XGBoost training complete!")
    print(f"  - Model: {model_path}")
    print(f"  - Test RMSE: {metrics['rmse']:.4f}")
    print(f"  - Test MAE: {metrics['mae']:.4f}")


if __name__ == "__main__":
    main()
