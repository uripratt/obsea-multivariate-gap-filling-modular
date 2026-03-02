#!/usr/bin/env python
"""
Script 6: Train LSTM model
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger
from src.data import load_obsea_data
from src.models import LSTMImputer
from src.evaluation import calculate_metrics, save_metrics
import logging

logger = setup_logger(
    name="lstm_train",
    log_file="results/logs/06_lstm_train.log",
    level=logging.INFO
)


def main():
    """Train LSTM model."""
    logger.info("="*80)
    logger.info("LSTM TRAINING")
    logger.info("="*80)
    
    # Load configurations
    data_config = load_config("configs/data_config.yaml")
    lstm_config = load_config("configs/lstm_config.yaml")
    gap_config = load_config("configs/gap_simulation.yaml")
    
    # Settings
    target_var = data_config['variables']['target']
    gap_dir = Path(gap_config['simulation']['output_dir'])
    
    # Load data
    logger.info("\nLoading data...")
    train_df = load_obsea_data(f"data/processed/{data_config['data']['train_file']}")
    val_df = load_obsea_data(f"data/processed/{data_config['data']['val_file']}")
    
    # Initialize model
    logger.info("\nInitializing LSTM model...")
    model = LSTMImputer(
        hidden_size=lstm_config['model']['hidden_size'],
        num_layers=lstm_config['model']['num_layers'],
        dropout=lstm_config['model']['dropout'],
        bidirectional=lstm_config['model']['bidirectional'],
        sequence_length=lstm_config['model']['sequence_length'],
        batch_size=lstm_config['training']['batch_size'],
        epochs=lstm_config['training']['epochs'],
        learning_rate=lstm_config['training']['learning_rate'],
        device=lstm_config['hardware']['device'],
    )
    
    # Train
    logger.info("\nTraining model...")
    model.fit(
        train_df,
        target_var=target_var,
        validation_data=val_df,
    )
    
    # Save
    model_path = Path(lstm_config['output']['model_dir']) / "lstm_model.pth"
    model.save(str(model_path))
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    df_gapped = load_obsea_data(str(gap_dir / "test_random_medium_gapped.csv"))
    ground_truth = load_obsea_data(str(gap_dir / "test_random_medium_truth.csv"))
    
    imputed = model.predict(df_gapped, target_var)
    
    gap_mask = ground_truth[target_var].notna()
    metrics = calculate_metrics(
        ground_truth.loc[gap_mask, target_var].values,
        imputed.loc[gap_mask].values,
    )
    
    logger.info("\nTest Metrics:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    
    # Save metrics
    metrics_dir = Path(lstm_config['output']['model_dir'])
    save_metrics(metrics, str(metrics_dir / "test_metrics.json"), "LSTM")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    
    print(f"\n✓ LSTM training complete!")
    print(f"  - Model: {model_path}")
    print(f"  - Test RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
