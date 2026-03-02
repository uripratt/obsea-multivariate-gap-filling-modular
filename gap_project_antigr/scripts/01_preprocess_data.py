#!/usr/bin/env python
"""
Script 1: Preprocess OBSEA data
- Load raw CSV
- Apply QC filtering
- Remove outliers
- Normalize data
- Split into train/val/test
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger
from src.data import (
    load_obsea_data,
    print_data_summary,
    filter_by_qc,
    DataPreprocessor,
    temporal_train_val_test_split,
)
import pandas as pd
import logging

# Set up logging
logger = setup_logger(
    name="preprocess",
    log_file="results/logs/01_preprocess.log",
    level=logging.INFO
)

def main():
    """Main preprocessing routine."""
    logger.info("="*80)
    logger.info("OBSEA DATA PREPROCESSING")
    logger.info("="*80)
    
    # Load configuration
    config = load_config("configs/data_config.yaml")
    
    data_cfg = config['data']
    vars_cfg = config['variables']
    preproc_cfg = config['preprocessing']
    split_cfg = config['temporal_split']
    
    # Load raw data
    logger.info("\n1. Loading raw data...")
    raw_file = Path(data_cfg['raw_dir']) / data_cfg['raw_file']
    df = load_obsea_data(str(raw_file))
    
    print_data_summary(df, vars_cfg['all_vars'])
    
    # Apply QC filtering
    logger.info("\n2. Applying QC filters...")
    for var in vars_cfg['all_vars']:
        qc_col = f"{var}_QC"
        if qc_col in vars_cfg['qc_columns'] and qc_col in df.columns:
            df = filter_by_qc(
                df,
                variable=var,
                qc_column=qc_col,
                good_flags=preproc_cfg['qc_good_flags'],
                bad_flags=preproc_cfg['qc_bad_flags'],
            )
    
    # Initialize preprocessor
    logger.info("\n3. Preprocessing (outlier removal & normalization)...")
    preprocessor = DataPreprocessor(
        outlier_method=preproc_cfg['outlier_method'],
        outlier_threshold=preproc_cfg['outlier_threshold'],
        normalization=preproc_cfg['normalization'],
    )
    
    # Fit and transform
    df_processed = preprocessor.fit_transform(
        df,
        variables=vars_cfg['all_vars'],
        remove_outliers=True,
    )
    
    # Split into train/val/test
    logger.info("\n4. Splitting into train/validation/test sets...")
    train_df, val_df, test_df = temporal_train_val_test_split(
        df_processed,
        train_ratio=split_cfg['train_ratio'],
        val_ratio=split_cfg['val_ratio'],
        test_ratio=split_cfg['test_ratio'],
    )
    
    # Save processed data
    logger.info("\n5. Saving processed data...")
    processed_dir = Path(data_cfg['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    df_processed.to_csv(processed_dir / data_cfg['preprocessed_file'])
    train_df.to_csv(processed_dir / data_cfg['train_file'])
    val_df.to_csv(processed_dir / data_cfg['val_file'])
    test_df.to_csv(processed_dir / data_cfg['test_file'])
    
    logger.info(f"Saved preprocessed data to {processed_dir}")
    
    # Save preprocessor
    import joblib
    preprocessor_path = processed_dir / 'preprocessor.pkl'
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("="*80)
    
    print("\n✓ Preprocessing complete!")
    print(f"  - Preprocessed file: {processed_dir / data_cfg['preprocessed_file']}")
    print(f"  - Train: {len(train_df)} records")
    print(f"  - Val:   {len(val_df)} records")
    print(f"  - Test:  {len(test_df)} records")

if __name__ == "__main__":
    main()
