#!/usr/bin/env python
"""
Script 08: Generate Web App Data
Generates full reconstructed datasets for VARMA and Bi-LSTM models
and saves them to the web application's data directory.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger
from src.data import load_obsea_data
from src.models import LSTMImputer, VARMAImputer

logger = setup_logger(
    name="generate_webapp_data",
    log_file="results/logs/08_generate_webapp_data.log",
    level=logging.INFO
)

def main():
    logger.info("="*80)
    logger.info("GENERATING WEB APP DATASETS")
    logger.info("="*80)

    # 1. Load Configurations
    # Determine Project Root (parent of scripts/)
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    data_config = load_config(str(BASE_DIR / "configs/data_config.yaml"))
    lstm_config = load_config(str(BASE_DIR / "configs/lstm_config.yaml"))
    varma_config = load_config(str(BASE_DIR / "configs/varma_config.yaml"))
    
    # Paths
    # We are in gap_project_antigr. We want to go to ../scripts/webapp/data (sibling folder)
    webapp_data_dir = BASE_DIR.parent / "scripts/webapp/data"
    
    # Ensure raw data is loaded (the one with gaps)
    # If standard load fills gaps, we might need a specific 'raw' loader.
    # Assuming 'load_obsea_data' loads the CSV as-is.
    raw_path = BASE_DIR / "data/processed" / data_config['data']['preprocessed_file']
    if not raw_path.exists():
        logger.warning(f"Preprocessed file not found at {raw_path}. Trying raw.")
        raw_path = BASE_DIR / "data/raw" / data_config['data']['raw_file']
        
    logger.info(f"Loading data from: {raw_path}")
    df = load_obsea_data(str(raw_path))
    target_var = data_config['variables']['target']
    
    logger.info(f"Target Variable: {target_var}")
    logger.info(f"Data Shape: {df.shape}")

    # ==========================================
    # 2. Generate Bi-LSTM Reconstruction
    # ==========================================
    try:
        model_path = BASE_DIR / lstm_config['output']['model_dir'] / "lstm_model.pth"
        if model_path.exists():
            logger.info(f"\n[Bi-LSTM] Loading model from {model_path}...")
            lstm_model = LSTMImputer.load(str(model_path))
            
            logger.info("[Bi-LSTM] Predicting full series...")
            # Predict fills NaN values in the target column
            # Ensure df has NaNs where appropriate. 
            # If df is already filled, this won't change anything unless we force NaNs?
            # We assume df has NaNs (Gaps).
            imputed_series = lstm_model.predict(df, target_var)
            
            # Create output DataFrame
            df_bilstm = df.copy()
            df_bilstm[target_var] = imputed_series
            
            # Save
            output_file = webapp_data_dir / "OBSEA_multivariate_30min_BILSTM.csv"
            logger.info(f"[Bi-LSTM] Saving to {output_file}...")
            df_bilstm.to_csv(output_file)
            print(f"✓ Generated Bi-LSTM data: {output_file}")
            
        else:
            logger.warning(f"[Bi-LSTM] Model not found at {model_path}. Skipping.")
    
    except Exception as e:
        logger.error(f"[Bi-LSTM] Error: {e}")

    # ==========================================
    # 3. Generate VARMA Reconstruction
    # ==========================================
    try:
        # VARMA might not be saved to disk in the same way, usually retrained or saved as pickle
        # Check if VARMA config has model_dir
        # Assuming we might need to train it if not saved?
        # For this script, let's assume we train/fit on available data then predict gaps.
        # VARMA is fast enough to retrain typically.
        
        logger.info(f"\n[VARMA] Training/Refitting for full series...")
        
        # VARMA needs predictor variables
        # predictors = ['PSAL', 'TEMP_STD'] # Example, should get from config/correlation
        predictors = ['PSAL'] # Simplified for demo
        
        varma_model = VARMAImputer(
            p=varma_config.get('model', {}).get('p', 2),
            q=varma_config.get('model', {}).get('q', 0),
            batch_size=varma_config.get('training', {}).get('batch_size', 128),
            epochs=varma_config.get('training', {}).get('epochs', 50),
            learning_rate=varma_config.get('training', {}).get('learning_rate', 0.001),
            device=varma_config.get('hardware', {}).get('device', 'cpu'),
            verbose=True
        )
        
        varma_model.fit(df, target_var, predictors)
        
        logger.info("[VARMA] Predicting full series...")
        imputed_series = varma_model.predict_series(df, target_var, predictors)
        
        df_varma = df.copy()
        df_varma[target_var] = imputed_series
        
        output_file = webapp_data_dir / "OBSEA_multivariate_30min_VARMA.csv"
        logger.info(f"[VARMA] Saving to {output_file}...")
        df_varma.to_csv(output_file)
        print(f"✓ Generated VARMA data: {output_file}")

    except Exception as e:
        logger.error(f"[VARMA] Error: {e}")

    logger.info("\nDone.")

if __name__ == "__main__":
    main()
