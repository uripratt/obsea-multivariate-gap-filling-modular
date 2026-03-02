import pandas as pd
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from src.models.multivariate_lstm_model import MultivariateLSTMImputer
from src.models.xgboost_model import XGBoostImputer

def test_lstm_robustness():
    print("Testing Bi-LSTM Robustness...")
    idx = pd.date_range('2020-01-01', periods=200, freq='30min')
    data = pd.DataFrame({
        'TEMP': np.sin(np.arange(200) * 0.1) * 5 + 15,
        'PSAL': np.random.normal(38, 0.1, 200),
    }, index=idx)
    
    # Create a gap
    data.loc[idx[100:150], 'TEMP'] = np.nan
    
    imputer = MultivariateLSTMImputer(
        target_var='TEMP', 
        predictor_vars=['TEMP', 'PSAL'], 
        sequence_length=48, 
        epochs=1, 
        device='cpu'
    )
    
    # Mocking training to avoid full run
    imputer.mean = data.mean()
    imputer.std = data.std()
    imputer.std[imputer.std == 0] = 1.0
    
    # Create a simple model manually to test predict logic
    from src.models.multivariate_lstm_model import MultivariateLSTMModel
    imputer.model = MultivariateLSTMModel(
        input_size=3, # TEMP, PSAL, TEMP_is_observed
        hidden_size=16,
        num_layers=1,
        bidirectional=True
    )
    
    # Test Predict
    print("Running LSTM predict...")
    try:
        result = imputer.predict(data)
        print("✅ LSTM Predict finished without errors.")
        
        # Verify result structure
        if len(result) == len(data):
            print(f"✅ Result length matches: {len(result)}")
        else:
            print(f"❌ Result length mismatch: {len(result)} vs {len(data)}")
            
    except Exception as e:
        print(f"❌ LSTM Predict failed: {e}")
        import traceback
        traceback.print_exc()

def test_xgboost_robustness():
    print("\nTesting XGBoost Robustness...")
    idx = pd.date_range('2020-01-01', periods=100, freq='30min')
    data = pd.DataFrame({
        'TEMP': np.sin(np.arange(100) * 0.1) * 5 + 15,
        'PSAL': np.random.normal(38, 0.1, 100),
    }, index=idx)
    
    # Create a gap
    data.loc[idx[50:60], 'TEMP'] = np.nan
    
    imputer = XGBoostImputer(xgb_params={'n_estimators': 10, 'device': 'cpu'})
    
    print("Fitting XGBoost (small)...")
    try:
        imputer.fit(data, target_var='TEMP', multivariate_vars=['PSAL'])
        print("✅ XGBoost Fit finished.")
        
        print("Predicting XGBoost...")
        result = imputer.predict(data, multivariate_vars=['PSAL'])
        print("✅ XGBoost Predict finished.")
        
        # Check if is_observed exists in feature list
        if 'TEMP_is_observed' in imputer.feature_columns:
            print("✅ 'TEMP_is_observed' found in model feature columns.")
        else:
            print(f"❌ 'TEMP_is_observed' NOT found in columns: {imputer.feature_columns}")
            
    except Exception as e:
        print(f"❌ XGBoost test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lstm_robustness()
    test_xgboost_robustness()
