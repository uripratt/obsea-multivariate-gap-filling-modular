import os
import numpy as np

# ==============================================================================
# CONFIGURACIÓN GENERAL Y DE DIRECTORIOS
# ==============================================================================
# Base paths
BASE_DIR = 'output_lup'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset paths
CTD_FILE = "OBSEA_CTD_L2_2012_2024.csv"
BUOY_FILE = "OBSEA_Buoy_Airmar_2022_2024.csv"
METEO_FILE = "OBSEA_CTVG_Vantage_Pro2_30min_nodup.csv"  # Modificado el 13 Nov
AWAC_CURRENTS_FILE = "OBSEA_AWAC_Currents_L2_2012_2024.csv"

# Output paths
UNIFIED_DATA_FILE = os.path.join(DATA_DIR, "OBSEA_multivariate_30min.csv")
GAP_STATS_FILE = os.path.join(DATA_DIR, "OBSEA_gap_statistics.csv")

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================
ENABLE_MODELS = {
    'linear': True,
    'time': True, 
    'spline_linear': True,
    'spline_quadratic': True,
    'spline_cubic': True,
    'polynomial_2': True,
    'polynomial_3': True,
    'varma': True,
    'missforest': False, 
    'xgboost': True,
    'bilstm': True, 
    'imputeformer': False, 
    'saits': False, 
    'brits': False, 
}

# The single best performing model for imputation across the board
SINGLE_BEST_MODEL = 'xgboost_pro' 

# Control parameters for Selective Interpolation mechanism
# Enable smart selection algorithm based on data constraints
SMART_SELECTION = True

# Used by the selection function to run the benchmark programmtically and then use the results
BENCHMARK_MODELS = {
    'linear': True,
    'xgboost': True,
    'xgboost_pro': False, 
    'bilstm': False, 
    'dr_lstm': False # Experimental DR-LSTM
}

# Configuración del Benchmark y Prevención de Data Leakage
BENCHMARK_CONFIG = {
    'temporal_split': True,       # Walk-Forward Split para aislar training/test
    'train_ratio': 0.7,           # 70% train (pasado), 30% test (futuro)
    'n_folds': 3,                 # Expanding Window Cross-Validation (futuro)
    'seed': 42,
}

# Control de verbosidad (Solución al "Logging Ruidoso")
LOGGING_CONFIG = {
    'pipeline_level': 'INFO',
    'dl_framework_level': 'WARNING',  # Silenciar PyTorch/PyPOTS durante training
    'benchmark_level': 'INFO',
}

# Fallback basic models for very short sequences (used by 'bilstm' and 'xgboost_pro')
# XGBoost Pro uses PCHIP, BiLSTM uses 'time'
FALLBACK_INTERPOLATION = 'time'

# Deep Learning / ML Configuration
HARDWARE_CONFIG = {
    'dl_rnn_batch_size': 32,      # BiLSTM
    'dl_transformer_batch_size': 16, # SAITS, ImputeFormer 
    'dl_epochs': 50,              # Epochs limit
    'dl_patience': 10,            # Early stopping patience
    'xgb_n_estimators': 300,      # XGBoost
    'joblib_n_jobs': 4,           # For MissForest and similar
    'use_parallel_processing': True # Allows concurrent model execution
}

# General Interpolation configuration
INTERPOLATION_CONFIG = {
    'sequence_length': 200,    # Default window for DL
    'min_samples': 24,         # Minimum valid samples needed to train properly (approx 12h)
}

# Quality Control thresholds
QC_CONFIG = {
    'max_fill_gap': 20,       # PCHIP/Spline MAX points to fill
    'bilstm_max_gap_size': 4000,
    'max_consecutive_nans': 100000 
}

HIGH_QUALITY_VARIABLES = [
    # CTD SBE16
    'TEMP', 'PSAL', 'PRES', 'CNDC', 'SVEL',
    # AWAC Corrientes - Superficie (2m)
    'AWAC2M_CSPD', 'AWAC2M_CDIR', 'AWAC2M_UCUR', 'AWAC2M_VCUR', 'AWAC2M_ZCUR',
    # AWAC Corrientes - Fondo (18m)
    'AWAC18M_CSPD', 'AWAC18M_CDIR', 'AWAC18M_UCUR', 'AWAC18M_VCUR', 'AWAC18M_ZCUR',
    # Meteo Boya (Airmar, justo arriba de OBSEA)
    'BUOY_WSPD', 'BUOY_WDIR', 'BUOY_AIRT', 'BUOY_CAPH',
    # Meteo Terrestre (CTVG, 4km tierra adentro)
    'CTVG_WSPD', 'CTVG_WDIR', 'CTVG_AIRT', 'CTVG_CAPH', 'CTVG_RELH',
]

# ==============================================================================
# DEFINICIONES DE VARIABLES FÍSICAS Y LÍMITES (QARTOD)
# ==============================================================================
# Format: Variable: (Min, Max, Unit, Description, Spike_Threshold, Gradient_Threshold, Flatline_Points)
CONFIG = {
    # CTD
    'TEMP': (10, 30, 'ºC', 'Seawater Temperature', 0.5, 0.2, 10),
    'PSAL': (30, 40, 'psu', 'Practical Salinity', 1.0, 0.5, 10),
    'PRES': (10, 30, 'dbar', 'Seawater Pressure', 2.0, 1.0, 48),
    'CNDC': (3, 6, 'S/m', 'Electrical Conductivity', 0.5, 0.2, 10),
    'SVEL': (1400, 1600, 'm/s', 'Sound Velocity in Seawater', 10.0, 5.0, 10),

    # AWAC CURRENTS - Los mismos límites se aplican a ambos bins (2m y 18m)
    'AWAC2M_CSPD': (0, 2, 'm/s', 'Sea Water Speed (2m)', 0.5, 0.2, 10),
    'AWAC2M_CDIR': (0, 360, 'deg', 'Sea Water Direction (2m)', 180.0, 90.0, 48),
    'AWAC2M_UCUR': (-2, 2, 'm/s', 'Eastward Current (2m)', 0.5, 0.2, 10),
    'AWAC2M_VCUR': (-2, 2, 'm/s', 'Northward Current (2m)', 0.5, 0.2, 10),
    'AWAC2M_ZCUR': (-1, 1, 'm/s', 'Vertical Current (2m)', 0.3, 0.1, 10),
    'AWAC18M_CSPD': (0, 2, 'm/s', 'Sea Water Speed (18m)', 0.5, 0.2, 10),
    'AWAC18M_CDIR': (0, 360, 'deg', 'Sea Water Direction (18m)', 180.0, 90.0, 48),
    'AWAC18M_UCUR': (-2, 2, 'm/s', 'Eastward Current (18m)', 0.5, 0.2, 10),
    'AWAC18M_VCUR': (-2, 2, 'm/s', 'Northward Current (18m)', 0.5, 0.2, 10),
    'AWAC18M_ZCUR': (-1, 1, 'm/s', 'Vertical Current (18m)', 0.3, 0.1, 10),

    # METEO - Boya Besos (Airmar 200WX, justo encima de OBSEA)
    'BUOY_WSPD': (0, 40, 'm/s', 'Wind Speed (Buoy)', 5.0, 3.0, 24),
    'BUOY_WDIR': (0, 360, 'deg', 'Wind Direction (Buoy)', 180.0, 90.0, 48),
    'BUOY_AIRT': (-5, 45, 'ºC', 'Air Temperature (Buoy)', 2.0, 1.0, 10),
    'BUOY_CAPH': (950, 1050, 'hPa', 'Atmospheric Pressure (Buoy)', 10.0, 5.0, 24),

    # METEO - CTVG (Vantage Pro2, estación terrestre a 4km)
    'CTVG_WSPD': (0, 40, 'm/s', 'Wind Speed (CTVG)', 5.0, 3.0, 24),
    'CTVG_WDIR': (0, 360, 'deg', 'Wind Direction (CTVG)', 180.0, 90.0, 48),
    'CTVG_AIRT': (-5, 45, 'ºC', 'Air Temperature (CTVG)', 2.0, 1.0, 10),
    'CTVG_CAPH': (950, 1050, 'hPa', 'Atmospheric Pressure (CTVG)', 10.0, 5.0, 24),
    'CTVG_RELH': (0, 100, '%', 'Relative Humidity (CTVG)', 15.0, 10.0, 24),

    # DERIVADAS (No aplican checks estrictos QARTOD general)
    'SIGMA_T': (20, 30, 'kg/m^3', 'Density anomaly', 2.0, 1.0, 10),
}

CONFIG['data_paths'] = {
    'CTD': '/home/uripratt/Documents/PhD/OBSEA_data/CTD/OBSEA_CTD_multivar_QC_minimal_small_interpolated.csv'
}

CONFIG['variables'] = {
    'CTD':        ['TEMP', 'PSAL', 'PRES', 'CNDC', 'SVEL'],
    'AWAC_2M':    ['AWAC2M_CSPD', 'AWAC2M_CDIR', 'AWAC2M_UCUR', 'AWAC2M_VCUR', 'AWAC2M_ZCUR'],
    'AWAC_18M':   ['AWAC18M_CSPD', 'AWAC18M_CDIR', 'AWAC18M_UCUR', 'AWAC18M_VCUR', 'AWAC18M_ZCUR'],
    'BUOY_METEO': ['BUOY_WSPD', 'BUOY_WDIR', 'BUOY_AIRT', 'BUOY_CAPH'],
    'CTVG_METEO': ['CTVG_WSPD', 'CTVG_WDIR', 'CTVG_AIRT', 'CTVG_CAPH', 'CTVG_RELH'],
}

CONFIG['output_dir'] = DATA_DIR


# ==============================================================================
# GAP CATEGORIES DEFINITION (The "Scale-Aware" Implementation)
# ==============================================================================
class GapInfo:
    def __init__(self, start_idx, end_idx, length, category):
        """
        Clase base para guardar la info de un gap identificado.
        """
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.length = length
        self.category = category

    def __repr__(self):
        return f"Gap({self.length} pts, {self.category})"

GAP_CATEGORIES = {
    'micro': {'max_hours': 6, 'description': 'Up to 6 hours (<= 12 points)'},
    'short': {'max_hours': 24, 'description': '6 to 24 hours (13 - 48 points)'},
    'medium': {'max_hours': 72, 'description': '1 to 3 days (49 - 144 points)'},
    'long': {'max_hours': 168, 'description': '3 to 7 days (145 - 336 points)'},
    'extended': {'max_hours': 240, 'description': '7 to 10 days (337 - 480 points)'},
    'gigant': {'max_hours': float('inf'), 'description': 'More than 10 days (> 480 points)'}
}
