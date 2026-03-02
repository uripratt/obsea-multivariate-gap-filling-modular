# OBSEA Gap Filling Pipeline

A scientifically rigorous, reproducible, and modular pipeline for multivariate time series imputation on long-term underwater observatory data from OBSEA (NW Mediterranean Sea).

## Overview

This project implements and compares multiple gap-filling approaches for oceanographic time series:

- **Baseline Methods**: Linear, PCHIP, cubic spline interpolation
- **Machine Learning**: XGBoost, MissForest
- **Deep Learning**: LSTM, Temporal Convolutional Networks (TCN), Chronos (Transformer)

The pipeline supports:
- ✅ Artificial gap simulation for validation
- ✅ Multivariate imputation using related sensors
- ✅ Temporal feature engineering (lags, seasonality)
- ✅ Comprehensive evaluation metrics
- ✅ Scientific visualization and reporting

## Project Structure

```
gap_project_antigr/
├── data/
│   ├── raw/                    # Original OBSEA CSV data
│   ├── processed/              # Preprocessed datasets
│   └── simulated_gaps/         # Artificial gaps for validation
├── src/
│   ├── data/                   # Data loading, preprocessing, gap simulation
│   ├── features/               # Feature engineering (temporal & multivariate)
│   ├── models/                 # All imputation models
│   ├── evaluation/             # Metrics and gap-specific analysis
│   ├── visualization/          # Plotting utilities
│   └── utils/                  # Configuration and logging
├── configs/                    # YAML configuration files
├── scripts/                    # Training and evaluation scripts
├── notebooks/                  # Jupyter notebooks for exploration
├── results/                    # Model outputs, predictions, figures
└── tests/                      # Unit tests
```

## Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU (optional, for deep learning models)

### Setup

```bash
# Clone or navigate to the project directory
cd gap_project_antigr

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

```bash
# Preprocess raw data (QC filtering, normalization)
python scripts/01_preprocess_data.py

# Create artificial gaps for validation
python scripts/02_simulate_gaps.py
```

### 2. Train Baseline Models

```bash
# Train all baseline interpolation methods
python scripts/03_train_baseline.py
```

### 3. Train ML/DL Models

```bash
# XGBoost
python scripts/04_train_xgboost.py

# LSTM
python scripts/06_train_lstm.py

# TCN
python scripts/07_train_tcn.py
```

### 4. Evaluate and Compare

```bash
# Comprehensive evaluation of all models
python scripts/09_evaluate_all.py

# Generate report with visualizations
python scripts/10_generate_report.py
```

## Configuration

All models are configured via YAML files in `configs/`:

- `data_config.yaml` - Data paths, variables, preprocessing
- `gap_simulation.yaml` - Gap patterns and simulation parameters
- `xgboost_config.yaml` - XGBoost hyperparameters
- `lstm_config.yaml` - LSTM architecture and training
- `tcn_config.yaml` - TCN architecture

Edit these files to customize experiments.

## Usage Examples

### Python API

```python
from src.data import load_obsea_data, DataPreprocessor, GapSimulator
from src.models import BaselineImputer
from src.evaluation import calculate_metrics

# Load data
df = load_obsea_data('data/raw/OBSEA_CTD_multivar_QC_minimal_small_interpolated.csv')

# Preprocess
preprocessor = DataPreprocessor(
    outlier_method='zscore',
    normalization='standard'
)
df_clean = preprocessor.fit_transform(df, variables=['TEMP', 'PSAL'])

# Simulate gaps
simulator = GapSimulator(random_seed=42)
df_gapped, ground_truth = simulator.create_random_gaps(
    df_clean,
    variables=['TEMP'],
    num_gaps=50,
    min_length='6H',
    max_length='3D'
)

# Impute with baseline
imputer = BaselineImputer(method='pchip')
df_imputed = df_gapped.copy()
df_imputed['TEMP'] = imputer.impute(df_gapped['TEMP'])

# Evaluate
gap_mask = ground_truth['TEMP'].notna()
metrics = calculate_metrics(
    ground_truth['TEMP'][gap_mask].values,
    df_imputed['TEMP'][gap_mask].values
)
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Jupyter Notebooks

Explore the data and results interactively:

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## Evaluation Metrics

The pipeline calculates:

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)  
- **Bias** (Mean error)
- **R²** score
- **Skill scores** (relative to baseline)
- **Error vs gap length** analysis
- **Error vs gap position** (start/middle/end)

## Scientific Validation Strategy

1. **Artificial Gap Simulation**: Create realistic gap patterns in observed data
2. **Temporal Split**: Train/validation/test split preserving time order
3. **Multiple Gap Patterns**: Random, clustered, multivariate
4. **Gap Length Stratification**: Error analysis by gap duration

## Output Structure

Results are saved to `results/`:

```
results/
├── models/                     # Trained model weights
│   ├── xgboost/
│   ├── lstm/
│   └── tcn/
├── predictions/                # Imputed time series
├── metrics/                    # JSON files with metrics
└── figures/                    # Plots and visualizations
```

## Reproducibility

- All random seeds are set via configuration
- Configuration files are version-controlled
- Results include full metadata (model version, hyperparameters)
- Requirements.txt pins exact dependency versions

## Citation

If you use this pipeline in your research, please cite:

```
[Your citation here]
```

## Dataset

OBSEA (Ocean observatory) CTD data:
- Location: NW Mediterranean Sea
- Variables: Temperature, Salinity, Sound Velocity, Pressure
- Frequency: 30-minute intervals
- Duration: ~15 years (2009-2024)

## License

[Your license here]

## Contact

[Your contact  information]

## Acknowledgments

- OBSEA observatory team
- [Funding sources]
