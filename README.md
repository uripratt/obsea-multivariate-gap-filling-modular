# OBSEA Multivariate Time Series Imputation Benchmark 🌊

A scientifically rigorous, reproducible, and computationally optimized pipeline for evaluating **Machine Learning**, **Deep Learning**, and **Classical Statistical Methods** in the context of filling gaps in oceanographic time series (CTD data from the OBSEA underwater observatory).

## 🎯 Overview & Objectives

Oceanographic datasets frequently suffer from prolonged data gaps due to bio-fouling, sensor malfunction, or severe storms. This project provides a robust testing ground to compare how different families of algorithms perform when reconstructing missing values, leveraging *multivariate correlations* (e.g., using Salinity and Sound Velocity to reconstruct Temperature).

This pipeline has been carefully audited and designed to prevent temporal data leakage and preserve autoregressive behavior, ensuring that Deep Learning models evaluate actual forecasting capability instead of simple interpolation.

## 🧠 Supported Models

The benchmark evaluates the following gap-filling techniques:

1. **Classical & Statistical:**
   - Linear Interpolation
   - Time-weighted Interpolation
   - Spline Interpolation (Cubic)
   - Polynomial Interpolation
   - VARMA (Vector Autoregression Moving-Average)

2. **Machine Learning:**
   - **XGBoost:** Trained dynamically on synthetic gaps to prevent 1-step-ahead data leakage.
   - **XGBoost PRO:** Enhanced with temporal features (lags, rolling stats) and residual learning.
   - **MissForest:** Iterative random forest imputation.

3. **Deep Learning (PyTorch & PyPOTS):**
   - **Multivariate Bi-LSTM:** Custom bidirectional LSTM with step-by-step recursive inference to preserve AR properties.
   - **ImputeFormer:** Transformer-based imputation architecture.
   - **SAITS:** Self-Attention-based Imputation for Time Series.
   - **BRITS & BRITS PRO:** Bidirectional Recurrent Imputation for Time Series.

## 📁 Repository Structure

The repository is modular and focused strictly on the gap-filling pipeline:

```text
├── gap_project_antigr/          # Core Python Library
│   ├── src/
│   │   ├── models/              # Model wrappers (PyTorch, PyPOTS, XGBoost scripts)
│   │   ├── features/            # Temporal & multivariate feature engineering
│   │   ├── evaluation/          # Gap analysis and metrics engines
│   │   ├── utils/               # Configuration and logging
│   └── requirements.txt         # Specific ML/DL requirements for the core lib
│
├── scripts/                     # Execution Pipeline 
│   ├── lup_data_obsea_analysis_jupyterhub.py   # Main robust benchmarking pipeline
│   ├── concatenate_nc_files.py                 # (Optional) NetCDF to CSV conversion
│   ├── preproces_data.py                       # (Optional) QC and data scrubbing
│   └── requirements.txt                        # General execution requirements
```

## 🚀 Quick Start Guide

### 1. Environment Setup

It is highly recommended to run this repository in a virtual environment. The deep learning models require PyTorch to be installed; while CPU fallback is supported, a CUDA-enabled GPU is strongly advised.

```bash
# Create and activate virtual environment
python -m venv venv_obseactd
source venv_obseactd/bin/activate

# Install dependencies
pip install -r gap_project_antigr/requirements.txt
pip install -r scripts/requirements.txt
```

### 2. Prepare the Data

By default, the pipeline expects a clean CSV file (`OBSEA_CTD_Science_30min.csv`) in the parent directory of `scripts/`. 

*(Optional)* If you are starting from raw NetCDF files from the OBSEA observatory:
```bash
# 1. Concatenate raw NetCDFs
python scripts/concatenate_nc_files.py

# 2. Apply Quality Control (QC) filtering and smoothing
python scripts/preproces_data.py
```

### 3. Run the Benchmark

The primary execution script is `lup_data_obsea_analysis_jupyterhub.py`. It automatically slices the data, injects synthetic gaps of varying sizes, runs all enabled models, and computes comparative metrics.

```bash
cd scripts
python lup_data_obsea_analysis_jupyterhub.py
```

## ⚙️ Configuration

You can enable or disable specific models and change hardware execution behaviors by modifying the configuration blocks at the top of `scripts/lup_data_obsea_analysis_jupyterhub.py`:

```python
# Hardware Tuning
HARDWARE_CONFIG = {
    'use_parallel_processing': True, # CPU parallelism
    'joblib_n_jobs': -1,             # Use all CPU cores
}

# Model Master Switch
BENCHMARK_MODELS = {
    'linear': True,
    'xgboost_pro': True,
    'bilstm': True,
    'saits': False, # Disable models here
    # ...
}
```

## 📊 Evaluation & Scientific Methodology

The pipeline evaluates performance across different gap categories (e.g., *Micro-gaps* < 6 hours, *Storm-gaps* up to 3 days). 

For each category, it injects deterministic synthetic missing data sections into the actual observational data, and forces the models to reconstruct those sections. It then compares the reconstruction against the hidden ground truth.

**Metrics Calculated:**
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **Precision %** (Tolerance-based accuracy)

All Deep Learning and ML models utilize strictly `StandardScaler` normalized data internally to prevent gradient divergence, and predictions are automatically denormalized back to physically coherent oceanographic ranges (Celsius, Psu, m/s).

## 📄 License & Academic Integrity

This code was developed as part of academic research at the Technical University of Catalonia (UPC). If utilizing this codebase or its logic for academic purposes, please provide appropriate attribution.
