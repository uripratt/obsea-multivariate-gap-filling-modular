# OBSEA Multivariate Gap-Filling & Oceanographic Analysis Pipeline (v2.0)

A scientifically rigorous, high-performance pipeline for the integration, quality control, and multivariate reconstruction of oceanographic time-series from the **OBSEA Underwater Observatory** (Vilanova i la Geltrú, NW Mediterranean).

---

## 🏛️ Project Architecture: Modular Evolution
The project has evolved from a monolithic legacy version into a **professional Python package** (`obsea_pipeline/`) designed for PhD-level research and production-grade monitoring.

- **Ingestion (`sta_connector.py`)**: Seamless integration with **SensorThings API (STA) v1.1**, featuring recursive fetching and depth-bin selection for ADCP (AWAC) current profiles.
- **Quality Control (`qc/`)**: Advanced implementation of **QARTOD standards** combined with **Rolling MAD (Median Absolute Deviation)** for robust outlier detection and soft-flagging.
- **Oceanographic Foundations (`preprocessing/`)**: 100% vectorized calculations (TEOS-10 compliant) for:
    - Seawater Density ($\sigma_t$).
    - Vertical Stratification and Buoyancy Frequency ($N^2$).
    - Wind Stress ($\tau$) and Wind U/V components.
- **Gap Analysis Engine (`gaps/`)**: A **Scale-Aware** detection system that classifies missing data into six categories: *Micro (0-6h), Short (6-24h), Medium (1-3d), Long (3-7d), Extended (7-30d), and Gigant (>30d)*.

---

## 🌊 Scientific Benchmarking Methodology
The pipeline features a best-in-class benchmarking suite that ensures **Ground Truth Integrity**:
- **Artificial Contiguous Gaps**: Simulates realistic sensor failures by injecting blocks of NaNs.
- **100% Completeness Guard**: Gaps are strictly placed over 100% complete data segments to ensure valid error quantification.
- **Bilateral Context Validity**: Ensures 48h of valid data before and after each gap for fair model context learning.
- **Spectral Metrics (PSD Error)**: Beyond RMSE/MAE, the pipeline uses **Power Spectral Density (PSD)** via Welch's method to penalize models that smooth high-frequency signals (e.g., tides, internal waves).

---

## 🚀 Execution Modes (CLI)
Orchestrated via `main_obsea.py`, the pipeline supports high-level operational modes:

| Mode | Command | Description |
| :--- | :--- | :--- |
| **Ingest** | `--mode ingest` | Clean fetch and synchronization of instruments. |
| **Benchmark** | `--mode benchmark` | Rigorous model evaluation (supports `--methods` and `--limit`). |
| **Production** | `--mode production` | Automated scale-aware filling for real gaps using winning models. |
| **Plot** | `--mode plot` | Multi-panel QC visualizations for instrument groups. |

### Example usage:
```bash
# Run a selective local benchmark for the last 30 days
python3 main_obsea.py --mode benchmark --limit 30 --no-cache --methods linear varma xgboost_pro
```

---

## 🖥️ Web Dashboard (Interactive Frontend)
The pipeline exports results directly to a **Modern Premium Dashboard** (`webapp/`):
- **Performance Heatmaps**: Multi-metric model comparison (RMSE vs Spectral Error).
- **Scale-Aware Table**: Category-based model selection ranking.
- **Correlation Matrix**: Pre-calculated multivariate relationship panel.
- **Dynamic Gap Visualization**: Interactive timeline of observatory status.

---

## 🛠️ Technology Stack
- **Core Engine**: Python 3.12+, Pandas, NumPy (Vectorized Operations), Scipy (Spectral Analysis).
- **Machine Learning**: XGBoost (Highly Optimized Loops), PyTorch (SAITS, BRITS, ImputeFormer - GPU Accelerated).
- **Storage**: Parquet for high-precision archival, JSON for Webapp metadata.
- **Frontend**: Vanilla JS, Plotly.js, Rich (CLI UI).

---

## 🎓 PhD Research Context
This project serves as the technical backbone for a PhD thesis focused on **Multivariate Imputation Methods for Marine Observatories**. It aims to solve the problem of signal smoothing in traditional interpolation by leveraging deep learning architectures (Transformers, RNNs with attention) and physical cross-correlations.

*OBSEA Observatory - SARTI/UPC Research Group*
