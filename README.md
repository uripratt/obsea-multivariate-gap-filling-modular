# OBSEA Multivariate Gap-Filling & Oceanographic Analysis Pipeline (v2.0)

A scientifically rigorous, high-performance pipeline for the integration, quality control, and multivariate reconstruction of oceanographic time-series from the OBSEA Underwater Observatory (Vilanova i la Geltrú, NW Mediterranean).

This repository contains the modular architecture developed for advanced oceanographic data processing. It supersedes previous monolithic scripts by introducing a formal component separation optimized for automated benchmarking and scalable imputation of missing data.

---

## Architecture and Core Modules

The system is structured as a Python package (`obsea_pipeline/`) targeting PhD-level research standards:

- **Ingestion (`sta_connector.py`)**: Designed for systematic synchronization with the SensorThings API (STA) v1.1. It enables recursive fetching across multiple instrument groups and manages depth-bin selection for ADCP (AWAC) current profiling mechanisms.
- **Quality Control (`qc/`)**: Implements strict QARTOD standards (Quality Assurance/Quality Control of Real-Time Oceanographic Data), further stabilized through a Rolling Median Absolute Deviation (MAD) algorithm. This dual approach provides robust detection and soft-flagging of sensor artifacts such as spikes, flatlines, and abrupt gradient shifts.
- **Oceanographic Physical Derivations (`preprocessing/`)**: Employs fully vectorized NumPy calculations compliant with the TEOS-10 standard to dynamically compute absolute derivations:
  - Seawater Density Anomaly ($\sigma_t$).
  - Vertical Stratification and Brunt-Väisälä Buoyancy Frequency ($N^2$).
  - Wind Stress ($\tau$) and decomposed Wind U/V vectors.
- **Gap Analysis Engine (`gaps/`)**: A Scale-Aware detection mechanism that categorizes Missing Not At Random (MNAR) data phenomena into physical temporal scales: *Micro (0-6h), Short (6-24h), Medium (1-3d), Long (3-7d), Extended (7-30d), and Gigant (>30d)*.

---

## Scientific Benchmarking Methodology

To solve the prevalent issue of artificial model assessment, the evaluation framework has been strictly constrained to guarantee absolute Ground Truth validity during simulation:

1. **Topological Contiguity Constraints**: Simulates authentic sensor failures by injecting NaN segments instead of randomly scattering distinct points.
2. **Ground Truth Integrity Verification**: Artificial gaps are exclusively placed over temporally continuous 100% observed data periods. This removes the inherent bias of evaluating Root Mean Square Error (RMSE) against previously interpolated values.
3. **Bilateral Context Requisites**: The masking algorithm necessitates a minimum surrounding buffer of 48 hours of uninterrupted natural data adjacent to each synthetic gap, verifying that all predictive algorithms can utilize legitimate historical contexts.
4. **Spectral Error Metrics (PSD)**: Recognizing that classic error metrics (RMSE, MAE) inherently favor models that temporally smooth predictions, this pipeline implements Welch’s method to estimate Power Spectral Density (PSD). Comparing the Log-Power Spectra between the ground truth and the imputed signal severely penalizes models that fail to preserve high-frequency physical phenomena such as tidal harmonics or internal waves.

---

## Evaluated Imputation Typologies

The benchmarking suite classifies models into five distinct typologies based on architectural complexity to drive the scale-aware routing engine:

| Typology | Evaluated Models | Core Mechanism | Pipeline Role (Target Gap Scale) |
| :--- | :--- | :--- | :--- |
| **1. Topological & Univariate** | Linear, Time, Splines, PCHIP | Fast polynomial/linear estimations bridging immediate gap edges without external variables. | **Baseline**. Ideal for **Micro (<6h)** gaps maintaining physical inertia. |
| **2. Multivariate Statistical** | VARMA | Explores classical linear cross-correlations (e.g. Temp vs Salinity). | **Benchmark Reference**. Struggles with long-term non-stationarity. |
| **3. Iterative Machine Learning** | XGBoost Pro | GPU-optimized Gradient Boosting using cyclical temporal encoding and full multivariate correlation matrices. | **Speed/Precision Champion**. Deployed for **Short/Medium (6h-72h)** gaps. |
| **4. Recurrent Neural Networks** | Bi-LSTM | Bidirectional sequence analysis retaining the physical momentum of the ocean before and after gaps. | **Inertia Capturing**. Ideal for complex **Medium (1d-3d)** gaps. |
| **5. Attention & Transformers** | SAITS, BRITS, ImputeFormer | SOTA Deep Learning utilizing self-attention to map long-range, cross-variable physical relationships. | **Heavy Artillery**. Essential for **Long, Extended & Gigant (>3d)** missing events. |


---

## Execution Interface (CLI)

The pipeline is orchestrated via `main_obsea.py`, providing a declarative interface for the operational modes:

| Mode | Command | Description |
| :--- | :--- | :--- |
| **Ingest** | `--mode ingest` | Direct programmatic fetch and unified DataFrame creation without executing models. |
| **Benchmark** | `--mode benchmark` | Rigorous, multi-model simulation. Customizable via parameter limits. |
| **Production** | `--mode production` | Autonomous operation executing the Scale-Aware model routing. Resolves true missing values using the historically proven model per gap interval. |
| **Plot** | `--mode plot` | Visualization utility for immediate instrumental QC auditing. |

### Operational Flags
The execution can be finely tuned using the following CLI arguments:

| Flag | Parameter | Description |
| :--- | :--- | :--- |
| **`--start`** | `YYYY-MM-DD` | Historical start date for data ingestion and processing. |
| **`--end`** | `YYYY-MM-DD` | Historical end date for data ingestion and processing. |
| **`--limit`** | `Integer` | Restricts the execution to the last *N* days (calculated retroactively from `--end` or today). |
| **`--no-cache`** | *N/A* | Forces the pipeline to bypass the local `.parquet` state and re-execute STA API ingestion and Vectorized QC processing from scratch. |
| **`--methods`** | `string list` | (Benchmark Mode Only). A space-separated list of model keys to selectively evaluate (e.g., `linear varma xgboost_pro`), bypassing heavier unselected models. |

### Usage Example:
To perform a local benchmarking routine spanning a pre-validated high-availability 3-year contiguous window:

```bash
python3 main_obsea.py --mode benchmark --start 2018-05-01 --end 2021-05-01 --no-cache --methods linear time splines varma xgboost_pro
```

---

## Web Visualization Ecosystem

The computational backend exports artifacts specifically structured for decoupling to a lightweight JavaScript Dashboard (`webapp/`):
- High-efficiency representation of multivariate time series.
- Category-based model benchmarking heatmaps emphasizing Spectral vs. Absolute Error tradeoffs.
- Pre-calculated multidimensional correlation matrices to prevent client-side computational freezing.

---

## Scientific Context

This project serves as the computational framework for doctoral research in **Multivariate Imputation Methods for Marine Observatories**. It addresses the classical signal smoothing limitations of standard temporal interpolation by actively utilizing non-linear physical cross-correlations and multi-head attention components within varying Deep Learning architectures (Transformers, Dual-Direction RNNs).

*OBSEA Observatory - SARTI/UPC Research Group*
