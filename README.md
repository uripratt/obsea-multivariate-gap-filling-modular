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

The benchmarking suite systematically evaluates a wide array of imputation algorithms. To comprehend their scale-aware routing, the models are classified into five distinct typologies ordered by complexity:

### 1. Topological & Univariate Interpolations
*(Linear, Time, Splines, PCHIP)*
- **Mechanism**: Extremely efficient polynomial or linear estimations bridging the immediate edges of a gap. They do not utilize external variables.
- **Role in Pipeline**: Baseline estimators. Ideal for **Micro (<6h)** gaps where physical oceanographic variables maintain inertia and have not undergone significant non-linear shifts.

### 2. Classical Multivariate Statistical Models
*(VARMA - Vector Autoregression Moving-Average)*
- **Mechanism**: A traditional statistical approach that explores linear cross-correlations (e.g., Temperature reacting to Salinity).
- **Role in Pipeline**: Included as a benchmark baseline. While mathematically sound, it struggles with the characteristic non-stationarity and long-term seasonality of marine temporal series.

### 3. Iterative Machine Learning (Tree-Based)
*(XGBoost Pro)*
- **Mechanism**: A feature-engineered Gradient Boosting mechanism optimized for parallel GPU/CPU execution. It breaks the temporal sequence constraint by treating timestamps as cyclically encoded features (hour, day of year) while leveraging the full multivariate correlation matrix.
- **Role in Pipeline**: Represents the optimal Pareto frontier of speed vs. precision. It is the reigning champion for **Short to Medium (6h-72h)** gaps, heavily penalizing local noise without catastrophic memory overhead.

### 4. Deep Learning: Recurrent Neural Networks
*(Bi-LSTM - Bidirectional Long Short-Term Memory)*
- **Mechanism**: Analyzes the continuous sequence of data forwards and backwards. The hidden states act as "memory cells" capable of retaining the physical momentum of the ocean before the gap occurs and reconciling it with the states observed immediately after.
- **Role in Pipeline**: Exceptional at capturing physical inertia, typically deployed for complex **Medium (1d-3d)** gaps where temporal memory is strictly required.

### 5. Deep Learning: Attention & Transformer Architectures
*(PyPOTS Framework: SAITS, BRITS, ImputeFormer)*
- **Mechanism**: State-of-the-Art models specifically tailored for irregularly sampled multivariate time series. Operating on self-attention mechanisms and dual-directional RNNs, they compute complex, long-range cross-variable physical attention maps (e.g., deducing deep underwater currents solely from sustained surface wind stress).
- **Role in Pipeline**: The heavy artillery. Deployed exclusively for **Long, Extended, and Gigant (>3d)** gaps. These models are capable of reconstructing entirely missing physical events (like a multi-day storm) by inferring the missing data from the remaining active instruments.

---

## Execution Interface (CLI)

The pipeline is orchestrated via `main_obsea.py`, providing a declarative interface for the operational modes:

| Mode | Command | Description |
| :--- | :--- | :--- |
| **Ingest** | `--mode ingest` | Direct programmatic fetch and unified DataFrame creation without executing models. |
| **Benchmark** | `--mode benchmark` | Rigorous, multi-model simulation. Customizable via parameter limits. |
| **Production** | `--mode production` | Autonomous operation executing the Scale-Aware model routing. Resolves true missing values using the historically proven model per gap interval. |
| **Plot** | `--mode plot` | Visualization utility for immediate instrumental QC auditing. |

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
