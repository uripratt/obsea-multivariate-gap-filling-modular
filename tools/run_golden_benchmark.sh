#!/bin/bash

# OBSEA Golden Benchmark Automation Script
# Purpose: Generate the high-fidelity dataset from API + Fixes and launch the 5-method benchmark.
# Expected Environment: GPU-enabled server (e.g. NVIDIA A40) with requirements.txt installed.

set -e # Exit immediately if a command exits with a non-zero status.

echo "================================================================"
echo "   OBSEA MODULAR PIPELINE: GOLDEN BENCHMARK (2018-2019)         "
echo "================================================================"

# 1. Build the Golden Dataset (API + TEOS-10 + AWAC Fusion)
# This will take several minutes to fetch the 2-year window from the STA API.
echo "[STEP 1/2] Generating High-Fidelity Unified Dataset..."
python3 tools/build_production_database.py

# 2. Run the Benchmark
# This evaluates Linear, XGBoost, SAITS, BRITS, and BiLSTM models.
# Results will be saved in output_lup/tables/ and output_lup/figures/
echo "[STEP 2/2] Launching 5-Method Multivariate Benchmark Suite..."
python3 main_obsea.py \
  --mode benchmark \
  --csv-input output_lup/data/OBSEA_golden_unified_2018_2019.csv \
  --methods linear xgboost saits brits bilstm \
  --start 2018-01-01 \
  --end 2019-12-31 \
  --no-cache

echo "================================================================"
echo "   BENCHMARK COMPLETE!                                          "
echo "   Check 'output_lup/figures/' for comparison charts.           "
echo "================================================================"
