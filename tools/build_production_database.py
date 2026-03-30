import os
import sys
import subprocess

# Add parent dir to sys.path
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TOOLS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from obsea_pipeline.config.settings import CONFIG
from obsea_pipeline.ingestion.awac_processor import AWACProcessor
from obsea_pipeline.preprocessing.oceanography import add_derived_features

import argparse

def build_golden_database(start_date, end_date):
    print(f"=== Building Golden Dataset (API + AWAC + TEOS-10) for {start_date} to {end_date} ===")
    
    # Step 1: Ingest RAW data from STA API using the modular orchestrator
    print(f"\n[1/4] Fetching raw telemetry from STA API ({start_date} -> {end_date})...")
    cmd = [
        "python3", "main_obsea.py", 
        "--mode", "ingest", 
        "--start", start_date, 
        "--end", end_date, 
        "--no-cache"
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    
    # Step 2: Load the ingested API data
    import pandas as pd
    import numpy as np
    ingested_path = os.path.join(CONFIG['output_dir'], "ingested_data.csv")
    df_api = pd.read_csv(ingested_path)
    df_api['Timestamp'] = pd.to_datetime(df_api['Timestamp']).dt.tz_localize(None)
    df_api.set_index('Timestamp', inplace=True)

    # NEW: Step 2.5: Fuse with High-Density CTD NetCDF Archive
    print("\n[1.5/4] Fusing high-density CTD NetCDF Archive (2009-2024)...")
    ctd_arch_path = "data/exported_data/RAW/OBSEA_CTD_30min_nc_RAW.csv"
    if os.path.exists(ctd_arch_path):
        df_ctd_arch = pd.read_csv(ctd_arch_path)
        df_ctd_arch['TIME'] = pd.to_datetime(df_ctd_arch['TIME']).dt.tz_localize(None)
        df_ctd_arch.set_index('TIME', inplace=True)
        # Standardize column names to match main pipeline
        ctd_map = {'TEMP': 'TEMP', 'PSAL': 'PSAL', 'CNDC': 'CNDC', 'PRES': 'PRES', 'SVEL': 'SVEL'}
        # Clean archive: -999 to NaN
        df_ctd_arch.replace([-999.0, -999.99, 99.99], np.nan, inplace=True)
        
        # Priority merge: Archive first, then API gaps
        for var in ctd_map.keys():
            if var in df_api.columns and var in df_ctd_arch.columns:
                print(f"  Merging {var} (Arch: {df_ctd_arch[var].count()}, API: {df_api[var].count()})")
                df_api[var] = df_ctd_arch[var].combine_first(df_api[var])
                # Also merge QC flags if present
                qc_col = f"{var}_QC"
                if qc_col in df_api.columns and qc_col in df_ctd_arch.columns:
                    df_api[qc_col] = df_ctd_arch[qc_col].combine_first(df_api[qc_col])
        
        df = df_api
    else:
        print("  WARNING: CTD Archive not found. Proceeding with API telemetry only.")
        df = df_api

    # Step 3: Apply TEOS-10 Bio-fouling Correction
    print("\n[2/4] Applying TEOS-10 Bio-fouling restoration to CTD Datastreams...")
    df = add_derived_features(df)
    
    # Step 4: Inject 15-Year AWAC Archives
    print("\n[3/4] Fusing high-fidelity AWAC/ADCP Archival data...")
    processor = AWACProcessor()
    
    hist_csv = "data/exported_data/adcp/historical_adcp_unified_2010_2025.csv"
    api_cur_csv = "data/exported_data/OBSEA_AWAC_currents_API_binned.csv"
    api_wav_csv = "data/exported_data/RAW/OBSEA_AWAC_waves_full_nc_RAW.csv"
    
    final_out = os.path.join(CONFIG['output_dir'], "OBSEA_full_golden_dataset.csv")
    
    # Prepare a temporary base for fusion
    temp_base = os.path.join(CONFIG['output_dir'], "temp_base_for_fusion.csv")
    df.to_csv(temp_base)
    
    processor.fuse_and_export(
        hist_csv=hist_csv,
        api_cur_csv=api_cur_csv,
        api_wav_csv=api_wav_csv,
        prod_csv=temp_base,
        output_csv=final_out
    )
    
    print(f"\n[4/4] SUCCESS! Full Golden Dataset generated at: {final_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    args = parser.parse_args()
    
    build_golden_database(start_date=args.start, end_date=args.end)
