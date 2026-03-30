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
    
    # Step 2: Load the ingested data
    import pandas as pd
    ingested_path = os.path.join(CONFIG['output_dir'], "ingested_data.csv")
    df = pd.read_csv(ingested_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

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
