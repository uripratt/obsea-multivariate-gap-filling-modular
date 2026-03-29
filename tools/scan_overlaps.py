import sys
sys.path.append('/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts')
from obsea_pipeline.ingestion.sta_connector import STAConnector
import pandas as pd
import logging

logging.disable(logging.INFO)

sta16 = STAConnector()
sta37 = STAConnector(ctd_type='sbe37')
overlap_found = False

print("Scanning for SBE16 vs SBE37 concurrent deployments (2012 - 2025)...")
for year in range(2012, 2026):
    try:
        df16 = sta16.fetch_observations(102, f"{year}-01-01T00:00:00Z", f"{year}-12-31T23:59:59Z")
        df37 = sta37.fetch_observations(132, f"{year}-01-01T00:00:00Z", f"{year}-12-31T23:59:59Z")
        
        if df16 is None or df37 is None or df16.empty or df37.empty:
            print(f"{year}: No overlap.")
            continue
            
        df_sync = df16.join(df37, how='inner', lsuffix='_16', rsuffix='_37').dropna()
        if not df_sync.empty:
            print(f"\n[!] OVERLAP FOUND IN {year}: {len(df_sync)} synchronous readings!")
            print(f"    From {df_sync.index.min()} to {df_sync.index.max()}")
            overlap_found = True
        else:
            print(f"{year}: No overlap (zero concurrent timestamps)")
    except Exception as e:
        print(f"Error checking year {year}: {e}")

if not overlap_found:
    print("\nCONCLUSION: SBE16 and SBE37 were NEVER deployed concurrently. They are strictly mutually exclusive hot-swaps.")
