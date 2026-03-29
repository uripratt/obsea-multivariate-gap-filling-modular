import pandas as pd
import numpy as np
import os

# Paths
HISTORICAL_CSV = "/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts/exported_data/adcp/historical_adcp_unified_2010_2025.csv"
API_CURRENTS_CSV = "/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts/exported_data/OBSEA_AWAC_currents_API_binned.csv"
API_WAVES_CSV = "/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts/exported_data/RAW/OBSEA_AWAC_waves_full_nc_RAW.csv"
PRODUCTION_CSV = "/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts/output_lup/data/OBSEA_multivariate_30min.csv"
OUTPUT_CSV = "/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts/output_lup/data/OBSEA_multivariate_unified_30min.csv"

# Mapping: Historical Name -> API Binned Name
MAPPING = {
    "AWAC_Hm0": "VHM0",
    "AWAC_Tp": "VTPK",
    "AWAC_WDIR": "VMDR",
    "AWAC18M_CSPD": "CSPD_BOT",
    "AWAC18M_CDIR": "CDIR_BOT",
    "AWAC18M_UCUR": "UCUR_BOT",
    "AWAC18M_VCUR": "VCUR_BOT",
    "AWAC18M_ZCUR": "ZCUR_BOT",
    "AWAC2M_CSPD": "CSPD_SURF",
    "AWAC2M_CDIR": "CDIR_SURF"
}

def load_api_csv(path):
    print(f"Loading {os.path.basename(path)}...")
    df = pd.read_csv(path, low_memory=False)
    # The new files use "TIME" column with string timestamps or epoch
    if 'TIME' in df.columns:
        df.index = pd.to_datetime(df['TIME'], errors='coerce')
        # If all NaT, try epoch
        if df.index.isnull().all():
            df.index = pd.to_datetime(pd.to_numeric(df['TIME'], errors='coerce'), unit='s')
    else:
        df = pd.read_csv(path, index_col=0, low_memory=False)
        df.index = pd.to_datetime(df.index, errors='coerce')
            
    df = df[df.index.notnull()].sort_index()
    if 'TIME' in df.columns:
        df = df.drop(columns=['TIME'])
        
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Resample to 30min
    return df.resample('30T').mean()

def run_fusion():
    print("Starting OBSEA Data Fusion (API + Historical Nortek)...\n")
    
    # 1. Load Datasets
    df_hist = pd.read_csv(HISTORICAL_CSV, index_col='Timestamp', parse_dates=True)
    df_api_cur = load_api_csv(API_CURRENTS_CSV)
    df_api_wav = load_api_csv(API_WAVES_CSV)
    
    # FIX: Correct the +1 hour Local Time vs UTC lag in the API waves dataset
    df_api_wav.index = df_api_wav.index - pd.Timedelta(hours=1)
    df_prod = pd.read_csv(PRODUCTION_CSV, index_col=0, parse_dates=True)
    
    # 2. Prepare Unified AWAC
    print("Merging AWAC API components...")
    df_api_awac = df_api_cur.join(df_api_wav, how='outer', lsuffix='_cur', rsuffix='_wav')
    
    # 3. Apply Fusion with Priority to Historical data
    print("Fusing Historical data with API data...")
    unified_awac_cols = {}
    for hist_var, std_name in MAPPING.items():
        if hist_var in df_hist.columns:
            s_hist = df_hist[hist_var]
            if std_name in df_api_awac.columns:
                s_api = df_api_awac[std_name]
                # Combine: Historical fills the main trunk, API fills gaps if hist is missing
                # Actually historical coverage is much better, let's use it as base.
                s_unified = s_hist.combine_first(s_api)
                name = f"AWAC_{std_name}"
                unified_awac_cols[name] = s_unified
                print(f"Fused {std_name} (Hist + API) -> {name}")
            else:
                name = f"AWAC_{std_name}"
                unified_awac_cols[name] = s_hist
                print(f"Added {std_name} (Hist only) -> {name}")

    df_unified_awac = pd.DataFrame(unified_awac_cols)
    
    # 4. Integrate into Production Dataset
    print("\nIntegrating AWAC into Production Dataset...")
    # Join on index
    df_final = df_prod.join(df_unified_awac, how='outer')
    
    # Sort and clip to reasonable range (2009 to 2025)
    df_final = df_final.sort_index()
    df_final = df_final[df_final.index >= '2009-01-01']
    
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Total columns: {list(df_final.columns)}")
    
    # 5. Save Output
    print(f"\nSaving unified dataset to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV)
    print("Fusion Complete.")

if __name__ == "__main__":
    run_fusion()
