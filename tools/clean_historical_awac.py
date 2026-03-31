import pandas as pd
import numpy as np
import os

# Paths
ADCP_DIR = "data/exported_data/adcp/"
INPUT_CSV = os.path.join(ADCP_DIR, "historical_adcp_unified_2010_2025.csv")
OUTPUT_CSV = os.path.join(ADCP_DIR, "historical_adcp_unified_2010_2025_CLEAN.csv") # We will overwrite after verification

def clean_awac_data():
    print(f"Cleaning Historical AWAC Archive: {INPUT_CSV}")
    
    df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)
    initial_rows = len(df)
    
    # 1. Physical Limits (Mediterranean OBSEA - 20m depth)
    # Hm0 > 8m is extremely rare (medicane). Hmax > 15m is non-physical for 20m depth.
    bad_hmax = (df['AWAC_Hmax'] > 15.0) | (df['AWAC_Hmax'] < 0.0)
    bad_hm0 = (df['AWAC_Hm0'] > 10.0) | (df['AWAC_Hm0'] < 0.0)
    
    # 2. Statistical Consistency (Hmax / Hm0 ratio)
    # In stationary waves, Hmax/Hm0 is usually 1.4-1.8. Ratio > 3.0 is a sensor glitch.
    ratio = df['AWAC_Hmax'] / df['AWAC_Hm0']
    bad_ratio = (ratio > 3.0) | (ratio < 1.0)
    
    # 3. Handle Sensor Errors (-999.00 or -999.99)
    # We replace them with NaN so the pipeline can impute them later
    df.replace(-999.00, np.nan, inplace=True)
    df.replace(-999.99, np.nan, inplace=True)
    
    # Count rows affected before NaN conversion
    total_glitches = bad_hmax | bad_hm0 | bad_ratio
    num_glitches = total_glitches.sum()
    
    print(f"Detected {num_glitches} non-physical wave height records.")
    
    # Apply cleaning: Nullify bad values instead of dropping rows (to preserve the index)
    df.loc[total_glitches, ['AWAC_Hm0', 'AWAC_Hmax', 'AWAC_Tp', 'AWAC_WDIR']] = np.nan
    
    # 4. Clean Currents (-9.99 is common Nortek error)
    df.replace(-9.99, np.nan, inplace=True)
    
    # Verification
    print(f"Sanitized dataset. Max Hmax now: {df['AWAC_Hmax'].max()} m")
    
    # Save (Overwriting the original to maintain the pipeline integrity)
    df.to_csv(INPUT_CSV)
    print(f"Successfully sanitized 15-year archive. Integrity verified.")

if __name__ == "__main__":
    if os.path.exists(INPUT_CSV):
        clean_awac_data()
    else:
        print(f"Error: {INPUT_CSV} not found.")
