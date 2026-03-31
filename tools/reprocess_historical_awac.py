import os
import pandas as pd
import numpy as np
import tarfile
import glob
from datetime import datetime

# ==============================================================================
# OBSEA AWAC ARCHIVAL PARSER (Scientific v2.0)
# Based on User-Provided Schemas for .wap (Waves) and .wpa (Currents)
# ==============================================================================

ADCP_DIR = "data/exported_data/adcp/"
OUTPUT_MASTER = os.path.join(ADCP_DIR, "historical_adcp_unified_2010_2025.csv")
TMP_EXTRACT = os.path.join(ADCP_DIR, "tmp_restoration")

# SCHEMA: .wap (Waves)
# 0:Day 1:Month 2:Year 3:Hour 4:Minute 5:Second 6:Hm0 7:Hs 8:H10 9:Hmax 10:Tm02 11:Tp 12:Tm01 13:DirTp 14:Spread 15:MeanDir ...
WAP_COLS = ["Day", "Month", "Year", "Hour", "Minute", "Second", "Hm0", "Hs", "H10", "Hmax", "Tm02", "Tp", "Tm01", "DirTp", "Spread", "MeanDir"]

# SCHEMA: .wpa (Currents Profile)
# Header line: 0:Day 1:Month 2:Year 3:Hour 4:Minute 5:Second ... 14:NumBins
WPA_HEADER_COLS = ["Day", "Month", "Year", "Hour", "Minute", "Second", "ErrorCode", "Ensemble", "Battery", "SoundSpeed", "Heading", "Pitch", "Roll", "Temperature", "Pressure", "Counter", "HeadingStd", "NumBeams", "NumBins"]
# Bin line: 0:Bin 1:Range 2:Speed 3:Direction 4:U 5:V 6:W ...
WPA_BIN_COLS = ["Bin", "Range", "Speed", "Direction", "U", "V", "W", "ErrorVel", "Amp1", "Amp2", "Amp3", "Amp4"]

def parse_wap_content(content):
    """Parses a .wap wave parameter content from memory."""
    data = []
    for line in content.splitlines():
        parts = line.split()
        if len(parts) < 16: continue
        try:
            ts = datetime(int(parts[2]), int(parts[1]), int(parts[0]), int(parts[3]), int(parts[4]), int(parts[5]))
            row = {
                'Timestamp': ts,
                'AWAC_Hm0': float(parts[6]),
                'AWAC_Hmax': float(parts[9]),
                'AWAC_Tp': float(parts[11]),
                'AWAC_WDIR': float(parts[13])
            }
            data.append(row)
        except Exception: continue
    return pd.DataFrame(data)

def parse_wpa_content(content):
    """Parses a .wpa current profile content from memory."""
    data = []
    lines = content.splitlines()
    idx = 0
    while idx < len(lines):
        header_parts = lines[idx].split()
        if len(header_parts) < 16:
            idx += 1
            continue
        try:
            ts = datetime(int(header_parts[2]), int(header_parts[1]), int(header_parts[0]), int(header_parts[3]), int(header_parts[4]), int(header_parts[5]))
            num_bins = int(header_parts[18])
            
            surf_val = {'CSPD': np.nan, 'CDIR': np.nan, 'UCUR': np.nan, 'VCUR': np.nan, 'ZCUR': np.nan}
            bot_val = {'CSPD': np.nan, 'CDIR': np.nan, 'UCUR': np.nan, 'VCUR': np.nan, 'ZCUR': np.nan}
            
            idx += 1
            for b in range(num_bins):
                if idx >= len(lines): break
                bin_parts = lines[idx].split()
                if len(bin_parts) < 8:
                    idx += 1
                    continue
                
                dist = float(bin_parts[1])
                if 1.5 <= dist <= 2.5:
                    surf_val = {'CSPD': float(bin_parts[2]), 'CDIR': float(bin_parts[3]), 'UCUR': float(bin_parts[4]), 'VCUR': float(bin_parts[5]), 'ZCUR': float(bin_parts[6])}
                if 17.5 <= dist <= 19.5:
                    bot_val = {'CSPD': float(bin_parts[2]), 'CDIR': float(bin_parts[3]), 'UCUR': float(bin_parts[4]), 'VCUR': float(bin_parts[5]), 'ZCUR': float(bin_parts[6])}
                idx += 1
            
            row = {
                'Timestamp': ts,
                'AWAC18M_CSPD': bot_val['CSPD'], 'AWAC18M_CDIR': bot_val['CDIR'], 'AWAC18M_UCUR': bot_val['UCUR'], 'AWAC18M_VCUR': bot_val['VCUR'], 'AWAC18M_ZCUR': bot_val['ZCUR'],
                'AWAC2M_CSPD': surf_val['CSPD'], 'AWAC2M_CDIR': surf_val['CDIR'], 'AWAC2M_UCUR': surf_val['UCUR'], 'AWAC2M_VCUR': surf_val['VCUR'], 'AWAC2M_ZCUR': surf_val['ZCUR']
            }
            data.append(row)
        except Exception:
            idx += 1
            continue
    return pd.DataFrame(data)

def reprocess_all():
    all_waves = []
    all_currents = []
    
    tar_files = sorted(glob.glob(os.path.join(ADCP_DIR, "*.tar.gz")))
    
    for tar_path in tar_files:
        year = os.path.basename(tar_path).split('_')[0]
        print(f"Processing Year: {year}...")
        
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".wap"):
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('ascii', errors='ignore')
                        df = parse_wap_content(content)
                        if not df.empty: all_waves.append(df)
                elif member.name.endswith(".wpa"):
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('ascii', errors='ignore')
                        df = parse_wpa_content(content)
                        if not df.empty: all_currents.append(df)

    print("Consolidating datasets...")
    df_waves = pd.concat(all_waves).drop_duplicates(subset='Timestamp').set_index('Timestamp').sort_index()
    df_cur = pd.concat(all_currents).drop_duplicates(subset='Timestamp').set_index('Timestamp').sort_index()
    
    # CRITICAL: Clean error codes BEFORE resampling to prevent mean pollution
    for df in [df_waves, df_cur]:
        df.replace([-999.0, -999.99, -9.99, 99.99, -99.9, -99], np.nan, inplace=True)
    
    df_waves = df_waves.resample('30T').mean()
    df_cur = df_cur.resample('30T').mean()
    
    # Join
    df_master = df_waves.join(df_cur, how='outer')
    
    # FINAL SCIENTIFIC QC
    # Wave physical bounds
    df_master.loc[df_master['AWAC_Hmax'] > 15.0, ['AWAC_Hmax', 'AWAC_Hm0', 'AWAC_Tp']] = np.nan
    df_master.loc[df_master['AWAC_Hm0'] > 10.0, ['AWAC_Hm0', 'AWAC_Hmax']] = np.nan
    
    # Direction normalization (ensure 0-360)
    for col in ['AWAC_WDIR', 'AWAC18M_CDIR', 'AWAC2M_CDIR']:
        if col in df_master.columns:
            df_master[col] = df_master[col] % 360
    
    # Save (Standard CSV formatting, NO interpolation)
    df_master.to_csv(OUTPUT_MASTER)
    print(f"SUCCESS! New 15-year archive generated (RAW): {OUTPUT_MASTER}")
    print(df_master[['AWAC_Hm0', 'AWAC_Hmax', 'AWAC_WDIR']].head(10))

if __name__ == "__main__":
    reprocess_all()
