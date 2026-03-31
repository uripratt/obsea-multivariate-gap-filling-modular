import pandas as pd
import numpy as np
import os

class AWACProcessor:
    """
    Handles the ingestion, correction, and fusion of AWAC ADCP raw Historical data and 
    STA API real-time data, exporting a Triple-Variant multivariate dataset.
    """
    def __init__(self):
        self.mapping = {
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

    def load_api_csv(self, path):
        print(f"Loading API file: {os.path.basename(path)}...")
        df = pd.read_csv(path, low_memory=False)
        if 'TIME' in df.columns:
            df.index = pd.to_datetime(df['TIME'], errors='coerce')
        else:
            df = pd.read_csv(path, index_col=0, low_memory=False)
            df.index = pd.to_datetime(df.index, errors='coerce')
            
        df = df[df.index.notnull()].sort_index()
        if 'TIME' in df.columns:
            df = df.drop(columns=['TIME'])
            
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter NoData
        df.replace([-999.0, -999.9, -999.99, 99.9, 99.99], np.nan, inplace=True)
        return df.resample('30T').mean()

    def fuse_and_export(self, hist_csv, api_cur_csv, api_wav_csv, prod_csv, output_csv):
        print("\n=== AWAC Triple-Variant Fusion Engine ===")
        # 1. Load Data
        df_hist = pd.read_csv(hist_csv, index_col='Timestamp', parse_dates=True)
        df_api_cur = self.load_api_csv(api_cur_csv)
        df_api_wav = self.load_api_csv(api_wav_csv)
        df_prod = pd.read_csv(prod_csv, index_col=0, parse_dates=True)
        
        # Clean the base dataset to remove any existing model predictions from previous runs
        model_cols = [c for c in df_prod.columns if any(mod in c for mod in ['_TIME', '_SPLINES', '_VARMA', '_XGBOOST', '_IMPUTEFORMER', '_ANOMALY', '_BILSTM', '_SAITS', '_BRITS'])]
        
        # CRITICAL FIX: Also drop any base AWAC columns that might have come natively from the API extraction 
        # so they do NOT overlap with our unified Tri-Variant injection!
        awac_cols = [c for c in df_prod.columns if 'AWAC' in c]
        
        cols_to_drop = list(set(model_cols + awac_cols))
        
        if cols_to_drop:
            print(f"Removing {len(cols_to_drop)} legacy/overlapping columns from base dataset...")
            df_prod = df_prod.drop(columns=cols_to_drop)
        
        # 2. Correct API Waves Time Shift (-1 Hour)
        print("Applying -1 Hour UTC correction to API Wave dataset...")
        df_api_wav.index = df_api_wav.index - pd.Timedelta(hours=1)
        
        # Merge API components
        df_api_awac = df_api_cur.join(df_api_wav, how='outer', lsuffix='_cur', rsuffix='_wav')
        
        # 3. Create Triple-Variant Structure
        print("Generating _HIST, _API, and Unified Ground Truth variants...")
        variant_cols = {}
        
        for hist_var, api_var in self.mapping.items():
            unified_name = hist_var if "AWAC" in hist_var else f"AWAC_{hist_var}"
            if "AWAC_" not in unified_name and not unified_name.startswith("AWAC"):
                unified_name = f"AWAC_{unified_name}"
            
            # Specific clean names for Waves
            if unified_name == "AWAC_Hm0": unified_name = "AWAC_VHM0"
            if unified_name == "AWAC_Tp": unified_name = "AWAC_VTPK"
            if unified_name == "AWAC_WDIR": unified_name = "AWAC_VMDR"
                
            s_hist = df_hist[hist_var] if hist_var in df_hist.columns else pd.Series(dtype=float)
            s_api = df_api_awac[api_var] if api_var in df_api_awac.columns else pd.Series(dtype=float)
            
            # The Unified variable priorities HIST over API gaps
            s_unified = s_hist.combine_first(s_api)
            
            # Store the 3 variants
            variant_cols[f"{unified_name}_HIST"] = s_hist
            variant_cols[f"{unified_name}_API"] = s_api
            variant_cols[unified_name] = s_unified
            
            print(f"  Processed {unified_name} (Hist points: {s_hist.count()}, API points: {s_api.count()}, Unified points: {s_unified.count()})")

        df_variants = pd.DataFrame(variant_cols)
        
        # 4. Integrate with Main Production Set
        print("\nIntegrating AWAC Variants into main Production Dataset...")
        df_final = df_prod.join(df_variants, how='outer')
        df_final.index = pd.to_datetime(df_final.index, errors='coerce')
        df_final = df_final[df_final.index.notnull()]
        df_final = df_final.sort_index()
        df_final = df_final[df_final.index >= pd.to_datetime('2009-01-01')]
        
        print(f"Final dataset shape: {df_final.shape}")
        
        # Enforce pure integer formatting for all QC flags (strip .0 appended by pandas NaN floats)
        for c in df_final.columns:
            if c.endswith('_QC'):
                df_final[c] = df_final[c].round().astype('Int64')
        
        # Export
        print(f"Exporting to: {output_csv}...")
        df_final.to_csv(output_csv)
        print("Fusion and Export Complete! You can now analyze the 3 variants side-by-side.")

if __name__ == "__main__":
    processor = AWACProcessor()
    
    BASE = "/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts"
    processor.fuse_and_export(
        hist_csv=f"{BASE}/exported_data/adcp/historical_adcp_unified_2010_2025.csv",
        api_cur_csv=f"{BASE}/exported_data/OBSEA_AWAC_currents_API_binned.csv",
        api_wav_csv=f"{BASE}/exported_data/RAW/OBSEA_AWAC_waves_full_nc_RAW.csv",
        prod_csv=f"{BASE}/output_lup/data/OBSEA_multivariate_30min.csv",
        output_csv=f"{BASE}/output_lup/data/OBSEA_multivariate_unified_30min.csv"
    )
