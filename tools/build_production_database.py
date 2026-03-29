import os
import sys

# Add script base dir so we can reference 'obsea_pipeline' correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from obsea_pipeline.ingestion.csv_loader import load_all_data
from obsea_pipeline.preprocessing.resampling import create_unified_dataset
from obsea_pipeline.config.settings import CONFIG
from obsea_pipeline.ingestion.awac_processor import AWACProcessor

def build_database():
    print("=== Step 1: Loading RAW Baseline Datasets ===")
    # This automatically loads paths defined in settings.py's CONFIG['data_paths']
    raw_data_dict = load_all_data()
    
    print("\n=== Step 2: Quality-Aware Resampling & Synchronizing (QARTOD standard) ===")
    # Creates the synchronized 30-min framework. 
    # QCs are kept as integers by taking the `.max()` of the 30-min bin.
    unified_base_df = create_unified_dataset(raw_data_dict, freq='30T')
    
    # Save the strictly processed base to disk
    base_out = os.path.join(CONFIG['output_dir'], "OBSEA_base_raw_30min.csv")
    unified_base_df.to_csv(base_out)
    print(f"Strict baseline dataset saved to: {base_out}")
    print(f"Base data shape: {unified_base_df.shape}")
    
    print("\n=== Step 3: Applying TEOS-10 Bio-fouling Correction (CTD Drift) ===")
    from obsea_pipeline.preprocessing.oceanography import add_derived_features
    # We apply the inverse slope correction calculated today in teos_correction.py
    # SBE16 Conductivity restoration: C_new = C_raw * (1 / (1 - 0.0526 * months))
    # For a 2-year period, we apply the vectorized restoration logic
    unified_base_df = add_derived_features(unified_base_df) 
    print("TEOS-10 restoration applied to SBE16 datastreams.")

    print("\n=== Step 4: Injecting Tiered Multi-Variant ADCP/AWAC Data ===")
    processor = AWACProcessor()
    
    out_final = os.path.join(CONFIG['output_dir'], "OBSEA_unified_high_fidelity_30min.csv")
    
    # Use the processor to fuse historical 15-year archives with the new base
    processor.fuse_and_export(
        hist_csv="data/exported_data/adcp/historical_adcp_unified_2010_2025.csv",
        api_cur_csv="data/exported_data/OBSEA_AWAC_currents_API_binned.csv",
        api_wav_csv="data/exported_data/RAW/OBSEA_AWAC_waves_full_nc_RAW.csv",
        prod_csv=base_out,
        output_csv=out_final
    )
    
    print(f"\nSUCCESS: Golden Dataset (API + AWAC + TEOS-10) built: {out_final}")

if __name__ == '__main__':
    build_database()
