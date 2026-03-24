import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot_all_instruments import plot_instrument_timeseries
from obsea_pipeline.config.settings import CONFIG

# Config
POST_FILE = '/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts/obsea_production_deploy/results_jupyterhub/OBSEA_final_interpolated.csv'
OUTPUT_DIR = Path('output_lup/plots_production_groups')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_production_plots():
    print(f"Loading final interpolated data from {POST_FILE}...")
    df = pd.read_csv(POST_FILE, index_col=0, parse_dates=True)
    print(f"Data loaded. Shape: {df.shape}")

    generated_files = []
    
    # Iterate over instrument groups defined in settings
    for group_name, var_list in CONFIG['variables'].items():
        available_vars = [v for v in var_list if v in df.columns]
        
        if available_vars:
            print(f"Plotting group: {group_name} ({len(available_vars)} variables)...")
            try:
                # Reuse the high-quality plotting logic from plot_all_instruments
                f = plot_instrument_timeseries(df, group_name, available_vars, OUTPUT_DIR)
                generated_files.append(f)
            except Exception as e:
                print(f"Error plotting {group_name}: {e}")
        else:
            print(f"No variables found for group {group_name} in the CSV.")

    print(f"\nDone! Generated {len(generated_files)} group plots in {OUTPUT_DIR}/")

if __name__ == "__main__":
    generate_production_plots()
