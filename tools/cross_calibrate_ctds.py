import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%m/%d/%y %H:%M:%S')
logger = logging.getLogger(__name__)

# Import our STA connector 
sys.path.append("/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts")
from obsea_pipeline.ingestion.sta_connector import STAConnector

def fetch_ctd(start_date, end_date, ctd_type):
    """Fetches highly specific CTD metrics directly strictly for the calibration validation."""
    logger.info(f"Extracting {ctd_type.upper()} from {start_date} to {end_date}...")
    sta = STAConnector(ctd_type=ctd_type)
    
    # We ONLY want the CTD group, nothing else to save bandwidth.
    var_dict = sta.INSTRUMENT_GROUPS['CTD']
    
    # We must fetch each variable individually and join them by phenomenonTime
    df_ctd = pd.DataFrame()
    for var_name, ds_id in var_dict.items():
        df_obs = sta.fetch_observations(ds_id, start_time=f"{start_date}T00:00:00Z", end_time=f"{end_date}T23:59:59Z")
        if df_obs is not None and not df_obs.empty:
            # rename Value to the variable name
            df_obs = df_obs[['Value']].rename(columns={'Value': var_name})
            
            if df_ctd.empty:
                df_ctd = df_obs
            else:
                # Merge on index
                df_ctd = df_ctd.join(df_obs, how='outer')
                
    if df_ctd.empty:
        logger.warning(f"No data returned for {ctd_type} in this window!")
    else:
        logger.info(f"Extracted {len(df_ctd)} rows for {ctd_type}.")
        
    return df_ctd

def run_calibration(start_date, end_date, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch both datasets sequentially
    df_16 = fetch_ctd(start_date, end_date, 'sbe16')
    df_37 = fetch_ctd(start_date, end_date, 'sbe37')
    
    if df_16.empty or df_37.empty:
        logger.error("Missing data from one or both CTDs in this timeframe. Try a different year.")
        return
        
    # 2. Add Prefixes for differentiation and Merge
    df_16 = df_16.add_prefix('SBE16_')
    df_37 = df_37.add_prefix('SBE37_')
    
    # Inner join on Timestamp index to ONLY compare exact synchronous samples
    df_sync = df_16.join(df_37, how='inner').dropna()
    logger.info(f"Identified {len(df_sync)} synchronous overlapping measurements (30-min precision).")
    
    if len(df_sync) < 10:
        logger.error("Insufficient synchronous overlap to perform a robust statistical calibration.")
        return
        
    # 3. Statistical Discrepancy Analysis
    variables = ['TEMP', 'PSAL', 'PRES', 'CNDC']
    report = []
    
    for var in variables:
        col16 = f"SBE16_{var}"
        col37 = f"SBE37_{var}"
        
        if col16 not in df_sync.columns or col37 not in df_sync.columns:
            continue
            
        mae = np.mean(np.abs(df_sync[col16] - df_sync[col37]))
        rmse = np.sqrt(np.mean((df_sync[col16] - df_sync[col37])**2))
        corr = df_sync[col16].corr(df_sync[col37])
        bias = np.mean(df_sync[col16] - df_sync[col37])  # Positive means SBE16 reads higher
        
        report.append({
            'Variable': var,
            'Sync_Samples': len(df_sync),
            'Pearson_Corr': round(corr, 5),
            'RMSE': round(rmse, 4),
            'Mean_Absolute_Error': round(mae, 4),
            'Mean_Bias (16-37)': round(bias, 4)
        })
        
        # 4. Generate Scatter Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df_sync[col16], y=df_sync[col37], alpha=0.5, color='royalblue')
        
        # Plot perfect agreement line
        min_val = min(df_sync[col16].min(), df_sync[col37].min())
        max_val = max(df_sync[col16].max(), df_sync[col37].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Agreement (1:1)')
        
        plt.title(f"Instrument Cross-Calibration: {var}\nSBE16 vs SBE37 ({start_date} to {end_date})")
        plt.xlabel(f"SBE16 {var} (Primary)")
        plt.ylabel(f"SBE37 {var} (Backup)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"calibration_scatter_{var}.png", dpi=150)
        plt.close()
        
        # 4b. Generate Time-Series Plot
        plt.figure(figsize=(12, 5))
        plt.plot(df_sync.index, df_sync[col16], label=f'SBE16 {var}', color='royalblue', alpha=0.8, lw=1.5)
        plt.plot(df_sync.index, df_sync[col37], label=f'SBE37 {var}', color='darkorange', alpha=0.8, lw=1.5)
        plt.title(f"Time-Series Overlay: {var} (SBE16 vs SBE37)")
        plt.xlabel("Date (2012)")
        plt.ylabel(f"{var}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"calibration_timeseries_{var}.png", dpi=150)
        plt.close()
        
    df_report = pd.DataFrame(report)
    df_report.to_csv(out_dir / "ctd_calibration_stats.csv", index=False)
    
    logger.info("==================================================")
    logger.info("  CTD STATISTICAL CALIBRATION REPORT              ")
    logger.info("==================================================")
    for _, row in df_report.iterrows():
        logger.info(f"[{row['Variable']}] Corr: {row['Pearson_Corr']} | Bias: {row['Mean_Bias (16-37)']} | MAE: {row['Mean_Absolute_Error']}")
    logger.info("==================================================")
    logger.info(f"Plots and CSV exported to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTD Cross-Calibration Tool")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2018-06-30", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--out", type=str, default="output_lup/calibration", help="Output directory")
    args = parser.parse_args()
    
    run_calibration(args.start, args.end, args.out)
