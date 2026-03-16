import argparse
import pandas as pd
from pathlib import Path

from obsea_pipeline.config.settings import CONFIG, GAP_CATEGORIES
from obsea_pipeline.ingestion.sta_connector import STAConnector
from obsea_pipeline.ingestion.csv_loader import load_all_data
from obsea_pipeline.qc.checks import apply_instrumental_qc
from obsea_pipeline.preprocessing.oceanography import add_derived_features
from obsea_pipeline.preprocessing.resampling import resample_dataframe
from obsea_pipeline.gaps.analysis import detect_gaps
from plot_all_instruments import plot_instrument_timeseries
from obsea_pipeline.models.selector import selective_interpolation
from obsea_pipeline.benchmark.runner import benchmark_gap_filling
from obsea_pipeline.utils.logger import setup_logger

logger = setup_logger(name="obsea_main")

def run_pipeline(mode="production", limit_days=None, start_date=None, end_date=None, use_cache=True):
    logger.info(f"Starting OBSEA Pipeline V2 in {mode.upper()} mode")
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "obsea_preprocessed_cache.parquet"
    
    # =========================================================================
    # OPTIMAL CACHING MECHANISM
    # =========================================================================
    if use_cache and cache_file.exists():
        logger.info(f"Loading preprocessed dataset from local cache: {cache_file}")
        df_resampled = pd.read_parquet(cache_file)
        logger.info(f"Cache loaded successfully. Shape: {df_resampled.shape}")
        
    else:
        # 1. Ingestion
        logger.info("Initializing STA v1.1 API Connector...")
        sta = STAConnector()
        try:
            from datetime import datetime, timezone, timedelta
            
            # Ventana temporal de consulta. Para producción, se calculará dinámicamente.
            if start_date and end_date:
                start_str = f"{start_date}T00:00:00Z"
                end_str = f"{end_date}T23:59:59Z"
            elif limit_days:
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=limit_days)
                start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                # Default a un bloque de prueba si no se especifica explícitamente nada para no tumbar la API
                start_str = "2023-05-01T00:00:00Z"
                end_str = "2023-05-15T23:59:59Z"
        
            # Iterar sobre los 5 grupos de instrumentos seleccionados
            dfs = []
            for group_name, var_dict in STAConnector.INSTRUMENT_GROUPS.items():
                logger.info(f"  Fetching instrument group: {group_name} ({len(var_dict)} variables)...")
                
                # Determinar si es un grupo AWAC con filtro de profundidad
                depth_bin = STAConnector.AWAC_DEPTH_BINS.get(group_name, None)
                if depth_bin is not None:
                    logger.info(f"    → ADCP profile mode: selecting depth bin = {depth_bin}m")
                
                for var_name, ds_id in var_dict.items():
                    try:
                        df_var = sta.fetch_observations(ds_id, start_time=start_str, end_time=end_str, depth_bin=depth_bin)
                        if not df_var.empty:
                            df_var.rename(columns={'Value': var_name}, inplace=True)
                            dfs.append(df_var)
                            logger.info(f"    ✓ {var_name} (DS:{ds_id}): {len(df_var)} records")
                        else:
                            logger.warning(f"    ✗ {var_name} (DS:{ds_id}): 0 records")
                    except Exception as e:
                        logger.warning(f"    ✗ {var_name} (DS:{ds_id}) failed: {e}")
                        
            if dfs:
                df_raw = pd.concat(dfs, axis=1)
                df_raw.sort_index(inplace=True)
                logger.info(f"STA API Integration Successful. Unified DataFrame: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
            else:
                logger.warning("STA API Integration yielded 0 records across all instruments.")
                df_raw = pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"STA API Master Fetch failed ({e}). Falling back to CSV Loader...")
            df_raw = pd.DataFrame()
        
        if df_raw.empty:
            data_dict = load_all_data()
            if not data_dict:
                logger.error("No data available from API or CSV. Exiting.")
                return None
            # Use primary instrument data generically
            df_raw = list(data_dict.values())[0]
            if limit_days is not None:
                logger.info(f"Clipping target dataset to the last {limit_days} days for rapid testing...")
                cutoff_date = df_raw.index.max() - pd.Timedelta(days=limit_days)
                df_raw = df_raw[df_raw.index >= cutoff_date]
        
        # 2. QC & Preprocessing
        # Sanitize: la API STA puede devolver None en lugar de NaN para valores faltantes
        for col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
        logger.info("Applying QARTOD Quality Control...")
        df_qc = apply_instrumental_qc(df_raw)

        logger.info("Calculating Derived Oceanographic Variables...")
        df_ocean = add_derived_features(df_qc)
        
        logger.info("Resampling to Strict 30-min Grid...")
        df_resampled = resample_dataframe(df_ocean, freq='30min')
        
        # Save cache to bypass this entire block next time
        if use_cache:
            logger.info(f"Saving preprocessed dataset to cache: {cache_file}...")
            df_resampled.to_parquet(cache_file)
    # 3. Gap Analysis
    logger.info("Analyzing Multi-scale Data Gaps...")
    if 'TEMP' in df_resampled.columns:
        mask, gaps = detect_gaps(df_resampled['TEMP'])
    
    # 4. Branching logic based on mode
    
    if mode == "ingest":
        logger.info("Ingest mode complete. Exiting.")
        output_path = output_dir / "ingested_data.csv"
        df_resampled.to_csv(output_path)
        return df_resampled
        
    elif mode == "benchmark":
        logger.info("Executing Artificial Benchmark Simulation across 14 Native Deep Learning Models...")
        results = benchmark_gap_filling(df_resampled, test_variable='TEMP', gap_categories=list(GAP_CATEGORIES.keys()))
        logger.info("Benchmark complete.")
        return results
        
    elif mode == "production":
        logger.info("Running Scale-Aware Autonomous Interpolation Engine...")
        df_filled = selective_interpolation(df_resampled)
        
        output_path = output_dir / "OBSEA_final_interpolated.csv"
        df_filled.to_csv(output_path)
        logger.info(f"Pipeline complete. Production dataset exported to {output_path}")
        return df_filled
        
    elif mode == "plot":
        logger.info("Generating multi-panel gap classification plots...")
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        for group_name, var_list in CONFIG['variables'].items():
            available_vars = [v for v in var_list if v in df_resampled.columns]
            if available_vars:
                logger.info(f"  Plotting {group_name}...")
                f = plot_instrument_timeseries(df_resampled, group_name, available_vars, plot_dir)
                generated_files.append(f)
        
        logger.info(f"Generated {len(generated_files)} plots in {plot_dir}")
        return df_resampled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBSEA Standalone Pipeline V2")
    parser.add_argument("--mode", type=str, choices=["ingest", "benchmark", "production", "plot"], 
                        default="production", help="Execution mode to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit history days for faster debugging (e.g. 30)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true", help="Force fetching from API instead of loading the cache")
    args = parser.parse_args()
    
    try:
        run_pipeline(mode=args.mode, limit_days=args.limit, 
                     start_date=args.start, end_date=args.end, 
                     use_cache=not args.no_cache)
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user.")
