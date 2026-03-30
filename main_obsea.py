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
from rich.console import Console
from rich.table import Table
from rich import print as rprint

logger = setup_logger(name="obsea_main")
console = Console()

def show_data_summary(df):
    """Muestra una tabla elegante con el estado actual del dataset."""
    table = Table(title="[bold blue]OBSEA Data Audit - Pre-Interpolation[/bold blue]", show_header=True, header_style="bold magenta")
    table.add_column("Instrument/Variable", style="cyan")
    table.add_column("Total Points", justify="right")
    table.add_column("Observed", justify="right", style="green")
    table.add_column("Missing (NaN)", justify="right", style="red")
    table.add_column("% Missing", justify="right")

    total_pts = len(df)
    for col in df.columns:
        if col == 'Timestamp': continue
        missing = df[col].isna().sum()
        observed = total_pts - missing
        pct = (missing / total_pts) * 100
        
        color = "red" if pct > 50 else "yellow" if pct > 10 else "green"
        table.add_row(
            col, 
            f"{total_pts:,}", 
            f"{observed:,}", 
            f"{missing:,}", 
            f"[{color}]{pct:.2f}%[/{color}]"
        )
    
    console.print(table)

def run_pipeline(mode="production", limit_days=None, start_date=None, end_date=None, use_cache=True, methods=None, extreme_mode=False, ctd_type="sbe16", csv_input_path=None):
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
        
        # Apply temporal filtering if specified by CLI, even on cache
        if start_date and end_date:
            logger.info(f"Slicing cached dataset to the requested window: {start_date} -> {end_date}")
            df_resampled = df_resampled[(df_resampled.index >= start_date) & (df_resampled.index <= end_date)]
            
        logger.info(f"Cache loaded successfully. Shape: {df_resampled.shape}")
        
    else:
        # 1. Ingestion
        if csv_input_path and Path(csv_input_path).exists():
            logger.info(f"Loading user-specified external fallback dataset: {csv_input_path}. Bypassing API...")
            df_raw = pd.read_csv(csv_input_path, index_col=0, parse_dates=True)
            
            if limit_days is not None and not df_raw.empty:
                logger.info(f"Clipping target dataset to the last {limit_days} days for rapid testing...")
                cutoff_date = df_raw.index.max() - pd.Timedelta(days=limit_days)
                df_raw = df_raw[df_raw.index >= cutoff_date]
                
        else:
            logger.info(f"Initializing STA v1.1 API Connector (Targeting CTD: {ctd_type.upper()})...")
            sta = STAConnector(ctd_type=ctd_type)
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
            
                # Iterar sobre los grupos de instrumentos seleccionados (SBE16 es primario, SBE37 es backup)
                dfs = []
                # Forzamos la descarga de ambos CTDs para la fusión con criterio
                sta.INSTRUMENT_GROUPS['CTD_SBE16'] = sta.DATASTREAM_CTD_SBE16
                sta.INSTRUMENT_GROUPS['CTD_SBE37'] = sta.DATASTREAM_CTD_SBE37
                
                for group_name, var_dict in sta.INSTRUMENT_GROUPS.items():
                    logger.info(f"  Fetching instrument group: {group_name} ({len(var_dict)} variables)...")
                    depth_bin = sta.AWAC_DEPTH_BINS.get(group_name, None)
                    
                    for var_name, ds_id in var_dict.items():
                        try:
                            df_var = sta.fetch_observations(ds_id, start_time=start_str, end_time=end_str, depth_bin=depth_bin)
                            if not df_var.empty:
                                # Usamos prefijos para CTD para poder fusionarlos con criterio después
                                if "CTD" in group_name:
                                    final_col = f"{group_name}_{var_name}"
                                else:
                                    final_col = var_name # Mantener nombres originales para AWAC y METEO (Buoy/CTVG ya tienen nombres distintos)
                                    
                                df_var.rename(columns={'Value': final_col}, inplace=True)
                                dfs.append(df_var)
                                logger.info(f"    ✓ {final_col} (DS:{ds_id}): {len(df_var)} records")
                            else:
                                logger.warning(f"    ✗ {var_name} (DS:{ds_id}): 0 records")
                        except Exception as e:
                            logger.warning(f"    ✗ {var_name} (DS:{ds_id}) failed: {e}")
                            
                if dfs:
                    df_raw = pd.concat(dfs, axis=1)
                    df_raw.sort_index(inplace=True)
                    
                    # --- FUSIÓN CIENTÍFICA DE CTDs (CRITERIO AUDITADO) ---
                    # Basado en ctd_calibration_report.md:
                    # TEMP: Correlación 0.9999 -> Fusión directa.
                    # PRES: Offset de 0.5214 dbar (SBE16 es más profundo).
                    # PSAL: Bias de -1.76 PSU en SBE37 (requiere compensación).
                    
                    logger.info("Applying Scientific Fusion Criteria for Dual-CTD (SBE16 + SBE37)...")
                    ctd_map = {
                        'TEMP': 0.0,      # Sin offset
                        'PRES': 0.5214,   # SBE37 + 0.52 = SBE16
                        'PSAL': 1.7612,   # SBE37 + 1.76 = SBE16
                        'CNDC': 0.187,    # SBE37 + 0.18 = SBE16
                        'SVEL': 0.0       # Sin offset reportado
                    }
                    
                    for var, offset in ctd_map.items():
                        s16 = f"CTD_SBE16_{var}"
                        s37 = f"CTD_SBE37_{var}"
                        
                        if s16 in df_raw.columns:
                            if s37 in df_raw.columns:
                                # Aplicar criterio científico: Corregir SBE37 antes de fusionar
                                s37_corrected = df_raw[s37] + offset
                                gaps_filled = df_raw[s16].isna().sum() - df_raw[s16].combine_first(s37_corrected).isna().sum()
                                logger.info(f"  ✓ {var}: Fused SBE16 + SBE37 (Offset: {offset}, Gaps filled: {gaps_filled})")
                                df_raw[var] = df_raw[s16].combine_first(s37_corrected)
                            else:
                                df_raw[var] = df_raw[s16]
                        elif s37 in df_raw.columns:
                            # Si solo hay SBE37, lo usamos con su corrección
                            df_raw[var] = df_raw[s37] + offset

                    # METEO: Se quedan separadas (BUOY_AIRT vs CTVG_AIRT) como pediste.
                    
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
        
        # Save cache to bypass this entire block next time (siempre se guarda tras una recarga exitosa)
        logger.info(f"Saving preprocessed dataset to cache: {cache_file}...")
        df_resampled.to_parquet(cache_file)
    
    # Audit Table (Rich)
    show_data_summary(df_resampled)
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
        logger.info("Executing Artificial Benchmark Simulation...")
        if extreme_mode:
            logger.info("  => FASTEN SEATBELTS: Extreme targeting mode enabled. AI will be tested exclusively against storms!")
        results = benchmark_gap_filling(df_resampled, test_variable='TEMP', gap_categories=list(GAP_CATEGORIES.keys()), methods=methods, extreme_mode=extreme_mode)
        logger.info("Benchmark complete.")
        return results
        
    elif mode == "production":
        logger.info("Running Scale-Aware Autonomous Interpolation Engine...")
        df_filled = selective_interpolation(df_resampled)
        
        output_path_csv = output_dir / "OBSEA_final_interpolated.csv"
        df_filled.to_csv(output_path_csv)
        
        # Export as Parquet to maintain precision and save space
        output_path_parquet = output_dir / "OBSEA_final_interpolated.parquet"
        df_filled.to_parquet(output_path_parquet)
        
        # Pre-compute Correlation Matrix for Webapp to prevent client-side freezing
        logger.info("Pre-computing Correlation Matrix for Webapp...")
        valid_cols = [c for c in df_filled.columns if not c.endswith('_QC') and not c.endswith('_STD') and pd.api.types.is_numeric_dtype(df_filled[c])]
        corr_matrix = df_filled[valid_cols].corr(method='pearson').round(4)
        
        json_corr = []
        for col in corr_matrix.columns:
            row = {'': col, 'Variable': col}
            for col2 in corr_matrix.columns:
                row[col2] = corr_matrix.at[col, col2] if pd.notna(corr_matrix.at[col, col2]) else 0.0
            json_corr.append(row)
            
        import json
        with open(output_dir / "correlation_matrix.json", 'w') as f:
            json.dump(json_corr, f)
            
        corr_matrix.to_csv(output_dir / "correlation_matrix.csv")
        
        logger.info(f"Pipeline complete. Production dataset and metadata exported to {output_dir}")
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
    parser.add_argument("--methods", nargs="+", default=None, help="List of specific models to benchmark (e.g. --methods linear time splines varma xgboost)")
    parser.add_argument("--extreme", action="store_true", help="Force benchmark gap generation to target top 5% extreme events/storms.")
    parser.add_argument("--ctd", type=str, choices=["sbe16", "sbe37"], default="sbe16", help="Select which CTD instrument datastreams to fetch from the OBSEA API.")
    parser.add_argument("--csv-input", type=str, default=None, help="Inject a static unified CSV file dataset to bypass ingestion APIs.")
    args = parser.parse_args()
    
    try:
        run_pipeline(mode=args.mode, limit_days=args.limit, 
                     start_date=args.start, end_date=args.end, 
                     use_cache=not args.no_cache, methods=args.methods,
                     extreme_mode=args.extreme, ctd_type=args.ctd, csv_input_path=args.csv_input)
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user.")
