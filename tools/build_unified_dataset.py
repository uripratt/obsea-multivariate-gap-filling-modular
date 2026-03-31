#!/usr/bin/env python3
"""
tools/build_unified_dataset.py
──────────────────────────────
Builds the OBSEA_multivariate_unified_30min.csv from three layered sources,
in strict priority order (ERDDAP NetCDF → AWAC binary archive → STA API tail):

    Layer 1: data/erdap_obsea/**  (2009-2024, highest density QC'd data)
    Layer 2: data/exported_data/adcp/historical_adcp_unified_2010_2025.csv
             (fills minor AWAC current gaps from binary .gz source)
    Layer 3: STA API telemetry (fills 2024-2025 tail and any gaps)

Output: data/exported_data/OBSEA_multivariate_unified_30min.csv
"""

import os
import sys
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timezone

# ── Setup ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("obsea.unified_builder")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERDDAP_DIR = os.path.join(BASE_DIR, "data", "erdap_obsea")
OUT_DIR    = os.path.join(BASE_DIR, "data", "exported_data")
OUT_CSV    = os.path.join(OUT_DIR,  "OBSEA_multivariate_unified_30min.csv")

# AWAC depth bin indices (0-indexed; bins are 0,1,2,...,19 metres)
AWAC_BIN_2M  = 2   # index 2 → 2.0m
AWAC_BIN_18M = 18  # index 18 → 18.0m

# ERDDAP TIME units: seconds since 1970-01-01
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 HELPERS: NetCDF readers
# ══════════════════════════════════════════════════════════════════════════════

def _epoch_to_dt(seconds_arr):
    """Convert seconds-since-epoch to naive UTC DatetimeIndex."""
    return pd.to_datetime(seconds_arr, unit='s', utc=True).tz_localize(None)


def _extract_scalar_var(ds, var_name):
    """Extract a scalar (non-depth) 5D variable: (lat, lon, depth, sensor, time)."""
    v = ds.variables[var_name]
    arr = v[:].squeeze()
    # Flatten to 1D if needed, but ensure length matches TIME dimension
    if arr.ndim > 1:
        arr = arr.flatten()
    
    # Check if length matches TIME
    target_len = ds.variables['TIME'].shape[0]
    if arr.shape[0] != target_len:
        # If it's a constant or tiny array, repeat or padding won't help much, 
        # so we return a nan array to avoid the length mismatch crash.
        return np.full(target_len, np.nan)
        
    arr = np.ma.filled(arr, np.nan).astype(float)
    return arr


def _nc_files(folder):
    """Glob all .nc files inside a folder tree, sorted by filename."""
    return sorted(glob.glob(os.path.join(folder, "**", "*.nc"), recursive=True))


# ── CTD ────────────────────────────────────────────────────────────────────────
def load_erddap_ctd(erddap_dir):
    logger.info("Loading ERDDAP CTD NetCDF files...")
    ctd_folder = os.path.join(erddap_dir, "OBSEA_CTD_30min_nc")
    frames = []
    for fpath in _nc_files(ctd_folder):
        try:
            ds = nc.Dataset(fpath)
            times = _epoch_to_dt(ds.variables['TIME'][:])
            df = pd.DataFrame(index=times)
            
            # Check SENSOR_ID to see if calibration is needed
            # (In ERDDAP, SBE37-SN47472 often needs the +0.52 dbar and +1.76 PSU offset)
            is_sbe37 = False
            if 'SENSOR_ID' in ds.variables:
                sid = str(ds.variables['SENSOR_ID'][:].squeeze())
                if '47472' in sid or 'SBE37' in sid.upper():
                    is_sbe37 = True
            
            for var in ['TEMP', 'PRES', 'CNDC', 'PSAL', 'SVEL']:
                if var in ds.variables:
                    val = _extract_scalar_var(ds, var)
                    df[var] = val
            ds.close()
            # Filter NoData from NetCDF regardless of flags
            df.replace([-999.0, -999.9, -999.99, 99.9, 99.99], np.nan, inplace=True)
            frames.append(df)
        except Exception as e:
            logger.warning(f"  Skipping CTD file {os.path.basename(fpath)}: {e}")
    if not frames:
        logger.warning("  No CTD NetCDF files loaded.")
        return pd.DataFrame()
    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    logger.info(f"  CTD ERDDAP: {len(result):,} rows | {result['TEMP'].count():,} TEMP obs")
    return result


# ── AWAC Currents ──────────────────────────────────────────────────────────────
def load_erddap_awac_currents(erddap_dir):
    logger.info("Loading ERDDAP AWAC Currents NetCDF files...")
    folder = os.path.join(erddap_dir, "OBSEA_AWAC_currents_30min_nc")
    frames_2m, frames_18m = [], []

    for fpath in _nc_files(folder):
        try:
            ds = nc.Dataset(fpath)
            times = _epoch_to_dt(ds.variables['TIME'][:])
            depth_bins = ds.variables['DEPTH'][:]
            
            for bin_idx, prefix, frames in [(AWAC_BIN_2M, 'AWAC2M', frames_2m),
                                             (AWAC_BIN_18M, 'AWAC18M', frames_18m)]:
                if bin_idx >= len(depth_bins):
                    continue
                df = pd.DataFrame(index=times)
                for var in ['CSPD', 'CDIR', 'UCUR', 'VCUR', 'ZCUR']:
                    if var in ds.variables:
                        arr = ds.variables[var][:].squeeze()  # (depth, time) after squeeze
                        # After squeezing a (1,1,20,1,T) → (20,T) or (T,) if DEPTH=1
                        if arr.ndim == 2:
                            col = np.ma.filled(arr[bin_idx, :], np.nan).astype(float)
                        else:
                            col = np.ma.filled(arr, np.nan).astype(float)
                        df[f'{prefix}_{var}'] = col
                frames.append(df)
            ds.close()
        except Exception as e:
            logger.warning(f"  Skipping AWAC currents file {os.path.basename(fpath)}: {e}")

    def concat_frames(frames, label):
        if not frames:
            return pd.DataFrame()
        r = pd.concat(frames).sort_index()
        r = r[~r.index.duplicated(keep='first')]
        # Filter NoData
        r.replace([-999.0, -999.9, -999.99, 99.9, 99.99], np.nan, inplace=True)
        logger.info(f"  {label} ERDDAP: {len(r):,} rows")
        return r

    df2m  = concat_frames(frames_2m,  "AWAC 2M")
    df18m = concat_frames(frames_18m, "AWAC 18M")
    return df2m.join(df18m, how='outer') if not df2m.empty else df18m


# ── AWAC Waves ─────────────────────────────────────────────────────────────────
def load_erddap_awac_waves(erddap_dir):
    logger.info("Loading ERDDAP AWAC Waves NetCDF files...")
    folder = os.path.join(erddap_dir, "OBSEA_AWAC_waves_full_nc")
    frames = []
    for fpath in _nc_files(folder):
        try:
            ds = nc.Dataset(fpath)
            times = _epoch_to_dt(ds.variables['TIME'][:])
            df = pd.DataFrame(index=times)
            for var in ['VHM0', 'VTPK', 'VMDR']:
                if var in ds.variables:
                    df[f'AWAC_{var}'] = _extract_scalar_var(ds, var)
            ds.close()
            frames.append(df)
        except Exception as e:
            logger.warning(f"  Skipping AWAC waves file {os.path.basename(fpath)}: {e}")
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    # Filter NoData
    result.replace([-999.0, -999.9, -999.99, 99.9, 99.99], np.nan, inplace=True)
    logger.info(f"  AWAC Waves ERDDAP: {len(result):,} rows | {result['AWAC_VHM0'].count():,} VHM0 obs")
    return result


# ── Airmar Buoy ────────────────────────────────────────────────────────────────
def load_erddap_airmar(erddap_dir):
    logger.info("Loading ERDDAP Airmar Buoy NetCDF files (old 150WX + new 200WX)...")
    folder = os.path.join(erddap_dir, "OBSEA_Airmar_30min_nc")
    frames = []
    for fpath in _nc_files(folder):
        try:
            ds = nc.Dataset(fpath)
            times = _epoch_to_dt(ds.variables['TIME'][:])
            df = pd.DataFrame(index=times)
            for var in ['CAPH', 'AIRT', 'WDIR', 'WSPD']:
                if var in ds.variables:
                    df[f'BUOY_{var}'] = _extract_scalar_var(ds, var)
            ds.close()
            frames.append(df)
        except Exception as e:
            logger.warning(f"  Skipping Airmar file {os.path.basename(fpath)}: {e}")
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    # Filter NoData
    result.replace([-999.0, -999.9, -999.99, 99.9, 99.99], np.nan, inplace=True)
    logger.info(f"  Airmar Buoy ERDDAP: {len(result):,} rows | {result['BUOY_AIRT'].count():,} AIRT obs")
    return result


# ── CTVG Land Station ──────────────────────────────────────────────────────────
def load_erddap_ctvg(erddap_dir):
    logger.info("Loading ERDDAP CTVG Vantage Pro 2 NetCDF files...")
    folder = os.path.join(erddap_dir, "OBSEA_CTVG_Vantage_Pro2_30min_nc")
    frames = []
    for fpath in _nc_files(folder):
        try:
            ds = nc.Dataset(fpath)
            times = _epoch_to_dt(ds.variables['TIME'][:])
            df = pd.DataFrame(index=times)
            for var in ['CAPH', 'RELH', 'AIRT', 'WDIR', 'WSPD']:
                if var in ds.variables:
                    arr = _extract_scalar_var(ds, var)
                    col = f'CTVG_{var}'
                    df[col] = arr
            ds.close()
            # Convert CTVG WSPD from km/h to m/s (confirmed from NetCDF units)
            if 'CTVG_WSPD' in df.columns:
                df['CTVG_WSPD'] = df['CTVG_WSPD'] / 3.6
            frames.append(df)
        except Exception as e:
            logger.warning(f"  Skipping CTVG file {os.path.basename(fpath)}: {e}")
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    # Filter NoData
    result.replace([-999.0, -999.9, -999.99, 99.9, 99.99], np.nan, inplace=True)
    logger.info(f"  CTVG ERDDAP: {len(result):,} rows | {result['CTVG_AIRT'].count():,} AIRT obs")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: AWAC Binary Archive CSV overlay
# ══════════════════════════════════════════════════════════════════════════════

def load_awac_binary_archive(base_dir):
    """Load the preprocessed AWAC binary archive and return a standard DataFrame."""
    csv_path = os.path.join(base_dir, "data", "exported_data", "adcp",
                             "historical_adcp_unified_2010_2025.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"  AWAC binary archive not found: {csv_path}")
        return pd.DataFrame()

    logger.info(f"Loading AWAC binary archive: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
    df.set_index('Timestamp', inplace=True)

    # Map columns to our naming convention
    rename = {
        'AWAC2M_CSPD':  'AWAC2M_CSPD',  'AWAC2M_CDIR':  'AWAC2M_CDIR',
        'AWAC2M_UCUR':  'AWAC2M_UCUR',  'AWAC2M_VCUR':  'AWAC2M_VCUR',
        'AWAC2M_ZCUR':  'AWAC2M_ZCUR',
        'AWAC18M_CSPD': 'AWAC18M_CSPD', 'AWAC18M_CDIR': 'AWAC18M_CDIR',
        'AWAC18M_UCUR': 'AWAC18M_UCUR', 'AWAC18M_VCUR': 'AWAC18M_VCUR',
        'AWAC18M_ZCUR': 'AWAC18M_ZCUR',
        'AWAC_Hm0':     'AWAC_VHM0',    'AWAC_Tp':      'AWAC_VTPK',
        'AWAC_WDIR':    'AWAC_VMDR',
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
    df.replace([-999.0, -999.99, 99.99], np.nan, inplace=True)
    logger.info(f"  AWAC binary: {len(df):,} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3: STA API tail-fill (2024-2025 gaps)
# ══════════════════════════════════════════════════════════════════════════════

def load_sta_tail(base_dir, start_date, end_date):
    """Fetch STA API data for the tail period that ERDDAP doesn't cover."""
    logger.info(f"Fetching STA API tail ({start_date} → {end_date})...")
    sys.path.insert(0, base_dir)
    try:
        from obsea_pipeline.ingestion.sta_connector import STAConnector
        sta16 = STAConnector(ctd_type='sbe16')
        sta37 = STAConnector(ctd_type='sbe37')
        
        dfs = {}
        # CTD SBE16
        for var, ds_id in sta16.DATASTREAM_CTD_SBE16.items():
            df_v = sta16.fetch_observations(ds_id, start_time=f"{start_date}T00:00:00Z",
                                             end_time=f"{end_date}T23:59:59Z")
            if not df_v.empty:
                dfs[var] = df_v['Value']
        # CTD SBE37
        for var, ds_id in sta37.DATASTREAM_CTD_SBE37.items():
            df_v = sta37.fetch_observations(ds_id, start_time=f"{start_date}T00:00:00Z",
                                             end_time=f"{end_date}T23:59:59Z")
            if not df_v.empty and var not in dfs:
                dfs[f'SBE37_{var}'] = df_v['Value']
        # CTVG
        for var, ds_id in sta16.DATASTREAM_CTVG_METEO.items():
            df_v = sta16.fetch_observations(ds_id, start_time=f"{start_date}T00:00:00Z",
                                             end_time=f"{end_date}T23:59:59Z")
            if not df_v.empty:
                dfs[var] = df_v['Value']
        # Buoy (new Airmar 200WX)
        for var, ds_id in sta16.DATASTREAM_BUOY_METEO.items():
            df_v = sta16.fetch_observations(ds_id, start_time=f"{start_date}T00:00:00Z",
                                             end_time=f"{end_date}T23:59:59Z")
            if not df_v.empty:
                dfs[var] = df_v['Value']
        
        if dfs:
            result = pd.DataFrame(dfs)
            result.index = result.index.tz_localize(None) if result.index.tz else result.index
            # Filter NoData from API
            result.replace([-999.0, -999.9, -999.99, 99.9, 99.99], np.nan, inplace=True)
            logger.info(f"  STA API tail: {len(result):,} rows fetched")
            return result
    except Exception as e:
        logger.error(f"  STA tail-fill failed: {e}")
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BUILD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_unified_dataset(start="2009-01-01", end="2025-12-31",
                           skip_api=False, output_path=OUT_CSV):

    print("\n" + "="*70)
    print(f"  OBSEA Unified Dataset Builder ({start} → {end})")
    print("="*70)

    # ┌─ LAYER 1: ERDDAP NetCDF ─────────────────────────────────────────────
    print("\n[1/5] Loading ERDDAP NetCDF archives (2009-2024)...")
    df_ctd    = load_erddap_ctd(ERDDAP_DIR)
    df_awac_c = load_erddap_awac_currents(ERDDAP_DIR)
    df_awac_w = load_erddap_awac_waves(ERDDAP_DIR)
    df_airmar = load_erddap_airmar(ERDDAP_DIR)
    df_ctvg   = load_erddap_ctvg(ERDDAP_DIR)

    # Join all ERDDAP sources into one master frame
    print("\n[2/5] Merging ERDDAP sources into master frame...")
    master = df_ctd
    for df_part in [df_awac_c, df_awac_w, df_airmar, df_ctvg]:
        if not df_part.empty:
            master = master.join(df_part, how='outer') if not master.empty else df_part
    master.sort_index(inplace=True)
    logger.info(f"Master frame after ERDDAP merge: {master.shape}")

    # ┌─ LAYER 2: AWAC Binary Archive overlay ──────────────────────────────
    print("\n[3/5] Overlaying AWAC binary archive CSV...")
    df_awac_bin = load_awac_binary_archive(BASE_DIR)
    if not df_awac_bin.empty:
        awac_cols = [c for c in df_awac_bin.columns
                     if c.startswith('AWAC') and c in master.columns]
        for col in awac_cols:
            gaps_before = master[col].isna().sum()
            master[col] = master[col].combine_first(df_awac_bin[col])
            filled = gaps_before - master[col].isna().sum()
            if filled > 0:
                logger.info(f"  {col}: filled {filled:,} gaps from binary archive")
        # Add any columns from archive not yet in master
        new_cols = [c for c in df_awac_bin.columns if c not in master.columns]
        if new_cols:
            master = master.join(df_awac_bin[new_cols], how='outer')

    # ┌─ LAYER 3: STA API tail-fill ─────────────────────────────────────────
    if not skip_api:
        # Only fetch from the last ERDDAP date to the user-specified end
        erddap_last = "2024-12-31"
        if end > erddap_last:
            print(f"\n[4/5] STA API tail-fill ({erddap_last} → {end})...")
            df_tail = load_sta_tail(BASE_DIR, erddap_last, end)
            if not df_tail.empty:
                df_tail.index = df_tail.index.tz_localize(None) \
                    if hasattr(df_tail.index, 'tz') and df_tail.index.tz else df_tail.index
                # Map SBE37 columns to fill CTD gaps
                for var in ['TEMP', 'PRES', 'CNDC', 'PSAL', 'SVEL']:
                    s37_col = f'SBE37_{var}'
                    if s37_col in df_tail.columns:
                        # Apply scientific offsets to SBE37 variant before merging
                        if var == 'PRES': df_tail[s37_col] += 0.52
                        if var == 'PSAL': df_tail[s37_col] += 1.76
                        
                        if var in master.columns:
                            df_tail[var] = df_tail.get(var, pd.Series(dtype=float)).combine_first(df_tail[s37_col])
                        else:
                            df_tail[var] = df_tail[s37_col]
                        df_tail.drop(columns=[s37_col], inplace=True, errors='ignore')
                master = master.combine_first(df_tail)
    else:
        print("\n[4/5] Skipping STA API tail-fill (--skip-api flag set)")

    # ┌─ Clip to requested date range ──────────────────────────────────────
    master = master.loc[start:end]

    # ┌─ STEP 5: Resample to strict 30-min grid ─────────────────────────────
    print("\n[5/5] Resampling to strict 30-min grid and exporting...")
    master = master.resample('30min').mean()
    master.index.name = 'Timestamp'

    # ── Data Quality Report
    print(f"\n{'='*70}")
    print(f"  OBSEA Unified Dataset — Pre-Export Audit")
    print(f"  Range: {master.index.min()} → {master.index.max()}")
    print(f"  Total 30-min slots: {len(master):,}")
    print(f"{'='*70}")
    print(f"  {'Variable':<22} {'Observed':>10} {'% Missing':>10}")
    print(f"  {'-'*45}")
    for col in [c for c in master.columns if not c.endswith('_QC')]:
        obs = master[col].count()
        pct = (1 - obs / len(master)) * 100
        flag = "✓" if pct < 20 else ("⚠" if pct < 60 else "✗")
        print(f"  {flag} {col:<20} {obs:>10,} {pct:>9.1f}%")

    # ── Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master.to_csv(output_path)
    parquet_path = output_path.replace('.csv', '.parquet')
    master.to_parquet(parquet_path)
    print(f"\n  ✓ CSV exported:     {output_path}")
    print(f"  ✓ Parquet exported: {parquet_path}")
    print("="*70 + "\n")
    return master


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build OBSEA Unified Multivariate Dataset")
    parser.add_argument("--start",     default="2009-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",       default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--skip-api",  action="store_true",  help="Skip STA API tail-fill")
    parser.add_argument("--output",    default=OUT_CSV,      help="Output CSV path")
    args = parser.parse_args()

    build_unified_dataset(
        start=args.start,
        end=args.end,
        skip_api=args.skip_api,
        output_path=args.output,
    )
