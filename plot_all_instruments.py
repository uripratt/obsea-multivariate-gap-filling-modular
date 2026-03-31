"""
plot_all_instruments.py
=======================
Genera visualizaciones sofisticadas de series temporales con clasificación de gaps
para todos los instrumentos del observatorio OBSEA.

Estilo idéntico al usado en la Tesis doctoral (lup_data_obsea_analysis_jupyterhub.py),
con bandas de color superpuestas según la categoría del gap.

Uso:
    python plot_all_instruments.py                        # Últimos 30 días
    python plot_all_instruments.py --days 365             # Último año
    python plot_all_instruments.py --start 2023-01-01 --end 2023-12-31  # Rango custom
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from obsea_pipeline.ingestion.sta_connector import STAConnector
from obsea_pipeline.config.settings import GAP_CATEGORIES, CONFIG

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("plot_instruments")


# ==============================================================================
# GAP DETECTION (standalone, sin depender de analysis.py para este script)
# ==============================================================================
def detect_gaps_for_variable(series: pd.Series, freq_minutes: int = 30) -> pd.DataFrame:
    """
    Detecta gaps (regiones de NaN consecutivos) en una serie temporal.
    Devuelve un DataFrame con start, end, duration_hours, duration_points, category.
    """
    is_nan = series.isna()
    
    # Find gap boundaries
    diff = is_nan.astype(int).diff().fillna(0)
    gap_starts = series.index[diff == 1].tolist()
    gap_ends = series.index[diff == -1].tolist()
    
    # Handle edge cases
    if is_nan.iloc[0]:
        gap_starts.insert(0, series.index[0])
    if is_nan.iloc[-1]:
        gap_ends.append(series.index[-1])
    
    gaps = []
    for start, end in zip(gap_starts, gap_ends):
        mask = (series.index >= start) & (series.index < end)
        n_points = mask.sum()
        duration_hours = n_points * freq_minutes / 60.0
        
        # Classify gap using GAP_CATEGORIES
        category = 'gigant'
        for cat_name, cat_info in GAP_CATEGORIES.items():
            if duration_hours <= cat_info['max_hours']:
                category = cat_name
                break
        
        gaps.append({
            'variable': series.name,
            'start': start,
            'end': end,
            'duration_hours': duration_hours,
            'duration_points': n_points,
            'category': category,
        })
    
    return pd.DataFrame(gaps)


# ==============================================================================
# STYLING CONSTANTS
# ==============================================================================
GAP_COLORS = {
    'micro':    '#2ECC71',  # Verde – impacto mínimo
    'short':    '#F1C40F',  # Amarillo – gaps menores
    'medium':   '#E67E22',  # Naranja – gaps moderados
    'long':     '#E74C3C',  # Rojo – gaps significativos
    'extended': '#9B59B6',  # Púrpura – gaps mayores
    'gigant':   '#1C1C1C',  # Casi negro – gaps extremos
}

GAP_LABELS = {
    'micro':    '<1h',
    'short':    '1-6h',
    'medium':   '6h-3d',
    'long':     '3-30d',
    'extended': '30-60d',
    'gigant':   '>60d',
}

INSTRUMENT_TITLES = {
    'CTD':        'CTD SBE16 – Oceanographic Variables (20m depth)',
    'AWAC_2M':    'AWAC ADCP – Sea Currents at 2m (Surface Layer)',
    'AWAC_18M':   'AWAC ADCP – Sea Currents at 18m (Bottom Layer)',
    'AWAC_WAVES': 'AWAC ADCP – Wave Parameters (VHM0, VTPK, VMDR)',
    'BUOY_METEO': 'Airmar 200WX – Besos Buoy Meteorology (Offshore)',
    'CTVG_METEO': 'Vantage Pro2 – CTVG Land Station (4km Inland)',
}

LINE_COLORS = ['#2980B9', '#E67E22', '#27AE60', '#E74C3C', '#9B59B6',
               '#1ABC9C', '#D35400', '#34495E', '#F39C12', '#16A085']


# ==============================================================================
# MAIN PLOTTING ENGINE
# ==============================================================================
def plot_instrument_timeseries(df: pd.DataFrame, instrument_name: str,
                                variables: list, output_dir: Path):
    """
    Crea un plot multi-panel de alta calidad para un instrumento,
    con bandas de color clasificadas por duración del gap.
    """
    # Detect gaps for each variable
    all_gaps = []
    for var in variables:
        if var in df.columns:
            gaps = detect_gaps_for_variable(df[var])
            all_gaps.append(gaps)
    
    gaps_df = pd.concat(all_gaps, ignore_index=True) if all_gaps else pd.DataFrame()
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(22, 3.2 * n_vars), sharex=True)
    
    if n_vars == 1:
        axes = [axes]
    
    for i, (ax, var) in enumerate(zip(axes, variables)):
        if var not in df.columns:
            ax.text(0.5, 0.5, f'{var}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            continue
        
        # Get gaps for this specific variable
        var_gaps = gaps_df[gaps_df['variable'] == var] if not gaps_df.empty else pd.DataFrame()
        
        # Plot the time series (daily mean for visual clarity on long timescales)
        series = df[var].dropna()
        if len(series) == 0:
            continue
            
        # Decide resolution for plotting
        total_days = (df.index[-1] - df.index[0]).days
        if total_days > 365:
            plot_data = df[var].resample('D').mean()
            data_label = 'Data (daily mean)'
        elif total_days > 60:
            plot_data = df[var].resample('6h').mean()
            data_label = 'Data (6h mean)'
        else:
            plot_data = df[var]
            data_label = 'Data (30min)'
        
        # Plot with filled area for a "premium" look
        ax.fill_between(plot_data.index, plot_data.values, color=LINE_COLORS[i % len(LINE_COLORS)], alpha=0.1, zorder=1)
        ax.plot(plot_data.index, plot_data.values,
                lw=1.0, alpha=0.85, color=LINE_COLORS[i % len(LINE_COLORS)], zorder=3)
        
        # Overlay gap bands
        for _, gap in var_gaps.iterrows():
            cat = gap['category']
            if pd.notna(cat) and cat in GAP_COLORS:
                ax.axvspan(gap['start'], gap['end'],
                          color=GAP_COLORS[cat], alpha=0.15, zorder=2)
        
        # Clean label (remove prefix for readability)
        label = var
        for prefix in ['AWAC2M_', 'AWAC18M_', 'BUOY_', 'CTVG_']:
            label = label.replace(prefix, '')
        
        ax.set_ylabel(label, fontsize=12, fontweight='bold', color='#333')
        ax.grid(True, which='major', axis='both', alpha=0.3, zorder=0, linestyle='-')
        ax.grid(True, which='minor', axis='x', alpha=0.1, zorder=0, linestyle=':')
        ax.tick_params(axis='both', labelsize=10)
        
        # Statistics annotation box
        valid = df[var].dropna()
        missing_pct = (df[var].isna().sum() / len(df)) * 100
        n_gaps = len(var_gaps)
        
        if len(valid) > 0:
            stats_text = f"N={len(valid):,} | μ={valid.mean():.2f} | σ={valid.std():.2f}\nMissing: {missing_pct:.1f}% | Gaps: {n_gaps}"
        else:
            stats_text = f"Missing: {missing_pct:.1f}% | Gaps: {n_gaps}"
            
        ax.annotate(stats_text, xy=(0.99, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=9, color='#444', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='#ccc', alpha=0.9, lw=0.5))
    
    axes[-1].set_xlabel('Date', fontsize=11, fontweight='bold')
    
    # --- Build legend ---
    if not gaps_df.empty:
        gap_counts = gaps_df['category'].value_counts()
        total_gaps = len(gaps_df)
        
        legend_elements = []
        for cat in GAP_COLORS.keys():
            count = gap_counts.get(cat, 0)
            pct = (count / total_gaps * 100) if total_gaps > 0 else 0
            if count > 0:
                legend_elements.append(
                    Patch(facecolor=GAP_COLORS[cat], alpha=0.3,
                          edgecolor='#999', linewidth=0.5,
                          label=f"{cat} ({GAP_LABELS[cat]}): {count} ({pct:.1f}%)")
                )
        
        # Add data line to legend
        legend_elements.append(
            plt.Line2D([0], [0], color=LINE_COLORS[0], lw=1.2,
                      label=f'— Data')
        )
        
        fig.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=min(len(legend_elements), 4),
                  fontsize=9, framealpha=0.95,
                  edgecolor='#ccc')
    
    title = INSTRUMENT_TITLES.get(instrument_name, instrument_name)
    plt.suptitle(f'{title} — Time Series with Gap Classification',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    
    # Save
    output_file = output_dir / f"timeseries_gaps_{instrument_name.lower()}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  ✓ Saved: {output_file}")
    
    return output_file


# ==============================================================================
# DATA FETCHING & ORCHESTRATION
# ==============================================================================
def fetch_all_instruments(start_str: str, end_str: str) -> pd.DataFrame:
    """Descarga todas las variables de los 5 grupos de instrumentos."""
    sta = STAConnector()
    dfs = []
    
    for group_name, var_dict in STAConnector.INSTRUMENT_GROUPS.items():
        logger.info(f"  Fetching {group_name} ({len(var_dict)} variables)...")
        
        depth_bin = STAConnector.AWAC_DEPTH_BINS.get(group_name, None)
        if depth_bin is not None:
            logger.info(f"    → ADCP profile mode: depth bin = {depth_bin}m")
        
        for var_name, ds_id in var_dict.items():
            try:
                df_var = sta.fetch_observations(ds_id, start_time=start_str,
                                                 end_time=end_str, depth_bin=depth_bin)
                if not df_var.empty:
                    df_var.rename(columns={'Value': var_name}, inplace=True)
                    # Coerce to numeric (API can return None)
                    df_var[var_name] = pd.to_numeric(df_var[var_name], errors='coerce')
                    dfs.append(df_var)
                    logger.info(f"    ✓ {var_name} (DS:{ds_id}): {len(df_var)} records")
                else:
                    logger.warning(f"    ✗ {var_name} (DS:{ds_id}): 0 records")
            except Exception as e:
                logger.warning(f"    ✗ {var_name} (DS:{ds_id}) failed: {e}")
    
    if dfs:
        df = pd.concat(dfs, axis=1)
        df.sort_index(inplace=True)
        logger.info(f"Unified DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    else:
        logger.error("No data fetched!")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="OBSEA Multi-Instrument Time Series Visualizer")
    parser.add_argument("--days", type=int, default=None,
                       help="Fetch last N days from now (default: 30)")
    parser.add_argument("--start", type=str, default=None,
                       help="Start date ISO, e.g. 2023-01-01")
    parser.add_argument("--end", type=str, default=None,
                       help="End date ISO, e.g. 2023-12-31")
    parser.add_argument("--output", type=str, default="output_lup/plots",
                       help="Output directory for plots")
    args = parser.parse_args()
    
    # Determine time range
    if args.start and args.end:
        start_str = f"{args.start}T00:00:00Z"
        end_str = f"{args.end}T23:59:59Z"
    else:
        days = args.days or 30
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    logger.info(f"=== OBSEA Visualization Engine ===")
    logger.info(f"Time range: {start_str} → {end_str}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data
    logger.info("Phase 1: Fetching data from STA API...")
    df = fetch_all_instruments(start_str, end_str)
    
    if df.empty:
        logger.error("No data available. Exiting.")
        return
    
    # Generate plots for each instrument group
    logger.info("Phase 2: Generating visualizations...")
    generated_files = []
    
    for group_name, var_list in CONFIG['variables'].items():
        available_vars = [v for v in var_list if v in df.columns]
        if available_vars:
            logger.info(f"  Plotting {group_name} ({len(available_vars)} variables)...")
            f = plot_instrument_timeseries(df, group_name, available_vars, output_dir)
            generated_files.append(f)
        else:
            logger.warning(f"  ⚠ {group_name}: No data available, skipping plot.")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Generated {len(generated_files)} plots in {output_dir}/")
    for f in generated_files:
        logger.info(f"  📊 {f}")


if __name__ == "__main__":
    main()
