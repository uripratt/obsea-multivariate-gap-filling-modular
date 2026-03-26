import numpy as np
import pandas as pd
import logging
import math

logger = logging.getLogger(__name__)

# =============================================================================
# VECTORIZED OCEANOGRAPHIC COMPUTATIONS (UNESCO EOS-80)
# =============================================================================

def compute_density_sigma_vectorized(T: pd.Series, S: pd.Series, P: pd.Series) -> pd.Series:
    """
    Vectorized Seawater Density anomaly (Sigma-T) using full UNESCO EOS-80.
    ~100× faster than row-wise .apply(). Handles NaN via NumPy propagation.
    """
    kw = 19652.21 + 148.4206*T - 2.327105*T**2 + 1.360477e-2*T**3 - 5.155288e-5*T**4
    Aw = 3.239908 + 1.43713e-3*T + 1.16092e-4*T**2 - 5.77905e-7*T**3
    Bw = 8.50935e-5 - 6.12293e-6*T + 5.2787e-8*T**2

    K0 = kw + (54.6746 - 0.603459*T + 1.09987e-2*T**2 - 6.1670e-5*T**3)*S + \
         (7.944e-2 + 1.6483e-2*T - 5.3009e-4*T**2) * S**1.5
    K = K0 + (Aw + (2.2838e-3 - 1.0981e-5*T - 1.6078e-6*T**2)*S + 1.91075e-4*S**1.5)*P + \
        (Bw + (-9.9348e-7 + 2.0816e-8*T + 9.1697e-10*T**2)*S) * P**2

    rho_w = (999.842594 + 6.793952e-2*T - 9.095290e-3*T**2 +
             1.001685e-4*T**3 - 1.120083e-6*T**4 + 6.536336e-9*T**5)

    rho_0 = (rho_w + (0.824493 - 4.0899e-3*T + 7.6438e-5*T**2 -
             8.2467e-7*T**3 + 5.3875e-9*T**4)*S +
             (-5.72466e-3 + 1.0227e-4*T - 1.6546e-6*T**2)*S**1.5 +
             4.8314e-4*S**2)

    rho = rho_0 / (1 - P / K)
    return rho - 1000.0  # Sigma-T anomaly


def decompose_wind_uv_vectorized(wspd: pd.Series, wdir: pd.Series):
    """
    Vectorized wind UV decomposition. Converts meteorological convention to U/V components.
    Returns (U, V) as two pd.Series.
    """
    math_dir = np.deg2rad(270.0 - wdir)
    u = wspd * np.cos(math_dir)
    v = wspd * np.sin(math_dir)
    return u, v


def compute_wind_stress_vectorized(wspd: pd.Series) -> pd.Series:
    """Vectorized bulk formula: τ = ρ_air × C_d × U²"""
    rho_air = 1.225   # kg/m³
    Cd = 0.0013       # Drag coefficient over sea surface
    return rho_air * Cd * (wspd ** 2)


def compute_wave_energy_vectorized(vhm0: pd.Series, vtpk: pd.Series) -> pd.Series:
    """Vectorized wave power: P = (ρg²/64π) × Hs² × Te [W/m → kW/m]"""
    rho = 1025.0
    g = 9.81
    power_w = (rho * (g**2) / (64 * np.pi)) * (vhm0**2) * vtpk
    return power_w / 1000.0  # kW


def compute_brunt_vaisala(sigma_t_series: pd.Series, pres_series: pd.Series) -> pd.Series:
    """
    Buoyancy frequency N = sqrt(|N²|), where N² = (g/ρ₀) × dσ/dp.
    
    WARNING: This requires gap-free data for meaningful derivatives.
    Call AFTER interpolation, not before.
    """
    g = 9.81
    rho_0 = 1025.0
    dp = pres_series.diff().replace(0, np.nan)
    N2 = (g / rho_0) * (sigma_t_series.diff() / dp)
    return np.sqrt(np.abs(N2))


# =============================================================================
# LEGACY SCALAR FUNCTIONS (kept for backward compatibility)
# =============================================================================

def compute_density_sigma(temp, psal, pres):
    """Scalar wrapper — prefer compute_density_sigma_vectorized for DataFrames."""
    if temp is None or psal is None or pres is None:
        return np.nan
    try:
        return float(compute_density_sigma_vectorized(
            pd.Series([temp]), pd.Series([psal]), pd.Series([pres])
        ).iloc[0])
    except Exception:
        return np.nan

def decompose_wind_uv(wspd, wdir):
    """Scalar wrapper — prefer decompose_wind_uv_vectorized for DataFrames."""
    if np.isnan(wspd) or np.isnan(wdir):
        return np.nan, np.nan
    u, v = decompose_wind_uv_vectorized(pd.Series([wspd]), pd.Series([wdir]))
    return float(u.iloc[0]), float(v.iloc[0])

def compute_wind_stress(wspd):
    """Scalar wrapper."""
    if pd.isna(wspd): return np.nan
    return float(compute_wind_stress_vectorized(pd.Series([wspd])).iloc[0])

def compute_wave_energy(vhm0, vtpk):
    """Scalar wrapper."""
    if pd.isna(vhm0) or pd.isna(vtpk): return np.nan
    return float(compute_wave_energy_vectorized(pd.Series([vhm0]), pd.Series([vtpk])).iloc[0])


# =============================================================================
# MISSINGNESS FEATURES (MNAR)
# =============================================================================

def add_missingness_features(df):
    """
    Modelado explícito de valores faltantes (Missing Not At Random - MNAR).
    Crea un vector binario y un contador direccional de Delta T para ayudar a los
    modelos ML a entender físicamente cuándo los sensores fallan.
    """
    logger.info("Computing Explicit Missingness Features (MNAR)...")
    res = df.copy()
    
    # Procesar solo las variables puras del sensor (evitar derivadas si es posible)
    target_vars = [c for c in df.columns if c in ['TEMP', 'PSAL', 'WSPD', 'CSPD', 'VHM0', 'PRES']]
    
    for col in target_vars:
        # 1. Binary Mask (1 si existe valor real, 0 si es NaN o ha sido borrado por QC)
        res[f'{col}_MASK'] = res[col].notna().astype(int)
        
        # 2. Time Since Last Observation (Delta t)
        is_missing = res[col].isna()
        groups = (~is_missing).cumsum()
        res[f'{col}_DT'] = is_missing.groupby(groups).cumsum()
        
    return res


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def add_derived_features(df):
    """
    Agrega todas las características derivadas físicamente posibles al dataframe procesado.
    Uses vectorized operations for performance (~100× faster than row-wise .apply).
    """
    logger.info("Computing Oceanographic specific physical derivations (vectorized)...")
    result = df.copy()
    
    # 1. Density System (Sigma-T) — vectorized UNESCO EOS-80
    if all(c in df.columns for c in ['TEMP', 'PSAL', 'PRES']):
        result['SIGMA_T'] = compute_density_sigma_vectorized(df['TEMP'], df['PSAL'], df['PRES'])
        # NOTE: VB_FREQ requires gap-free data for dσ/dp. It is computed AFTER interpolation
        # in production mode. Here we only set a placeholder so the column exists for downstream code.
        result['VB_FREQ'] = np.nan
        logger.info("  → SIGMA_T computed. VB_FREQ deferred to post-interpolation.")

    # 2. Wind (UV Components + Stress) — vectorized
    # Search for any wind speed/direction columns with prefix matching
    wspd_col = next((c for c in df.columns if c.endswith('WSPD')), None)
    wdir_col = next((c for c in df.columns if c.endswith('WDIR')), None)
    if wspd_col and wdir_col:
        u, v = decompose_wind_uv_vectorized(df[wspd_col], df[wdir_col])
        result['WIND_U'] = u
        result['WIND_V'] = v
        result['WIND_STRESS'] = compute_wind_stress_vectorized(df[wspd_col])
        logger.info(f"  → WIND_U, WIND_V, WIND_STRESS computed from {wspd_col}/{wdir_col}.")

    # 3. Currents (UV Components) — vectorized
    cspd_col = next((c for c in df.columns if c.endswith('CSPD')), None)
    cdir_col = next((c for c in df.columns if c.endswith('CDIR')), None)
    if cspd_col and cdir_col:
        u, v = decompose_wind_uv_vectorized(df[cspd_col], df[cdir_col])
        result['CURR_U'] = u
        result['CURR_V'] = v
        logger.info(f"  → CURR_U, CURR_V computed from {cspd_col}/{cdir_col}.")

    # 4. Wave Energy — vectorized
    if all(c in df.columns for c in ['VHM0', 'VTPK']):
        result['WAVE_ENERGY'] = compute_wave_energy_vectorized(df['VHM0'], df['VTPK'])
        logger.info("  → WAVE_ENERGY computed.")

    # 5. Missingness Modeling (MNAR)
    result = add_missingness_features(result)

    return result


def compute_post_interpolation_features(df):
    """
    Features that require gap-free data. Call AFTER interpolation.
    Currently computes: VB_FREQ (Brunt-Väisälä frequency).
    """
    if all(c in df.columns for c in ['SIGMA_T', 'PRES']):
        df['VB_FREQ'] = compute_brunt_vaisala(df['SIGMA_T'], df['PRES'])
        logger.info("  → VB_FREQ computed post-interpolation.")
    return df

