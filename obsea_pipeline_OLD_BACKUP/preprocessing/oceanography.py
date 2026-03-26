import numpy as np
import pandas as pd
import logging
import math

logger = logging.getLogger(__name__)

def compute_density_sigma(temp, psal, pres):
    """
    Computes accurate Seawater Density anomaly (Sigma-T) using UNESCO/TEOS-10 derived approximations.
    """
    if temp is None or psal is None or pres is None:
        return np.nan
        
    try:
        T = temp
        S = psal
        p = pres
        
        kw = 19652.21 + 148.4206 * T - 2.327105 * T**2 + 1.360477e-2 * T**3 - 5.155288e-5 * T**4
        Aw = 3.239908 + 1.43713e-3 * T + 1.16092e-4 * T**2 - 5.77905e-7 * T**3
        Bw = 8.50935e-5 - 6.12293e-6 * T + 5.2787e-8 * T**2
        
        K0 = kw + (54.6746 - 0.603459 * T + 1.09987e-2 * T**2 - 6.1670e-5 * T**3) * S + \
             (7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * T**2) * S**1.5
        K = K0 + (Aw + (2.2838e-3 - 1.0981e-5 * T - 1.6078e-6 * T**2) * S + 1.91075e-4 * S**1.5) * p + \
            (Bw + (-9.9348e-7 + 2.0816e-8 * T + 9.1697e-10 * T**2) * S) * p**2
            
        rho_w = 999.842594 + 6.793952e-2 * T - 9.095290e-3 * T**2 + 1.001685e-4 * T**3 - 1.120083e-6 * T**4 + 6.536336e-9 * T**5
        
        rho_0 = rho_w + (0.824493 - 4.0899e-3 * T + 7.6438e-5 * T**2 - 8.2467e-7 * T**3 + 5.3875e-9 * T**4) * S + \
                (-5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2) * S**1.5 + 4.8314e-4 * S**2
                
        rho = rho_0 / (1 - p / K)
        return rho - 1000.0 # Sigma-T anomaly
    except Exception:
        return np.nan

def decompose_wind_uv(wspd, wdir):
    """
    Descompone la magnitud y dirección del viento en sus componentes U (Zonal) y V (Meridional).
    Convierte de la convención meteorológica ("hacia donde sopla") a la oceanográfica ("de donde viene") 
    o viceversa dependiendo del uso. Usualmente WDIR es de donde viene.
    """
    if np.isnan(wspd) or np.isnan(wdir):
        return np.nan, np.nan
        
    math_dir = 270 - wdir
    rad = math.radians(math_dir)
    u = wspd * math.cos(rad)
    v = wspd * math.sin(rad)
    return u, v

def compute_wind_stress(wspd):
    """
    Cálculo de estrés térmico de viento en superficie (tau) 
    útil para modelos XGBoost que traten la estratificación térmica del agua.
    """
    rho_air = 1.225 # kg/m^3
    Cd = 0.0013 # Coeficiente arrastre standard sobre el mar
    if pd.isna(wspd): return np.nan
    return rho_air * Cd * (wspd ** 2)

def compute_wave_energy(vhm0, vtpk):
    """
    Potencia aproximada de la ola (kW/m de cresta) basado en parámetros AWAC de STA v1.1
    P = (rho * g^2 / (64 * pi)) * Hs^2 * Te
    """
    try:
        if pd.isna(vhm0) or pd.isna(vtpk): return np.nan
        rho = 1025.0
        g = 9.81
        power_w = (rho * (g**2) / (64 * math.pi)) * (vhm0**2) * vtpk
        return power_w / 1000.0 # kW
    except:
        return np.nan

def compute_brunt_vaisala(sigma_t_series, pres_series):
    """
    Frecuencia de flotabilidad (Estratificación). Requiere derivadas temporales.
    Estimado superficial.
    """
    g = 9.81
    rho_0 = 1025.0
    N2 = (g / rho_0) * (sigma_t_series.diff() / pres_series.diff().replace(0, np.nan))
    return np.sqrt(np.abs(N2)) # Frecuencia N

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
        # Contamos cuántos steps han pasado desde el último 1 en la máscara
        is_missing = res[col].isna()
        # Cumsum de los saltos válidos para crear grupos
        groups = (~is_missing).cumsum()
        # El tamaño del grupo menos uno (porque el primero válido es 0 delta)
        # Usamos GroupBy cumcount para contar dentro de cada gap
        res[f'{col}_DT'] = is_missing.groupby(groups).cumsum()
        
    return res

def add_derived_features(df):
    """
    Agrega todas las características derivadas físicamente posibles al dataframe procesado.
    """
    logger.info("Computing Oceanographic specific physical derivations...")
    result = df.copy()
    
    # 1. Sistema Densidad (Sigma-T)
    if all(c in df.columns for c in ['TEMP', 'PSAL', 'PRES']):
        result['SIGMA_T'] = df.apply(lambda row: compute_density_sigma(row['TEMP'], row['PSAL'], row['PRES']), axis=1)
        # Brunt-Vaisala estimation (requires consecutive data, might have gaps so handle carefully)
        result['VB_FREQ'] = compute_brunt_vaisala(result['SIGMA_T'], result['PRES'])

    # 2. Viento (Componentes UV y Stress)
    if all(c in df.columns for c in ['WSPD', 'WDIR']):
        result[['WIND_U', 'WIND_V']] = df.apply(lambda r: pd.Series(decompose_wind_uv(r['WSPD'], r['WDIR'])), axis=1)
        result['WIND_STRESS'] = df['WSPD'].apply(compute_wind_stress)

    # 3. Corrientes (Componentes UV)
    if all(c in df.columns for c in ['CSPD', 'CDIR']):
        result[['CURR_U', 'CURR_V']] = df.apply(lambda r: pd.Series(decompose_wind_uv(r['CSPD'], r['CDIR'])), axis=1)

    # 4. Oleaje (Energía)
    if all(c in df.columns for c in ['VHM0', 'VTPK']):
         result['WAVE_ENERGY'] = df.apply(lambda r: compute_wave_energy(r['VHM0'], r['VTPK']), axis=1)

    # 5. Missingness Modeling (MNAR)
    result = add_missingness_features(result)

    return result
