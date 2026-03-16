import numpy as np
import pandas as pd
import logging

from obsea_pipeline.config.settings import QC_CONFIG

logger = logging.getLogger(__name__)

def interpolate_linear(series, max_gap=None):
    """
    Interpolación Lineal básica.
    """
    limit = max_gap if max_gap else QC_CONFIG['max_fill_gap']
    return series.interpolate(method='linear', limit=limit)

def interpolate_time(series, max_gap=None):
    """
    Interpolación basada en el índice temporal (fechas).  
    Suele ser la mejor opción de backup para Micro gaps si no hay dependencias cruzadas.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("interopolate_time requiere un pd.DatetimeIndex")
        
    limit = max_gap if max_gap else QC_CONFIG['max_fill_gap']
    return series.interpolate(method='time', limit=limit)

def interpolate_spline(series, order=1, max_gap=None):
    """
    Interpolación mediante Splines de orden variable.
    """
    limit = max_gap if max_gap else QC_CONFIG['max_fill_gap']
    return series.interpolate(method='spline', order=order, limit=limit)

def interpolate_polynomial(series, order=2, max_gap=None):
    """
    Interpolación Polinómica de orden variable.
    """
    limit = max_gap if max_gap else QC_CONFIG['max_fill_gap']
    return series.interpolate(method='polynomial', order=order, limit=limit)

def interpolate_pchip(series, max_gap=None):
    """
    Piecewise Cubic Hermite Interpolating Polynomial.
    Excelente preservador de monotonía (no inventa picos artificiales como 'spline').
    """
    limit = max_gap if max_gap else QC_CONFIG['max_fill_gap']
    return series.interpolate(method='pchip', limit=limit)

def run_basic_models(df):
    """
    Aplica todos los modelos básicos en paralelo y devuelve un diccionario de outputs.
    Útil para el script de benchmark.
    """
    results = {}
    
    logger.info("  Running [basic models]...")
    results['linear'] = df.apply(lambda col: interpolate_linear(col))
    results['time'] = df.apply(lambda col: interpolate_time(col))
    results['spline_linear'] = df.apply(lambda col: interpolate_spline(col, order=1))
    results['spline_quadratic'] = df.apply(lambda col: interpolate_spline(col, order=2))
    results['spline_cubic'] = df.apply(lambda col: interpolate_spline(col, order=3))
    results['polynomial_2'] = df.apply(lambda col: interpolate_polynomial(col, order=2))
    results['polynomial_3'] = df.apply(lambda col: interpolate_polynomial(col, order=3))
    results['pchip'] = df.apply(lambda col: interpolate_pchip(col))

    return results
