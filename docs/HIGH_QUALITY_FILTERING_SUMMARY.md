# High-Quality Variable Filtering - Implementation Summary

## Objetivo
Filtrar el dataset multivariante de OBSEA para trabajar **solo con variables de alta calidad** (≤25% de gaps), reduciendo la carga computacional y mejorando la fiabilidad de los modelos de interpolación.

## Variables de Alta Calidad Seleccionadas (11 variables)

### Variables CTD (6) - Core oceanográfico
- **PSAL** (Salinidad) - 18.93% gaps
- **PSAL_ANOMALY** (Anomalía de salinidad) - 18.93% gaps  
- **TEMP** (Temperatura) - 21.92% gaps
- **TEMP_ANOMALY** (Anomalía de temperatura) - 21.92% gaps
- **CNDC** (Conductividad) - 21.92% gaps
- **SVEL** (Velocidad del sonido) - 21.92% gaps

### Variables Meteorológicas (3)
- **LAND_RELH** (Humedad relativa tierra) - 20.82% gaps
- **LAND_WSPD** (Velocidad viento tierra) - 20.82% gaps
- **LAND_WDIR** (Dirección viento tierra) - 20.86% gaps

### Variables Derivadas (2)
- **LAND_CAPH** (Presión atmosférica tierra) - 20.82% gaps
- **LAND_AIRT** (Temperatura aire tierra) - 21.09% gaps

## Cambios Implementados

### 1. Script de Análisis (`analyze_variable_quality.py`)
- Analiza todas las variables del dataset
- Calcula el porcentaje de gaps para cada variable
- Identifica y exporta las variables con ≤25% gaps
- Genera reportes en CSV y listas en formato Python

### 2. Configuración en `lup_data_obsea_analysis.py`

#### Nueva Constante Global:
```python
HIGH_QUALITY_VARIABLES = [
    'PSAL', 'PSAL_ANOMALY', 'TEMP', 'TEMP_ANOMALY', 'CNDC', 'SVEL',
    'LAND_RELH', 'LAND_WSPD', 'LAND_WDIR', 'LAND_CAPH', 'LAND_AIRT'
]

USE_HIGH_QUALITY_FILTER = True  # Flag para activar/desactivar el filtro
```

#### Nueva Función de Filtrado:
```python
def filter_high_quality_variables(df, high_quality_vars, keep_qc_std=True)
```
- Filtra el DataFrame para mantener solo las variables de alta calidad
- Conserva automáticamente las columnas _QC y _STD asociadas
- Loggea las variables retenidas

### 3. Modificaciones en Pipeline Principal

#### Nuevo Step 5.5: Variable Quality Filtering
- Se aplica **antes** del benchmarking y la interpolación
- Reduce el dataset de 33 → 11 variables
- Filtro también aplicado a `gaps_df` para coherencia

#### Steps reorganizados:
- **Step 5.5**: Variable Quality Filtering (NUEVO)
- **Step 5.6**: Benchmarking (antes 5.5) - ahora sobre datos filtrados
- **Step 5.7**: Selective Interpolation (antes 5.6) - ahora sobre datos filtrados

## Beneficios

### 1. Reducción Computacional
- **Menos variables → menos entrenamiento** de modelos VARMA y Bi-LSTM
- Tiempo de ejecución estimado: **reducción de ~70%**
- Menor uso de memoria GPU

### 2. Mayor Fiabilidad
- Variables con <25% gaps = **más datos válidos para entrenamiento**
- Modelos de gap-filling más robustos
- Predicciones más confiables

### 3. Foco Científico
- Concentración en variables core: **CTD + Meteorología**
- Exclusión de variables con cobertura insuficiente (AWAC: ~76% gaps)
- Dataset resultante más coherente temporalmente

## Archivos Generados

### Análisis de Calidad
- `output_lup/tables/variable_quality_analysis.csv` - Análisis completo de todas las variables
- `output_lup/tables/high_quality_variables.csv` - Solo variables de alta calidad
- `output_lup/tables/high_quality_variable_names.txt` - Lista de nombres

### Pipeline Output (filtrado)
- `output_lup/data/OBSEA_multivariate_30min.csv` - **FILTRADO** a 11 variables
- `output_lup/data/OBSEA_multivariate_30min_interpolated.csv` - **FILTRADO** e interpolado
- Todas las figuras y tablas reflejan solo las variables de alta calidad

## Cómo Desactivar el Filtro

Si se necesita volver a procesar todas las variables:

```python
# En lup_data_obsea_analysis.py, línea ~219
USE_HIGH_QUALITY_FILTER = False
```

## Variables Excluidas (y sus % de gaps)

### AWAC Currents (~76% gaps)
- UCUR, VCUR, ZCUR, CSPD, CDIR

### AWAC Waves (~77% gaps)  
- VHM0, VTPK, VTM02, VMDR, VPED

### Variables con gaps moderados-altos (25-76%)
- PRES, DENS, features derivados oceánicos con dependencia de AWAC
- Variables Airmar (ATMS, DRYT, etc.) - preferencia por CTVG (land station)

## Próximos Pasos

1. **Ejecutar pipeline con filtro activado**
   ```bash
   python3 lup_data_obsea_analysis.py
   ```

2. **Verificar resultados** en `output_lup/`

3. **Actualizar webapp** para mostrar solo variables de alta calidad

4. **(Opcional)** Ajustar configuración de modelos para dataset reducido
