# Improved Bi-LSTM Model - Implementation Guide

## 📋 Overview

Se ha implementado una versión **científicamente rigurosa** del modelo Bi-LSTM con las siguientes mejoras:

### ✅ Mejoras Implementadas

1. **Train/Validation/Test Split Temporal (60%/20%/20%)**
   - Split cronológico (NO aleatorio)
   - Train: 60% primeros datos
   - Validation: 20% siguientes
   - Test: 20% últimos datos
   
2. **NO Shuffle - Orden Temporal Preservado**
   - `shuffle=False` en todos los DataLoaders
   - Respeta la autocorrelación temporal
   - Evita data leakage

3. **Enfoque Multivariado**
   - Usa correlaciones entre variables
   - Auto-selección de predictores
   - Ejemplo: TEMP predice usando TEMP, PSAL, LAND_AIRT, LAND_WSPD

4. **Early Stopping**
   - Monitorea validation loss
   - Patience = 10 epochs
   - Guarda mejor modelo automáticamente

5. **Métricas Comprehensivas**
   - Train loss por epoch
   - Validation loss por epoch
   - Test loss final
   - Mejor epoch identificado

---

## 📁 Archivos Creados

### 1. **Core Model** 
`/gap_project_antigr/src/models/multivariate_lstm_model.py`

Contiene:
- `MultivariateTimeSeriesDataset`: Dataset para series multivariadas
- `MultivariateLSTMModel`: Arquitectura del modelo
- `MultivariateLSTMImputer`: Wrapper completo con train/val/test

**Características clave:**
```python
class MultivariateLSTMImputer:
    """
    - Temporal split (60%/20%/20%)
    - shuffle=False (preserva orden)
    - Early stopping (patience=10)
    - Multivariate input
    - Residual learning (anomaly)
    """
```

### 2. **Integration Module**
`/scripts/interpolate_bilstm_improved.py`

Función drop-in para el pipeline:
```python
def interpolate_bilstm_improved(df, target_var, predictor_vars=None, config=None):
    """
    Reemplazo directo para interpolate_bilstm() actual.
    
    Características:
    - Auto-selección de predictores por correlación
    - Train/val/test split
    - NO shuffle
    - Early stopping
    - Métricas detalladas
    """
```

---

## 🔄 Diferencias vs Modelo Actual

| Característica | Modelo Actual | Modelo Mejorado |
|----------------|---------------|-----------------|
| **Train/Val/Test** | ❌ NO (usa todo para train) | ✅ SÍ (60%/20%/20%) |
| **Shuffle** | ✅ shuffle=True | ❌ shuffle=False |
| **Variables** | Univariado | Multivariado |
| **Early Stopping** | ❌ NO | ✅ SÍ (patience=10) |
| **Validación** | ❌ NO | ✅ Test set separado |
| **Auto-predictors** | N/A | ✅ Por correlación |
| **Métricas** | Train loss | Train/Val/Test loss |

---

## 📊 Configuración

### Configuración Recomendada

```python
IMPROVED_BILSTM_CONFIG = {
    # Model architecture
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'sequence_length': 96,        # 48h context
    
    # Training
    'epochs': 100,                # Más epochs, pero con early stopping
    'batch_size': 32,             # Más pequeño para mejor convergencia
    'learning_rate': 0.001,
    
    # Early stopping
    'early_stopping_patience': 10, # Para si no mejora en 10 epochs
    
    # Data split
    'train_ratio': 0.6,           # 60% para entrenamiento
    'val_ratio': 0.2,             # 20% para validación
                                  # 20% test (implícito)
}
```

### Auto-selección de Predictores

```python
def select_best_predictors(df, target_var, max_predictors=5, min_correlation=0.3):
    """
    Selecciona automáticamente las variables más correlacionadas.
    
    Para TEMP (ejemplo):
    - Busca correlación con todas las variables
    - Filtra por correlation >= 0.3
    - Selecciona top 5
    - Resultado: [TEMP, PSAL, LAND_AIRT, SVEL, LAND_WSPD]
    """
```

---

## 🚀 Cómo Usar

### Opción 1: Reemplazar directamente en pipeline

En `lup_data_obsea_analysis.py`, reemplazar la función `interpolate_bilstm()`:

```python
# ANTES (línea ~1374)
def interpolate_bilstm(df, target_var, config_bilstm=None, predictor_vars=None):
    # ... código actual univariado ...

# DESPUÉS
from interpolate_bilstm_improved import interpolate_bilstm_improved as interpolate_bilstm

# Ahora todas las llamadas a interpolate_bilstm() usarán el modelo mejorado
```

### Opción 2: Usar selectivamente

```python
from interpolate_bilstm_improved import interpolate_bilstm_improved, IMPROVED_BILSTM_CONFIG

# Para variables críticas (TEMP, PSAL)
if var in ['TEMP', 'PSAL']:
    result = interpolate_bilstm_improved(df, var, config_bilstm=IMPROVED_BILSTM_CONFIG)
else:
    result = interpolate_bilstm(df, var)  # Modelo actual
```

### Opción 3: Testing independiente

```python
# Test script standalone
from interpolate_bilstm_improved import interpolate_bilstm_improved
import pandas as pd

df = pd.read_csv("output_lup/data/OBSEA_multivariate_30min.csv", 
                 index_col=0, parse_dates=True)

# Test en TEMP
result = interpolate_bilstm_improved(
    df=df,
    target_var='TEMP',
    predictor_vars=['TEMP', 'PSAL', 'LAND_AIRT', 'LAND_WSPD'],
    config_bilstm=IMPROVED_BILSTM_CONFIG
)
```

---

## 📈 Ejemplo de Output Esperado

```
╔══ Multivariate Bi-LSTM (Rigorous) for TEMP ══╗
│ Device: cuda
│ Mode: Train/Val/Test split with early stopping
│ Shuffle: DISABLED (temporal order preserved)
│ Auto-selecting predictors by correlation...
│ Predictors (5): TEMP, PSAL, LAND_AIRT, SVEL, LAND_WSPD
│ Correlations: {PSAL: 0.87, LAND_AIRT: 0.76, SVEL: 0.94, LAND_WSPD: 0.42}
│ Computing climatological anomaly for TEMP...
│ ✓ Anomaly computed (climatology removed)
│ Complete records: 180,245 / 272,631
│ Training Multivariate Bi-LSTM...

Initialized Multivariate LSTM Imputer
  Target: TEMP
  Predictors: ['TEMP', 'PSAL', 'LAND_AIRT', 'SVEL', 'LAND_WSPD']
  Split: Train=60%, Val=20%, Test=20%
  Sequence length: 96 (temporal context)
  Device: cuda

Temporal split:
  Train: 163,578 samples (2009-05-29 to 2018-09-15)
  Val: 54,526 samples (2018-09-15 to 2022-01-28)
  Test: 54,527 samples (2022-01-28 to 2024-12-16)

Created dataloaders (shuffle=False for temporal order)
  Train batches: 5,112
  Val batches: 1,704
  Test batches: 1,704

Model architecture:
  Input features: 5
  Hidden size: 128
  Num layers: 2
  Bidirectional: True
  Total parameters: 329,473

Starting training for up to 100 epochs...
Early stopping patience: 10 epochs

  Epoch [1/100] Train Loss: 0.012456, Val Loss: 0.009876 ← BEST
  Epoch [5/100] Train Loss: 0.007234, Val Loss: 0.006543 ← BEST
  Epoch [10/100] Train Loss: 0.005987, Val Loss: 0.005234 ← BEST
  Epoch [15/100] Train Loss: 0.005123, Val Loss: 0.004789 ← BEST
  Epoch [20/100] Train Loss: 0.004567, Val Loss: 0.004234 ← BEST
  Epoch [25/100] Train Loss: 0.004234, Val Loss: 0.004156 ← BEST
  Epoch [30/100] Train Loss: 0.003987, Val Loss: 0.004187 (patience: 1/10)
  Epoch [35/100] Train Loss: 0.003756, Val Loss: 0.004245 (patience: 6/10)

Early stopping triggered at epoch 36
Best validation loss: 0.004156 at epoch 26

Restored best model from epoch 26

Training complete!
  Best epoch: 26
  Best val loss: 0.004156
  Test loss: 0.004289

│ ✓ Training complete
│   Best epoch: 26
│   Val loss: 0.004156
│   Test loss: 0.004289
│ Predicting gaps in anomaly space...
│ Reconstructing full signal (adding climatology)...
│ ✓ Multivariate residual learning complete
╚════════════════════════════════════════════════╝
```

---

## ⚠️ Consideraciones Importantes

### 1. **Tiempo de Ejecución**

El modelo mejorado es **MÁS LENTO** que el actual:

- **Modelo actual**: ~20-30 min por variable (30 epochs)
- **Modelo mejorado**: ~40-60 min por variable (puede correr hasta 100 epochs)

**Razones:**
- Más epochs disponibles (100 vs 30)
- Cálculos de validación por epoch
- Procesamiento multivariado

**Mitigación:**
- Early stopping para en promedio 25-35 epochs
- Usar solo en variables críticas (TEMP, PSAL)

### 2. **Uso de Memoria**

El modelo multivariado usa **MÁS memoria GPU**:

- Input: `(batch, sequence, n_features)` en vez de `(batch, sequence, 1)`
- Con 5 predictores: ~5x más datos en GPU

**Mitigación:**
- batch_size reducido a 32 (vs 64)
- Limitar predictores a top 5

### 3. **Datos Requeridos**

El split temporal requiere **datos suficientes en cada período**:

- Mínimo recomendado: 10,000 muestras completas
- Para 60%/20%/20%: Train >= 6,000, Val >= 2,000, Test >= 2,000

**Para dataset OBSEA:**
- Datos completos: ~180,000-200,000 (excelente)
- Train: ~110,000 ✅
- Val: ~36,000 ✅
- Test: ~36,000 ✅

---

## 🎯 Recomendaciones de Uso

### Escenario 1: Investigación Científica Rigurosa

**Usar modelo mejorado para:**
- Paper científico que requiere validación rigurosa
- Comparación con otros métodos
- Evaluar generalización del modelo

**Variables:**
- TEMP, PSAL (core oceanográfico)
- LAND_AIRT, LAND_WSPD (meteorológicas críticas)

### Escenario 2: Producción / Gap Filling Operacional

**Usar modelo actual para:**
- Rapidez de ejecución
- Todas las variables restantes
- Gap filling rutinario

**Variables:**
- CNDC, SVEL, LAND_RELH, LAND_WDIR, etc.

### Escenario 3: Híbrido (Recomendado)

```python
# En selective_interpolation()

CRITICAL_VARS = ['TEMP', 'PSAL']

for var in variables:
    if var in CRITICAL_VARS:
        # Modelo mejorado con validación rigurosa
        result = interpolate_bilstm_improved(df, var, config=IMPROVED_CONFIG)
    else:
        # Modelo actual para rapidez
        result = interpolate_bilstm(df, var, config=CURRENT_CONFIG)
```

---

## 📊 Plan de Testing

### 1. **Test Inicial (Standalone)**

```bash
cd /home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts

# Test rápido en subset pequeño
python3 -c "
from interpolate_bilstm_improved import interpolate_bilstm_improved, IMPROVED_BILSTM_CONFIG
import pandas as pd

df = pd.read_csv('output_lup/data/OBSEA_multivariate_30min.csv', index_col=0, parse_dates=True)
df_test = df.iloc[:10000]  # Solo 10k puntos para test rápido

result = interpolate_bilstm_improved(df_test, 'TEMP', config_bilstm=IMPROVED_BILSTM_CONFIG)
print(f'Success! Gaps filled: {result.notna().sum() - df_test[\"TEMP\"].notna().sum()}')
"
```

### 2. **Test en Pipeline (Una Variable)**

Modificar `lup_data_obsea_analysis.py` temporalmente para testear solo TEMP:

```python
# En benchmark_gap_filling(), línea ~1775
methods = ['linear', 'time', 'varma', 'bilstm_improved']  # Añadir nuevo

elif method == 'bilstm_improved':
    from interpolate_bilstm_improved import interpolate_bilstm_improved
    interpolated = interpolate_bilstm_improved(df_test, test_variable)
```

### 3. **Comparación Completa**

Correr ambos modelos en paralelo y comparar:
- RMSE en test set
- MAE en test set
- R² score
- Tiempo de ejecución

---

## 📝 Checklist de Implementación

- [x] **Modelo multivariado implementado** (`multivariate_lstm_model.py`)
- [x] **Función de integración creada** (`interpolate_bilstm_improved.py`)
- [x] **Documentación completa** (este archivo)
- [ ] **Test standalone ejecutado**
- [ ] **Test en pipeline (1 variable)**
- [ ] **Comparación con modelo actual**
- [ ] **Decisión: reemplazar completamente o usar híbrido**
- [ ] **Integración final en pipeline**
- [ ] **Actualización de benchmarking**
- [ ] **Paper/documentación científica**

---

## 🔬 Hipótesis Científicas a Validar

1. **Multivariate > Univariate**
   - H0: Usar predictores correlacionados NO mejora RMSE
   - H1: Multivariate RMSE < Univariate RMSE

2. **Temporal Split mejora generalización**
   - H0: Test loss similar entre ambos modelos
   - H1: Modelo mejorado generaliza mejor (test loss más cercano a train loss)

3. **Shuffle=False preserva patrones temporales**
   - H0: Shuffle no afecta predicción
   - H1: shuffle=False mejora predicción de tendencias

---

**Siguiente Paso:** Esperar que termine el pipeline actual para ver resultados baseline, luego testear modelo mejorado.

**Fecha de Implementación:** 2026-01-30  
**Status:** ✅ READY TO TEST (NO EJECUTAR HASTA COMPLETAR PIPELINE ACTUAL)
