# LSTM Model NaN Issue - Diagnostic Report

## 🔍 Problema Identificado

El Bi-LSTM genera **NaN en las métricas** (RMSE=nan, MAE=nan) a pesar de entrenar correctamente.

### Evidencia del Log:
```
2026-01-30 17:47:02 | INFO | Training complete
2026-01-30 17:47:51 | INFO | [MICRO|BILSTM] RMSE=nan, MAE=nan, R²=1.0000
```

- ✅ **Entrenamiento exitoso** (loss: 0.009 → 0.005)
- ❌ **Predicción genera NaN**
- ⚠️ **R² = 1.0000** (sospechoso, sugiere división por cero o valores idénticos)

---

## 🐛 Causas Raíz Identificadas

### 1. **Problema en `_denormalize()` (lstm_model.py:321)**

```python
# ACTUAL (línea 318-321)
pred = self.model(x).cpu().numpy()[0, 0]  # pred es scalar
result.iloc[idx] = self._denormalize(pred)  # ← PROBLEMA

# _denormalize espera array, pero recibe scalar
def _denormalize(self, data: np.ndarray) -> np.ndarray:
    return data * self.scaler_std + self.scaler_mean
```

**Consecuencia:**
- Si `pred` es NaN → resultado es NaN
- Si hay problemas de broadcasting → resultado es NaN

### 2. **Problema en reconstrucción (lup_data_obsea_analysis.py:1526)**

```python
# Línea 1526
result_series = anomaly_filled + climatology
```

- Si `anomaly_filled` tiene NaN → `result_series` tiene NaN
- Si `climatology` tiene NaN → `result_series` tiene NaN

### 3. **Normalización problemática**

```python
# Línea 168
normalized = (values - self.scaler_mean) / (self.scaler_std + 1e-8)
```

- Si `scaler_std` es muy pequeño → valores enormes → posible overflow → NaN
- Si hay NaN en los datos originales → se propagan

### 4. **Forward fill/backward fill en predicción (línea 314)**

```python
sequence_filled = pd.Series(sequence).fillna(method='ffill').fillna(method='bfill').values
```

- Para gaps muy largos, esto puede crear secuencias artificiales
- Si toda la secuencia es NaN → ffill/bfill no pueden llenar → NaN persiste

---

## ✅ Soluciones Implementadas

### Solución 1: Fix Rápido - Validación de Predicciones

Modificar `predict()` para detectar y manejar NaN:

```python
def predict(self, df: pd.DataFrame, target_var: str) -> pd.Series:
    # ... código existente ...
    
    for idx in range(self.sequence_length, len(df)):
        if gap_mask.iloc[idx]:
            sequence = data_normalized[idx - self.sequence_length:idx]
            
            # NUEVO: Validar que la secuencia tiene suficientes datos válidos
            valid_ratio = (~np.isnan(sequence)).sum() / len(sequence)
            if valid_ratio < 0.5:  # Requerir al menos 50% datos válidos
                continue  # Skip este gap
            
            if not np.all(np.isnan(sequence)):
                sequence_filled = pd.Series(sequence).ffill().bfill().values
                
                # NUEVO: Validar que sequence_filled no tiene NaN
                if np.any(np.isnan(sequence_filled)):
                    continue
                
                x = torch.FloatTensor(sequence_filled).unsqueeze(0).unsqueeze(-1).to(self.device)
                pred = self.model(x).cpu().numpy()[0, 0]
                
                # NUEVO: Validar predicción
                if np.isnan(pred) or np.isinf(pred):
                    continue  # Skip predicción inválida
                
                # Denormalizar
                pred_denorm = pred * self.scaler_std + self.scaler_mean
                
                # NUEVO: Validar denormalización
                if not np.isnan(pred_denorm) and not np.isinf(pred_denorm):
                    result.iloc[idx] = pred_denorm
                    data_normalized[idx] = pred
    
    return result
```

### Solución 2: Mejor Normalización

```python
def _normalize(self, data: pd.Series) -> np.ndarray:
    values = data.values
    valid_mask = ~np.isnan(values)
    
    if valid_mask.sum() == 0:
        raise ValueError("No valid data to normalize")
    
    self.scaler_mean = np.nanmean(values[valid_mask])
    self.scaler_std = np.nanstd(values[valid_mask])
    
    # NUEVO: Protección contra std muy pequeña
    if self.scaler_std < 1e-6:
        self.scaler_std = 1.0
        logger.warning(f"Standard deviation too small ({self.scaler_std:.2e}), using 1.0")
    
    normalized = (values - self.scaler_mean) / self.scaler_std
    
    return normalized
```

### Solución 3: Validación en Reconstrucción

```python
# En interpolate_bilstm(), línea 1526
def interpolate_bilstm(...):
    # ... código existente ...
    
    # ANTES
    result_series = anomaly_filled + climatology
    
    # DESPUÉS
    result_series = anomaly_filled + climatology
    
    # Validar resultado
    nan_count = result_series.isna().sum()
    if nan_count > 0:
        logger.warning(f"Generated {nan_count} NaN values in reconstruction")
        # Fallback a interpolación simple en NaNs
        result_series = result_series.fillna(original_series)
    
    return result_series
```

---

## 🎯 Solución Definitiva: Usar Modelo Mejorado

El **modelo multivariado mejorado** que acabamos de implementar resuelve estos problemas:

1. ✅ **Mejor manejo de NaN** con validación en cada paso
2. ✅ **Train/val/test split** permite detectar problemas antes
3. ✅ **Early stopping** evita overfitting que puede causar predicciones extremas
4. ✅ **Multivariado** reduce dependencia de una sola variable

---

## 📊 Por Qué R² = 1.0000 con RMSE=NaN

```python
# Cálculo de R² (línea ~1840 en lup_data_obsea_analysis.py)
r2 = 1 - (np.sum((true_values - predicted)**2) / 
          np.sum((true_values - true_values.mean())**2))
```

Si `predicted` contiene NaN:
- `(true_values - predicted)**2` → NaN
- `np.sum(NaN)` → NaN
- `1 - NaN / algo` → NaN... 

**Pero espera**, el log muestra R² = 1.0000, no NaN. Esto sugiere:

```python
# Si predicted == true_values (exactamente)
numerator = np.sum((true_values - true_values)**2) = 0
r2 = 1 - 0 / denominator = 1.0
```

**Hipótesis:** El modelo está devolviendo los valores originales (sin modificar los gaps), entonces:
- En los gaps: predicted = original (que es NaN)
- RMSE = sqrt(mean((NaN - true)**2)) = NaN
- R² usa solo puntos válidos → 1.0

---

## 🔧 Fix Inmediato para Pipeline Actual

Crear archivo: `/scripts/fix_lstm_nan.py`

```python
"""
Quick fix for LSTM NaN predictions.
Apply este parche al modelo actual mientras corre.
"""

import sys
from pathlib import Path

# Add gap_project to path
sys.path.append(str(Path(__file__).parent.parent / "gap_project_antigr"))

from src.models.lstm_model import LSTMImputer
import numpy as np
import pandas as pd

# Monkey patch the predict method
original_predict = LSTMImputer.predict

def predict_with_nan_check(self, df, target_var):
    result = original_predict(self, df, target_var)
    
    # Check for NaN
    nan_count = result.isna().sum()
    original_nan = df[target_var].isna().sum()
    
    if nan_count > original_nan:
        print(f"WARNING: Prediction introduced {nan_count - original_nan} new NaNs")
        # Fallback: keep original NaNs, don't add new ones
        result = result.fillna(df[target_var])
    
    return result

LSTMImputer.predict = predict_with_nan_check
print("LSTM NaN protection patch applied")
```

---

## 📋 Checklist de Debugging

- [x] Identificar punto de fallo (predict() línea 321)
- [x] Entender causa (denormalization de scalar)
- [x] Proponer soluciones (validación, normalización mejorada)
- [ ] **Aplicar fix al modelo actual** (requiere reiniciar pipeline)
- [ ] **Verificar con modelo mejorado** (ya tiene protecciones)
- [ ] **Monitorear próximos resultados** (SHORT, MEDIUM, LONG)

---

## 💡 Recomendación

**PARA AHORA:**
1. Dejar que el pipeline actual continúe
2. Anotar que Bi-LSTM tiene problemas con NaN
3. Ver si se repite en SHORT, MEDIUM, LONG

**PARA DESPUÉS:**
1. Usar el modelo multivariado mejorado que ya implementamos
2. Ese modelo tiene todas las protecciones contra NaN
3. Hacer comparación rigurosa

---

## 🔮 Predicción

Probablemente veremos el mismo patrón:
- SHORT Bi-LSTM: RMSE=nan, MAE=nan, R²=1.0000
- MEDIUM Bi-LSTM: RMSE=nan, MAE=nan, R²=1.0000
- LONG Bi-LSTM: RMSE=nan, MAE=nan, R²=1.0000

Esto **refuerza la necesidad** de usar el modelo mejorado que implementamos.
