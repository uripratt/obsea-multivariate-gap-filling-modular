# Auditoría Técnica: XGBoost, XGBoost Pro y BiLSTM

He analizado en profundidad las implementaciones de los tres modelos principales (`xgboost_model.py`, `xgboost_model_pro.py` y `multivariate_lstm_model.py`) junto con el módulo de generación de variables (`temporal_features.py`).

Aunque la intención científica detrás de los modelos es muy buena (uso de STL, variables exógenas, decaimiento dinámico), existen **errores estructurales graves** en la implementación que están rompiendo su rendimiento y causando resultados mediocres en la práctica.

A continuación detallo los problemas críticos de cada modelo y sus soluciones.

---

## 1. XGBoost y XGBoost Pro

Ambos modelos comparten una arquitectura autorregresiva (predicen un paso, lo usan como lag para el siguiente). Tienen los siguientes problemas críticos:

### 🔴 Problema 1: El bug "Empty Dataset" en la generación de features
**Archivo**: `xgboost_model.py` (líneas 150-163) y `xgboost_model_pro.py` (líneas 168-181)

**El problema**:
Para evitar que los NaNs se propaguen al crear features de "lag" (retraso), el código intenta pre-llenar los datos con climatología *antes* de generar las features:
```python
climatology_fallback = df_masked[target_var].groupby(group_cols).transform('mean')
s_interp = df_masked[target_var].fillna(climatology_fallback)
df_for_features[target_var] = df_masked[target_var].fillna(s_interp)
df_features = feat_eng.fit_transform(df_for_features, ...)
```
Pero **luego**, al extraer el set de entrenamiento, hace esto:
```python
valid_mask = df[target_var].notna()  # df NO está pre-llenado
X = df_features.loc[valid_mask].copy()
```
**Efecto**: Estás entrenando el modelo para predecir valores reales `y` usando features `X` que fueron generadas sobre una serie pre-llenada con climatología. Esto confunde al modelo: aprende que cuando las variables de lag se parecen a la climatología perfecta, el resultado es el valor real ruidoso. Rompe por completo la relación autorregresiva natural.

### 🔴 Problema 2: Decaimiento Dinámico (Exposure Bias Mitigation) erróneo
**Archivo**: `xgboost_model.py` (línea 387) y `xgboost_model_pro.py` (línea 406)

**El problema (XGBoost Normal)**:
El modelo XGBoost normal fue cambiado para predecir valores *absolutos* (no residuales), pero conserva la lógica de decaimiento diseñada para residuales:
```python
decay_factor = np.exp(-consecutive_gap_idx / 576.0)
pred_absolute = (pred_absolute * decay_factor) + (base_vals[pos] * (1.0 - decay_factor))
```
**Efecto**: A medida que el gap se hace largo (ej. 3 días), el XGBoost normal abandona su propia predicción absoluta y empuja el valor hacia `base_vals`, que es **la climatología estática** o la simple interpolación lineal (guardada en `s_interp`). Por eso en gaps largos, XGBoost se vuelve una simple curva climática sin varianza. 

**El problema (XGBoost Pro)**:
El modelo Pro sí predice residuales, pero aplica el decaimiento de forma súper agresiva:
```python
decay_factor = np.exp(-consecutive_gap_idx / 576.0)
pred = pred * decay_factor
```
**Efecto**: El decaimiento es exponencial cóncavo. Para un `consecutive_gap_idx` pequeño (ej, 48 pasos = 24 horas), `np.exp(-48/576) = 0.92`. Para 3 días (144 pasos), es 0.77. Pese a que parece suave, XGBoost Pro pierde casi toda la amplitud de su predicción residual en muy pocas horas y converge a ser literalmente la base de STL (la cual en gaps muy largos tampoco es perfecta).

### 🔴 Problema 3: Propagación de Errores con Rolling Stats
**Archivo**: `temporal_features.py` y `xgboost_model_pro.py`

**El problema**: 
`XGBoost Pro` calcula estadísticas "rolling" (media, desviación estándar, min, max) dinámicamente durante el bucle de predicción (líneas 363-375). 
```python
window_data = y_values[start_w : pos + 1]
col = f"{target}_roll_mean_{window}"
val = np.nanmean(window_data)
```
**Efecto**: En un gap largo, `window_data` se llena completamente de *tus propias predicciones anteriores*. Si la predicción anterior tuvo un ligero error (ej. se quedó "enganchada" alta), la media movil será alta, lo que hará que el siguiente paso predicto sea aún más alto. **Esto causa derivas severas o líneas planas**.

---

## 2. BiLSTM (Multivariate LSTM)

El modelo BiLSTM ha sido rediseñado reciéntemente y es estructuralmente más sólido que XGBoost, pero sufre de problemas mortales de validación y de formato de datos.

### 🔴 Problema 1: Data Leakage Masivo en Validacion/Test
**Archivo**: `multivariate_lstm_model.py` (Dataset y Split)

**El problema**:
El `MultivariateTimeSeriesDataset` se crea después del `StandardScaler`, lo cual está bien. Sin embargo, usa el `df` original sin tapar los gaps reales.
Más importante, en la función `fit()`, hay un intento de resolver el STL:
```python
df[self.target_var] = df[self.target_var] - self.trend_comp - self.seasonal_comp
```
El `STL` se aplica sobre **todo** el DataFrame antes de hacer el split de Test/Val/Train (líneas 405-422). Al hacer `STL` sobre todo el dataset, el algoritmo loess mira hacia el futuro para calcular el componente estacional y de tendencia del pasado. 
**Efecto**: **Fuga de datos del futuro al pasado**. El modelo "aprende" sobre el set de test durante el entrenamiento porque las señales de tendencia/seasonality se calcularon usando los datos de test.

### 🔴 Problema 2: Arquitectura del Dataset (Manejo de NaNs en variables multivariadas)
**Archivo**: `multivariate_lstm_model.py` (Línea 105-108)

**El problema**:
El Dataset intenta ignorar secuencias con NaNs:
```python
is_valid = not np.isnan(sequence).any() and not np.isnan(target)
sequence = np.nan_to_num(sequence, nan=0.0) # Rellena con 0
```
Si usas *multivariate variables* (ej. Salinidad y Temperatura), y Temperatura tiene un hueco pero Salinidad no... `np.isnan(sequence).any()` será `True`, y el `is_valid` será `False`.
En el bucle de entrenamiento (líneas 544-545):
```python
valid_mask = mask_batch.view(-1)
if not valid_mask.any(): continue
```
**Efecto**: Si **cualquier** variable exógena (o el propio target en pasos históricos) tiene un NaN en una ventana de 96 pasos, **toda la ventana se descarta** del entrenamiento. En datos oceanográficos reales donde los gaps ocurren frecuentemente en distintos sensores en diferentes momentos, acabas destruyendo el 80% de tu dataset de entrenamiento. BiLSTM termina entrenando con una cantidad minúscula de datos perfectos, y falla horriblemente en generalizar a datos reales con huecos parciales.

### 🔴 Problema 3: Inicialización Predictiva para LSTM
**Archivo**: `multivariate_lstm_model.py` (predict method, línea 741-746)

**El problema**:
Al predecir recursivamente, el algoritmo toma la ventana anterior:
```python
sequence = data_np[idx - self.sequence_length:idx].copy()
nan_mask = np.isnan(sequence[:, target_idx])
if nan_mask.any():
    s = pd.Series(sequence[:, target_idx])
    sequence[:, target_idx] = s.ffill().bfill().fillna(0.0).values
```
**Efecto**: Si estás en un hueco de forma consecutiva, los valores anteriores deberían haber sido rellenados ya por tus predicciones anteriores (ver línea 789). Pero si hay NaNs residuales (por ej, porque las variables predictorias tienen gaps), haces un padding ultra-simple (`ffill` / `bfill`) en tiempo de inferencia que el modelo *jamás vio en entrenamiento* (porque en entrenamiento descartaba secuencias con NaNs). El cambio abrupto de distribución estadística destruye las predicciones del LSTM.

---

## Plan de Acción Recomendado

### Para XGBoost y XGBoost Pro:
1. **Quitar las `rolling_stats` dinámicas** durante el fill de gaps largos, o fijarlas estáticamente a la última observación real, para evitar propagación al infinito.
2. **Re-escribir el feature engineering** para que use la serie real con NaNs intactos durante el `fit()`. XGBoost maneja NaNs nativamente en sus *features*. Solo se deben tirar las filas donde el *target* (`y`) es NaN.
3. **Ajustar el decay_factor** (Exposure bias). En vez de que caiga exponencialmente hacia la climatología desde la hora 1, debe mantenerse 100% en la predicción del XGBoost por las primeras X horas, y solo decaer muy lentamente a partir de ahí.

### Para BiLSTM:
1. Cambiar la lógica del `MultivariateTimeSeriesDataset`. En lugar de rechazar ventanas con cualquier NaN, debe rellenar los NaNs de las variables *predictorias* con interpolación antes de extraer ventanas, y añadir una *Observation Mask* binaria como variable de entrada adicional (para que la red sepa que ese dato es interpolado). 
2. Realizar el STL *después* del split de Train/Val, o usando predicción hacia adelante, para evitar Data Leakage.
3. Reducir la dependencia autorregresiva total usando "Teacher Forcing probabilístico" durante el entrenamiento.
