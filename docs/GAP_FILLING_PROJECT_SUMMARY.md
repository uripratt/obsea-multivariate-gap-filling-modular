# Proyecto LUP: Multivariate Gap Filling Pipeline para Series Temporales Oceanográficas (OBSEA)

## 1. Introducción y Estado del Arte (State of the Art)

### El Estado del Arte en Oceanografía
El rellenado de huecos (*gap filling* o imputación) en series temporales oceanográficas es un desafío crítico. Los observatorios cabled (como OBSEA), boyas y moorings sufren interrupciones debido a *bio-fouling*, fallos de alimentación, cortes de comunicación o mantenimiento rutinario. Históricamente, la comunidad oceanográfica ha dependido de:
1.  **Métodos Estadísticos Simples:** Interpolación lineal, *splines* cúbicos o promedios climatológicos. Estos fallan dramáticamente en huecos largos (> 1 semana) porque ignoran la variabilidad de alta frecuencia y la inercia del mar.
2.  **Análisis Armónico y Climatológico:** Ajuste de curvas polinómicas o funciones sinusoidales. Capturan la estacionalidad anual, pero son rígidos e incapaces de reaccionar a eventos anómalos (borrascas, olas de calor marinas).
3.  **Machine Learning Temprano:** Redes Neuronales Artificiales (ANNs) básicas o Random Forests univariados, a menudo dependientes de una gran cantidad de datos históricos sin considerar la causalidad temporal estricta.

### La Brecha de Conocimiento (Gap in Knowledge)
La literatura actual adolece de varias deficiencias clave:
*   **Aislamiento Univariado:** La mayoría de modelos imputan una variable (e.g., Temperatura) usando solo su propio pasado. Ignoran la fuerte correlación termodinámica e hidrodinámica del océano (Salinidad, Vientos, Corrientes, Presión).
*   **Falta de Clasificación del Error por Escala:** Un modelo puede ser excelente rellenando 2 horas, pero inútil rellenando 2 meses. La literatura raramente evalúa la degradación temporal del error (*Scale-dependent degradation*).
*   **Física contra Datos:** Los modelos puramente de Deep Learning a menudo generan predicciones que violan principios físicos básicos (ej. picos de temperatura irreales) al carecer de restricciones ambientales.

### Tu Aportación como PhD (La Novedad Científica)
Tu trabajo propone el **MTSI Framework (Multivariate Time Series Imputation)** para observatorios multiparamétricos. Las aportaciones principales de tu herramienta automática son:
1.  **Fusión Multivariada Dinámica:** Utilizar variables ambientales cruzadas (e.g., usar los datos de la estación meteorológica AWAC/Airmar para rellenar vacíos en el CTD submarino).
2.  **Arquitectura "Auto-Supervised Residual-Learning":** En lugar de predecir la variable cruda, tus modelos XGBoost y Deep Learning aprenden a aislar puramente el *error residual* (ruido de alta frecuencia impulsado por la física) sobre una base estructural termodinámica.
3.  **Benchmarking Escalar Sistemático:** Clasificar los huecos en categorías (Micro, Small, Medium, Large, Extra-Large) y evaluar automáticamente qué familias algorítmicas (Interpolación, Ensemble Trees, RNNs o Attention/Transformers) dominan en cada escala temporal. 
4.  **Una Herramienta Operacional y Agnóstica:** Un pipeline (LUP) 100% automatizado que asimila datos brutos, aplica QC, inyecta huecos sintéticos para auto-evaluación y selecciona dinámicamente el mejor modelo en producción para crear un "Digital Twin" ininterrumpido.

---

## 2. Metodología: Step-by-Step

### A. Preprocesado (Data Cleaning)
1.  **Alineación Temporal (Resampling):** Los sensores muestrean a ritmos distintos. Se estandariza todo a una grilla temporal fija (e.g., 30 minutos).
2.  **Quality Control (QC):** Aplicación de banderas de calidad (Quality Flags). Se filtran valores atípicos severos (*spikes*) y rangos físicamente imposibles para esa latitud/batimetría. Todo valor fallido se convierte a `NaN`.
3.  **Interpolación Corta (Micro-gaps):** Huecos diminutos (e.g., fallos de comunicación de 1 hora) se rellenan inmediatamente con métodos deterministas (Lineal/PCHIP) para mantener un esqueleto básico continuo.

### B. Procesamiento Avanzado (Feature Engineering)
Para que los modelos de Machine Learning entiendan la física, la serie temporal se descompone y enriquece:
1.  **Decomposición STL (Seasonal-Trend decomposition using Loess):**
    *   La serie se separa en **Tendencia** (el calentamiento global o variaciones interanuales), **Estacionalidad** (el ciclo frío-calor del año) y **Residuo** (eventos climáticos súbitos).
    *   *Consideración:* Para evitar errores de memoria (OOM) en 15 años de datos, se procesa en bloques (*chunked processing*).
2.  **Climatología y Anomalías:** Se calcula cómo de diferente es el día actual respecto a la media histórica de ese mismo día del año.
3.  **Variables Cíclicas:** Se codifican la hora del día y el día del año como Senos y Cosenos para que las redes neuronales entiendan la periodicidad (sin saltos bruscos entre las 23:59 y las 00:00).
4.  **Derivadas Físicas (Inercia):** Cálculo de la velocidad de cambio (1ª derivada) y la aceleración térmica (2ª derivada).

### C. Clasificación de Huecos (Gap Categories)
Para el entrenamiento y validación auto-supervisada, se inyectan huecos intencionalmente. Se dividen en:
*   **Micro (< 6h):** Ruido de red.
*   **Small (6h - 24h):** Reinicios diarios o fallos menores.
*   **Medium (1d - 7d):** Fallos mecánicos leves u obstrucción por tormentas.
*   **Large (1w - 1m):** Fallo importante (calibración en laboratorio).
*   **X-Large (> 1m):** Rotura de cable o fin de campaña, dependientes enteramente de la meteorología exterior y Climatología.

---

## 3. Familias de Modelos Utilizados

El pipeline LUP ejecuta un benchmark masivo iterando por diferentes familias algorítmicas, desde el determinismo clásico hasta la vanguardia (*State-of-the-Art*) en *Attention Mechanisms*.

### 1. Métodos Deterministas (Baselines)
*   **Linear & PCHIP Interpolation:** Se traza una curva polinómica entre el punto A y B del hueco.
    *   *Pros:* Extremadamente rápidos, asumen inercia perfecta.
    *   *Cons:* Ignoran por completo cualquier tormenta o evento físico que ocurra *durante* el hueco. Fallan en categoráis *Medium* en adelante.

### 2. Ensemble Trees (Aproximación Termodinámica)
*   **XGBoost (Auto-Supervised Residual-Learning):** 
    *   *Principio:* Construye cientos de árboles de decisión estructurados que corrigen los errores de los anteriores.
    *   *Innovación PhD:* No predice la Temperatura bruta, sino que la descompone. Calcula una base lineal y obliga al modelo a aprender si el océano estará "por encima" o "por debajo" de esa base en función del viento y la corriente que detecten los sensores de superficie. Se ejecuta en modo *Bidireccional* (evalúa el pasado hacia el futuro y viceversa).
    *   *Pros:* Muy robusto contra *outliers*, entiende jerarquías físicas (ej. Si Rad. Solar = X Y Viento = Y -> Anomalía = Z).

### 3. Redes Neuronales Recurrentes (Memoria Secuencial)
*   **Bi-LSTM (Bidirectional Long Short-Term Memory):**
    *   *Principio:* Una RNN estructurada explícitamente para recordar secuencias a largo plazo y olvidar información inútil mediante "puertas" lógicas. "Bi-direccional" significa que la red lee la semana anterior y la semana posterior al hueco simultáneamente para encontrase en el medio.
    *   *Pros:* Excelente capturando dinámicas que dependen fuertemente del estado interior del agua (e.g. la difusión paulatina del calor).

### 4. Attention Models & Transformers (El Nuevo S.O.T.A.)
Estos modelos miran toda la serie a la vez (Atención) en lugar de paso-a-paso, buscando patrones ocultos a lo largo de los años de datos.
*   **SAITS (Self-Attention Imputation for Time Series):**
    *   Recrea la arquitectura *Transformer* (como la base de ChatGPT) pero ajustada para números continuos en vez de texto. Analiza qué puntos del pasado distante son estadísticamente relevantes para el hueco actual.
*   **ImputeFormer:**
    *   Un Transformer optimizado computacionalmente para series espaciotemporales infinitamente largas con *Low-Rank* Attention (para que no colapse la RAM).
*   **BRITS (Bidirectional Recurrent Imputation for Time Series):**
    *   Aunque es RNN, modela explícitamente el tiempo que los datos llevan desaparecidos (`time_gap_decay`). Extraordinario forzando a la red a "dudar" de sus propias predicciones contra más largo es un hueco.

### 5. Algoritmos Probabilísticos y Clásicos (Forests & ARMA)
*   **MissForest:** Un Random Forest imputador puro. Funciona iterativamente (predecir TEMP usando SAL, luego SAL usando TEMP, y así iterando hasta converger).
*   **VARMA / Auto-ARIMA:** Vector Autoregression Moving Average. El estándar econométrico. Asume linealidad pura en los cruces multivariados. Se usa para demostrar que el Deep Learning aporta verdadero valor no-lineal.

---

## 4. Conclusión Final para el Manuscrito
En tu introducción de la tesis o paper, tu narrativa es la siguiente:
*"Mientras que el despliegue de observatorios costeros de alta resolución se ha multiplicado, el aprovechamiento y calidad de esos datos se ve truncado por interrupciones crónicas. El proyecto LUP llena un vacío metodológico clave al ofrecer el primer framework multivariado totalmente estandarizado y automatizado en R/Python que evalúa sistemáticamente el salto cualitativo entre métodos estadísticos vs Deep Learning. Al introducir mecanismos de reconstrucción residual y contextualización multivariada, esta investigación no solo reduce dramáticamente la incertidumbre en los *gaps* meso-escalares (semanas a meses), sino que proporciona a la comunidad oceanográfica una herramienta *open-source* lista para integrar en servicios operacionales (Digital Twins) sin necesidad de re-entrenos manuales continuos."*
