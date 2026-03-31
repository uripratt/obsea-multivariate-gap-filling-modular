# Estructura del Proyecto OBSEA - Análisis Oceanográfico Multivariante

Este documento explica la organización, arquitectura y flujo de datos del ecosistema de scripts y la aplicación web para el análisis de datos de la plataforma OBSEA.

---

## 1. Arquitectura General (v1.0 Legacy vs v2.0 Modular)

El proyecto ha evolucionado hacia una arquitectura modular para mayor robustez científica y operativa:

-   **Version 1.0 (Legacy)**: Basada en scripts monolíticos (`lup_data_obsea_analysis_jupyterhub.py`) que gestionan todo el flujo secuencialmente. Útil para investigación rápida en cuadernos Jupyter.
-   **Version 2.0 (Modular - `obsea_pipeline/`)**: Un paquete Python estructurado que separa la ingesta, QC, modelos y visualización. Optimizado para producción, benchmarking automatizado y forecasting.
-   **Plataforma de Visualización (Web Dashboard)**: Interfaz profesional que consume los resultados de ambas versiones.

---

## 2. Nueva Estructura Modular (v2.0)

El corazón de la nueva versión es el paquete `obsea_pipeline/`:

-   `main_obsea.py`: Nuevo orquestador CLI avanzado (soporta `--mode ingest/benchmark/production/plot`).
-   `obsea_pipeline/ingestion/`: Conector recursivo a la API STA v1.1.
-   `obsea_pipeline/models/`: Wrappers inteligentes para 20+ modelos de imputación.
-   `obsea_pipeline/qc/`: Implementación modular de estándares QARTOD.
-   `obsea_pipeline/gaps/`: Motor de detección *Scale-Aware*.

---

## 2. Flujo de Datos (Pipeline)

El camino que siguen los datos desde el sensor hasta la web es el siguiente:

1.  **Ingesta**: Descarga y concatenación de archivos NetCDF (.nc) de los diferentes instrumentos (CTD, AWAC, Met) mediante `concatenate_nc_files.py`.
2.  **Procesamiento Central (`lup_data_obsea_analysis.py`)**: 
    *   Carga los CSVs generados.
    *   Aplica **Control de Calidad (QC)** siguiendo estándares QARTOD.
    *   Realiza el re-muestreo a una cuadrícula temporal común (30 min).
    *   Calcula **Variables Oceanográficas** (Densidad Poteencial, Estratificación, Estrés del Viento, etc.).
    *   Analiza y clasifica los **Gaps** (huecos de datos).
    *   Ejecuta el **Benchmarking** para comparar métodos de interpolación.
3.  **Exportación**: Los resultados se guardan en la carpeta `webapp/tables/`, `webapp/data/` y `webapp/figures/`, que son consumidos por la interfaz web.

---

## 3. Mapa de Directorios

### Raíz del Proyecto
-   `lup_data_obsea_analysis.py`: **Script principal**. Orquestador de todo el análisis científico.
-   `multivariate_dl_gap_filling.py`: Implementación de modelos Deep Learning (Bi-LSTM) para interpolación avanzada.
-   `PIPELINE_DOCUMENTATION.md`: Documentación detallada de las funciones científicas del script principal.
-   `exported_data/`: Contiene los CSVs intermedios y finales del procesamiento.
-   `OBSEA_*_nc/`: Carpetas con los archivos originales en formato NetCDF.
-   `webapp/`: Directorio de la aplicación web.

### Aplicación Web (`webapp/`)
-   `index.html`: Estructura de la interfaz (Dashboard, Instrumentos, Gaps, Métodos, Oceanografía).
-   `css/`: Estilos visuales (diseño oscuro, premium).
-   `js/`: Lógica de la aplicación:
    *   `app.js`: Coordinador principal. Maneja la navegación y la carga de datos.
    *   `charts.js`: Definición de gráficos Plotly (series temporales, comparativas, histogramas).
    *   `dataLoader.js`: Módulo para leer CSVs de forma eficiente.
    *   `methodAnalysis.js`: Lógica de interpolación en el cliente y gestión de colores de modelos.
    *   `gapTimeline.js`: Visualización específica de la línea de tiempo de huecos.
-   `data/`, `tables/`, `figures/`: Enlaces simbólicos a los resultados del pipeline de Python.

---

## 4. Componentes Clave

### Scripts de Entrenamiento y Visualización (Python)
-   **Análisis Estadístico**: `CTD_StadisticalAnalysis.py` realiza un análisis profundo de las variables del CTD.
-   **Visualización de Gaps**: Funciones dentro de `lup_data_obsea_analysis.py` generan heatmaps de disponibilidad de datos para más de 30 variables simultáneamente.
-   **Benchmarking**: El sistema simula huecos artificiales para calcular RMSE, MAE y R² de cada método (Lineal, Splines, VARMA, Bi-LSTM).

### Módulo de Interpolación (Web)
La sección de **Interpolación** en la web permite:
1.  **Comparativa de Algoritmos**: Gráfico de barras que muestra el rendimiento de cada modelo.
2.  **Simulador de Reconstrucción**: Permite elegir una variable y un método para ver "en vivo" cómo el algoritmo rellena los huecos existentes en el log original.
3.  **Métricas Científicas**: Tabla detallada con los errores de cada método por categoría de gap (micro, short, medium).

---

## 5. Resumen de Archivos de Datos Principales

| Archivo | Función |
| :--- | :--- |
| `OBSEA_multivariate_30min.csv` | Dataset unificado y limpio con todas las variables. |
| `interpolation_comparison.csv` | Resultados del benchmark (RMSE, R2) para la web. |
| `gap_summary.csv` | Resumen estadístico de los huecos por variable. |
| `descriptive_statistics.csv` | Estadísticas descriptivas (media, std, percentiles) del dataset. |
| `interpolation_tracking.csv` | Trazabilidad que indica qué método se usó para rellenar cada punto del dataset final. |
