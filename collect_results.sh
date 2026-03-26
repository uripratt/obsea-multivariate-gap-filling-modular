#!/bin/bash

# Script para recolectar y empaquetar resultados de ejecución (Benchmark + Producción)
# Uso: ./collect_results.sh [nombre_opcional]

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME=${1:-"audit_run_$TIMESTAMP"}
EXPORT_DIR="exports/$RUN_NAME"

echo "🚀 Iniciando recolección de resultados en: $EXPORT_DIR"

# 1. Crear estructura de carpetas
mkdir -p "$EXPORT_DIR/plots"
mkdir -p "$EXPORT_DIR/data"
mkdir -p "$EXPORT_DIR/logs"

# 2. Localizar y copiar el último log
LATEST_LOG=$(ls -t *.log 2>/dev/null | head -n 1)
if [ -n "$LATEST_LOG" ]; then
    echo "  📄 Copiando log más reciente: $LATEST_LOG"
    cp "$LATEST_LOG" "$EXPORT_DIR/logs/"
else
    echo "  ⚠️ No se encontraron archivos .log"
fi

# 3. Recopilar resultados de Benchmark (CSV y Gráficos)
BENCH_DIR="output_lup/benchmarks"
if [ -d "$BENCH_DIR" ]; then
    echo "  📊 Copiando métricas de benchmark..."
    cp "$BENCH_DIR/interpolation_comparison.csv" "$EXPORT_DIR/data/" 2>/dev/null
    cp "$BENCH_DIR/benchmark_results.png" "$EXPORT_DIR/plots/" 2>/dev/null
    
    if [ -d "$BENCH_DIR/gap_examples" ]; then
        echo "  🖼️ Copiando ejemplos visuales de gaps (comparativas)..."
        cp "$BENCH_DIR/gap_examples/comparison_"*".png" "$EXPORT_DIR/plots/" 2>/dev/null
        cp "$BENCH_DIR/gap_examples/residuals_"*".png" "$EXPORT_DIR/plots/" 2>/dev/null
        
        # Opcional: solo copiar una muestra de los mejores modelos si hay demasiados
        # cp "$BENCH_DIR/gap_examples/gap_"*".png" "$EXPORT_DIR/plots/" 2>/dev/null
    fi
else
    echo "  ⚠️ No se encontró la carpeta de benchmarks"
fi

# 4. Recopilar resultados de Producción (si existen)
PROD_DIR="output_lup/production"
if [ -d "$PROD_DIR" ]; then
    echo "  💾 Copiando dataset final de producción (cabecera)..."
    # Solo copiamos las primeras 1000 líneas para no saturar el zip si el CSV es gigante
    # O el CSV completo si prefieres
    LATEST_PROD=$(ls -t "$PROD_DIR"/*.csv 2>/dev/null | head -n 1)
    if [ -n "$LATEST_PROD" ]; then
        cp "$LATEST_PROD" "$EXPORT_DIR/data/"
    fi
fi

# 5. Generar Mini-Resumen para terminal
echo "------------------------------------------------" > "$EXPORT_DIR/summary_quickview.txt"
echo "RESUMEN DE EJECUCIÓN: $RUN_NAME" >> "$EXPORT_DIR/summary_quickview.txt"
echo "Fecha: $(date)" >> "$EXPORT_DIR/summary_quickview.txt"
echo "------------------------------------------------" >> "$EXPORT_DIR/summary_quickview.txt"
if [ -f "$EXPORT_DIR/data/interpolation_comparison.csv" ]; then
    echo "TOP RECOMENDACIONES (RMSE):" >> "$EXPORT_DIR/summary_quickview.txt"
    head -n 20 "$EXPORT_DIR/data/interpolation_comparison.csv" | cut -d',' -f1,2,3,6 >> "$EXPORT_DIR/summary_quickview.txt"
fi

# 6. Empaquetar todo en un .tar.gz
TAR_FILE="exports/${RUN_NAME}.tar.gz"
tar -czf "$TAR_FILE" -C "exports" "$RUN_NAME"

echo "------------------------------------------------"
echo "✅ PROCESO COMPLETADO"
echo "📂 Carpeta: $EXPORT_DIR"
echo "📦 Archivo para descargar: $TAR_FILE"
echo "------------------------------------------------"
echo "Instrucción para JupyterHub: Descarga el archivo $TAR_FILE"
echo "haciendo clic derecho sobre él en el navegador de archivos."
