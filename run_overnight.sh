#!/bin/bash

# ==============================================================================
# OBSEA OVERNIGHT PRODUCTION PIPELINE (2022-2023)
# ==============================================================================
# Este script está diseñado para ejecutarse durante 11-12 horas y dejar todo 
# el dataset de los últimos 2 años listo y graficado para mañana.

LOG_FILE="overnight_production_$(date +%Y%m%d_%H%M%S).log"

echo "=== INICIANDO PIPELINE NOCTURNO OBSEA ===" | tee -a $LOG_FILE
echo "Inicio: $(date)" | tee -a $LOG_FILE

# 1. FASE DE PREPROCESAMIENTO (15 Años Golden)
# Usa el dataset fusionado (Historical + AWAC + API) para máxima fidelidad.
echo "Fase 1: Preparando 15 años de datos Golden (--no-cache)..." | tee -a $LOG_FILE
python3 main_obsea.py --mode ingest --csv-input output_lup/data/OBSEA_full_golden_dataset.csv --no-cache 2>&1 | tee -a $LOG_FILE

# 2. FASE DE PLOTTING (Originales)
# Genera gráficos de los gaps en los 15 años completos.
echo "Fase 2: Generando gráficos de gaps originales del dataset Golden..." | tee -a $LOG_FILE
python3 main_obsea.py --mode plot --csv-input output_lup/data/OBSEA_full_golden_dataset.csv 2>&1 | tee -a $LOG_FILE

# 3. FASE DE BENCHMARK (PhD Quality Control)
# Simula gaps en 150 días para validar el sistema de Climatología.
echo "Fase 3: Ejecutando Benchmark de 150 días con Climatología..." | tee -a $LOG_FILE
python3 main_obsea.py --mode benchmark --csv-input output_lup/data/OBSEA_full_golden_dataset.csv --limit 150 2>&1 | tee -a $LOG_FILE

# 4. FASE DE PRODUCCIÓN (Full Reconstruction 2009-2025)
# Aplica la interpolación Scale-Aware + Seasonal Fallback.
echo "Fase 4: Ejecutando Reconstrucción Final de 15 años..." | tee -a $LOG_FILE
python3 main_obsea.py --mode production --csv-input output_lup/data/OBSEA_full_golden_dataset.csv 2>&1 | tee -a $LOG_FILE

# 5. FASE DE VERIFICACIÓN FINAL
# Genera los gráficos multivariante para el anexo de la tesis.
echo "Fase 5: Generando visualizaciones finales (15 años)..." | tee -a $LOG_FILE
python3 scripts/plot_production_groups.py 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    echo "✓ Fase 2 Completada. Gráficos listos en output_lup/plots_production_groups/" | tee -a $LOG_FILE
else
    echo "✗ Error en Fase 2." | tee -a $LOG_FILE
fi

echo "=== PIPELINE FINALIZADO CON ÉXITO ===" | tee -a $LOG_FILE
echo "Fin: $(date)" | tee -a $LOG_FILE
echo "Dataset final en: output_lup/data/OBSEA_final_interpolated.csv" | tee -a $LOG_FILE
