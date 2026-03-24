#!/bin/bash

# ==============================================================================
# OBSEA OVERNIGHT PRODUCTION PIPELINE (2022-2023)
# ==============================================================================
# Este script está diseñado para ejecutarse durante 11-12 horas y dejar todo 
# el dataset de los últimos 2 años listo y graficado para mañana.

LOG_FILE="overnight_production_$(date +%Y%m%d_%H%M%S).log"

echo "=== INICIANDO PIPELINE NOCTURNO OBSEA ===" | tee -a $LOG_FILE
echo "Inicio: $(date)" | tee -a $LOG_FILE

# 1. FASE DE INGESTA (2 Años)
# Descarga los datos crudos y prepara la caché local. (~11 horas)
echo "Fase 1: Ingesta Cruda 2022-2023 (--no-cache)..." | tee -a $LOG_FILE
python3 main_obsea.py --mode ingest --start 2022-01-01 --end 2023-12-31 --no-cache 2>&1 | tee -a $LOG_FILE

# 2. FASE DE PLOTTING (Originales)
# Genera gráficos de los gaps reales antes de interpolar. (~2 mins)
echo "Fase 2: Generando gráficos de gaps originales..." | tee -a $LOG_FILE
python3 main_obsea.py --mode plot 2>&1 | tee -a $LOG_FILE

# 3. FASE DE BENCHMARK (Control de Calidad)
# Simula gaps en 100 días para validar métricas (RMSE/MAE). (~45 mins)
echo "Fase 3: Ejecutando Benchmark de 100 días sobre la marcha..." | tee -a $LOG_FILE
python3 main_obsea.py --mode benchmark --limit 100 2>&1 | tee -a $LOG_FILE

# 4. FASE DE PRODUCCIÓN (El "Gold Standard")
# Aplica la interpolación Scale-Aware a los 2 años. (~1 hora sobre caché)
echo "Fase 4: Ejecutando Producción Scale-Aware sobre caché..." | tee -a $LOG_FILE
python3 main_obsea.py --mode production 2>&1 | tee -a $LOG_FILE

# 5. FASE DE VERIFICACIÓN FINAL
# Genera los gráficos multivariante sofisticados. (~5 mins)
echo "Fase 5: Generando visualizaciones finales por grupos..." | tee -a $LOG_FILE
python3 scripts/plot_production_groups.py 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    echo "✓ Fase 2 Completada. Gráficos listos en output_lup/plots_production_groups/" | tee -a $LOG_FILE
else
    echo "✗ Error en Fase 2." | tee -a $LOG_FILE
fi

echo "=== PIPELINE FINALIZADO CON ÉXITO ===" | tee -a $LOG_FILE
echo "Fin: $(date)" | tee -a $LOG_FILE
echo "Dataset final en: output_lup/data/OBSEA_final_interpolated.csv" | tee -a $LOG_FILE
