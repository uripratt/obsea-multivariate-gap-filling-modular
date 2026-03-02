import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
base_path = "/home/uripratt/Documents/PhD/OBSEA_data/CTD/OBSEA_AWAC_waves_full_nc/"
output_dir = "exported_data"
plot_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

csv_file = os.path.join(output_dir, "OBSEA_OBSEA_AWAC_waves_full_nc_RAW_shiat.csv")

cols_requested = [
   'CNDC', 'CNDC_QC', 'CNDC_STD', 'PSAL', 'PSAL_QC', 'PSAL_STD', 'TEMP', 'TEMP_QC',
     'TEMP_STD', 'SVEL', 'SVEL_QC', 'SVEL_STD', 'PRES', 'PRES_QC', 'PRES_STD', 'LATITUDE_QC', 'LONGITUDE_QC', 'DEPTH_QC']

# -----------------------------
# 0. INSPECCIÓN INICIAL
# -----------------------------
all_files = sorted(glob.glob(os.path.join(base_path, "*", "*.nc")))
if not all_files:
    print("❌ No hay archivos .nc")
    exit()

print("="*50)
with xr.open_dataset(all_files[0]) as ds_inspect:
    print(f"Variables detectadas: {list(ds_inspect.data_vars)}")
    # Ver cómo viene el tiempo en el NetCDF original
    t_var = next((c for c in ds_inspect.coords if 'time' in c.lower()), None)
    if t_var:
        print(f"Ejemplo tiempo RAW en NC: {ds_inspect[t_var].values[0]}")
print("="*50 + "\n")

# -----------------------------
# 1. EXPORTACIÓN CSV (CORRIGIENDO EL TIEMPO)
# -----------------------------
valid_files = []
if os.path.exists(csv_file): os.remove(csv_file)

print(f"--- PASO 1: Generando CSV ---")
first_file = True

for i, f in enumerate(all_files):
    try:
        with xr.open_dataset(f) as ds:
            valid_files.append(f)
            df = ds.to_dataframe().reset_index()
            
            raw_time_col = next((c for c in df.columns if 'time' in c.lower()), None)
            
            if raw_time_col:
                # CAMBIO CLAVE: Especificar unit='s' para que cuente SEGUNDOS
                # origin='unix' es por defecto 1970-01-01
                df['TIME_CONVERTED'] = pd.to_datetime(df[raw_time_col], unit='s', errors='coerce')
                
                if first_file:
                    print(f"Verificación de conversión para {os.path.basename(f)}:")
                    print(f" -> Valor RAW: {df[raw_time_col].iloc[0]}")
                    print(f" -> Valor CONVERTIDO: {df['TIME_CONVERTED'].iloc[0]}")
                    print("-" * 30)

                cols_present = [c for c in cols_requested if c in df.columns]
                df_export = df[['TIME_CONVERTED'] + cols_present].rename(columns={'TIME_CONVERTED': 'TIME'})

                # Guardar al CSV
                df_export.to_csv(csv_file, mode='a', header=first_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
                first_file = False
        
        if (i + 1) % 5 == 0:
            print(f"Progreso: {i + 1}/{len(all_files)} archivos procesados...")
                
    except Exception as e:
        print(f"⚠️ Error en {os.path.basename(f)}: {e}")

# -----------------------------
# 2. GENERACIÓN DE PLOTS
# -----------------------------
print(f"\n--- PASO 2: Generando Plots ---")

try:
    # Nota: Xarray suele manejar las unidades de tiempo automáticamente en open_mfdataset
    # si los archivos tienen el atributo 'units' correcto. 
    ds_plots = xr.open_mfdataset(valid_files, combine='by_coords', chunks={'time': 2000})

    for var in cols_requested:
        if var in ds_plots.data_vars:
            print(f"Graficando {var}...")
            plt.figure(figsize=(14, 5))
            ds_plots[var].plot(linewidth=0.5)
            plt.title(f"Serie Temporal: {var}")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f"{var}_vs_time.png"), dpi=150)
            plt.close()
except Exception as e:
    print(f"❌ Error en plots: {e}")

print("\n✅ Proceso completado. Revisa el CSV y los gráficos.")


