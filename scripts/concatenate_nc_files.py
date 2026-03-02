import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
base_path = "/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts/OBSEA_CTVG_Vantage_Pro2_30min_nc/"
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)
output_dir = "exported_data"
os.makedirs(output_dir, exist_ok=True)

pickle_file = os.path.join(output_dir, "df_all.pkl")  # Para guardar/recargar DataFrame
csv_file = os.path.join(output_dir, "OBSEA_OBSEA_AWAC_waves_full_nc_RAW_DT.csv")


# Flag: True si quieres recargar de los .nc, False si ya tienes el pickle
already_loaded = False

# -----------------------------
# CARGAR Y CONCATENAR ARCHIVOS
# -----------------------------
if not already_loaded or not os.path.exists(pickle_file):
    print("Cargando y concatenando archivos .nc...")

    all_files = sorted(glob.glob(os.path.join(base_path, "*", "*.nc")))
    print(f"Encontrados {len(all_files)} archivos.")

    all_variables = set()
    df_list = []

    for f in all_files:
        try:
            ds = xr.open_dataset(f)
            print(f"\nArchivo: {f}")
            print("Variables:", list(ds.data_vars))
            all_variables.update(ds.data_vars)

            df = ds.to_dataframe().reset_index()

            # Detectar la columna de tiempo automáticamente
            time_col = None
            for c in df.columns:
                if 'time' in c.lower():
                    time_col = c
                    break
            if time_col is None:
                print("No se encontró columna de tiempo, se salta este archivo.")
                continue

            # Convertir a datetime
            df['TIME'] = pd.to_datetime(df[time_col], errors='coerce', unit='s', origin='unix')

            df_list.append(df)

        except Exception as e:
            print(f"Error con {f}: {e}")
            continue

    # Concatenar todos los DataFrames
    df_all = pd.concat(df_list, ignore_index=True)
    print("\nLista completa de variables encontradas:", all_variables)

    # Guardar pickle para no repetir
    df_all.to_pickle(pickle_file)
    print(f"DataFrame concatenado guardado en: {pickle_file}")

else:
    print("Cargando DataFrame desde pickle...")
    df_all = pd.read_pickle(pickle_file)
    all_variables = set(df_all.columns) - {'TIME'}
    print("Carga completada desde pickle.")

# -----------------------------
# FUNCIONES PARA PLOTS
# -----------------------------
def plot_variable(df, var, save_dir):
    if var not in df.columns:
        return
    plt.figure(figsize=(14,5))
    plt.plot(df['TIME'], df[var], color='tab:blue', linewidth=1)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel(var, fontsize=12)
    plt.title(f"{var} vs Time", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    # Guardar el plot
    filename = os.path.join(save_dir, f"{var}_vs_time.png")
    plt.savefig(filename, dpi=300)
    plt.close()

# Plotear todas las variables y guardar
for var in all_variables:
    plot_variable(df_all, var, plot_dir)

print(f"Todos los plots se han guardado en la carpeta '{plot_dir}'.")

# -----------------------------
# EXPORTAR CSV
# -----------------------------

input()

cols_of_interest = ['PSAL_STD', 'TEMP_STD', 'SVEL_QC', 'SVEL', 'SVEL_STD', 'PRES', 'PRES_QC', 'PRES_STD', 'PSAL_QC', 'LONGITUDE_QC', 'CNDC_QC', 'PSAL', 'CNDC_STD', 'TEMP_QC', 'TEMP', 'LATITUDE_QC', 'DEPTH_QC', 'CNDC']

cols_exist = [c for c in cols_of_interest if c in df_all.columns]

df_export = df_all[cols_exist]

measurement_cols = [c for c in cols_exist if c != 'TIME']
df_export = df_export.dropna(subset=measurement_cols, how='all')

df_export.to_csv(csv_file, index=False)

 
print(f"Archivo CSV guardado en: {csv_file}")
