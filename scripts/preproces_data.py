# import rich as r
# import os
# import pandas as pd 
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# # --- Cargar datos ---
# df = pd.read_csv('/home/uripratt/Documents/PhD/OBSEA_data/CTD/exported_data/OBSEA_CTD_all_years.csv')

# # Seleccionar columnas de interés y convertir TIME
# qc_temp = df[['TIME', 'TEMP', 'TEMP_QC']]
# qc_temp['TIME'] = pd.to_datetime(qc_temp['TIME'])

# # Filtrar solo valores con QC=1
# qc_temp_clean = qc_temp[qc_temp['TEMP_QC'] == 1].set_index('TIME')

# # --- Detectar gaps ---
# time_diff = qc_temp_clean.index.to_series().diff().dropna()
# expected_freq = pd.Timedelta(minutes=30)
# gaps = time_diff[time_diff > expected_freq]

# # Crear DataFrame con información de gaps
# gaps_df = pd.DataFrame({
#     'gap_start': gaps.index - gaps.values,
#     'gap_end': gaps.index,
#     'duration_sec': gaps.dt.total_seconds()
# })

# gaps_df['duration_min'] = gaps_df['duration_sec'] / 60
# gaps_df['duration_days'] = gaps_df['duration_sec'] / (60*60*24)
# gaps_df['duration_hours'] = gaps_df['duration_sec'] / (60*60)

# # Clasificar gaps
# gaps_df['category'] = pd.cut(
#     gaps_df['duration_days'],
#     bins=[0, 1/24, 12/24, 1, 1000],  # <1h, 1h-1d, >1d
#     labels=['minimal','small', 'medium', 'large']
# )

# gap_counts = gaps_df['category'].value_counts().sort_index()


# # Filtrar gaps menores a 24 horas
# gaps_under_24h = gaps_df[gaps_df['duration_hours'] < 24]

# # # Histograma de duración de gaps < 24 horas
# # plt.figure(figsize=(10,5))
# # plt.hist(gaps_under_24h['duration_hours'], bins=24, color='tab:blue', edgecolor='black')
# # plt.xlabel("Duración del gap (horas)", fontsize=12)
# # plt.ylabel("Frecuencia", fontsize=12)
# # plt.title("Histograma de duración de gaps < 24h", fontsize=14, fontweight='bold')
# # plt.grid(True, linestyle='--', alpha=0.5)
# # plt.tight_layout()
# # plt.show()


# ###### GAPS PLOT ########

# labels = gaps_df['category'].cat.categories
# # --- Colors for gap categories ---
# gap_colors = {
#     'minimal': 'green',
#     'small': 'yellow',
#     'medium': 'orange',
#     'large': 'red'
# }

# # --- Duration intervals for legend (in hours/days) ---
# gap_intervals = {
#     'minimal': '<1h',
#     'small': '1-12h',
#     'medium': '12h-1d',
#     'large': '>1d'
# }

# # --- Figure ---
# plt.figure(figsize=(14,5))
# plt.plot(qc_temp_clean.index, qc_temp_clean['TEMP'], color='tab:blue', linewidth=1, label='TEMP (QC=1)')

# # --- Highlight gaps by category ---
# for _, row in gaps_df.iterrows():
#     cat = row['category']
#     if pd.notna(cat):
#         plt.axvspan(row['gap_start'], row['gap_end'], color=gap_colors[cat], alpha=0.3)

# plt.xlabel("Time", fontsize=12)
# plt.ylabel("Temperature [°C]", fontsize=12)
# plt.title("Time series of TEMP (QC=1) with gaps by category", fontsize=14, fontweight='bold')
# plt.grid(True, linestyle='--', alpha=0.5)

# # --- Calculate percentages for legend ---
# gap_percentages = gaps_df['category'].value_counts(normalize=True).sort_index() * 100

# # --- Create legend with intervals and percentages ---
# legend_elements = [
#     Patch(
#         facecolor=gap_colors[cat],
#         alpha=0.3,
#         label=f"{cat} ({gap_intervals[cat]}, {gap_percentages.get(cat,0):.1f}%)"
#     ) 
#     for cat in labels
# ]

# # Add TEMP line to legend
# legend_elements.append(plt.Line2D([0], [0], color='tab:blue', lw=1, label='TEMP (QC=1)'))

# plt.legend(handles=legend_elements)
# plt.tight_layout()
# plt.show()

# r.print(gaps)

# ####### MINIMAL AND SMALL GAPS INTERPOALTE ########


# # Seleccionar gaps a interpolar


# plt.figure(figsize=(14,5))

# # Serie original con todos los gaps
# plt.plot(qc_temp_clean.index, qc_temp_clean['TEMP'], color='blue', linewidth=1, label='Original TEMP (all gaps)')

# # Serie con interpolación solo en gaps minimal+small
# plt.plot(qc_temp_clean_interp.index, qc_temp_clean_interp['TEMP'], color='red', linewidth=1, label='TEMP with minimal+small gaps interpolated')

# plt.xlabel("Time")
# plt.ylabel("Temperature [°C]")
# plt.title("Comparison of original TEMP and interpolated TEMP (minimal+small gaps)")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()


import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rich as r

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv(
    '/home/uripratt/Documents/PhD/OBSEA_data/CTD/exported_data/OBSEA_CTD_all_years.csv'
)

print(df.columns)
input()
qc_temp = df[['TIME', 'TEMP', 'TEMP_QC']].copy()
qc_temp['TIME'] = pd.to_datetime(qc_temp['TIME'])

# QC = 1 only
qc_temp_clean = (
    qc_temp[qc_temp['TEMP_QC'] == 1]
    .set_index('TIME')
    .sort_index()
)

# Eliminar duplicados de timestamp
qc_temp_clean = qc_temp_clean.groupby(qc_temp_clean.index).mean()

# =========================================================
# 2. GAP DETECTION
# =========================================================
expected_freq = pd.Timedelta(minutes=30)
time_diff = qc_temp_clean.index.to_series().diff()

gaps = time_diff[time_diff > expected_freq]

gaps_df = pd.DataFrame({
    'gap_start': gaps.index - gaps.values,
    'gap_end': gaps.index,
    'duration_sec': gaps.dt.total_seconds()
})

gaps_df['duration_min'] = gaps_df['duration_sec'] / 60
gaps_df['duration_hours'] = gaps_df['duration_sec'] / 3600
gaps_df['duration_days'] = gaps_df['duration_sec'] / (3600 * 24)

# =========================================================
# 3. GAP CLASSIFICATION
# =========================================================
gaps_df['category'] = pd.cut(
    gaps_df['duration_days'],
    bins=[0, 1/24, 12/24, 1, 1000],
    labels=['minimal', 'small', 'medium', 'large']
)

r.print(gaps_df['category'].value_counts().sort_index())

# =========================================================
# 4. GAP OVERVIEW PLOT
# =========================================================
gap_colors = {
    'minimal': 'green',
    'small': 'yellow',
    'medium': 'orange',
    'large': 'red'
}

gap_intervals = {
    'minimal': '<1h',
    'small': '1–12h',
    'medium': '12h–1d',
    'large': '>1d'
}

plt.figure(figsize=(14, 5))
plt.plot(
    qc_temp_clean.index,
    qc_temp_clean['TEMP'],
    color='tab:blue',
    lw=1,
    label='TEMP (QC=1)'
)

for _, row in gaps_df.iterrows():
    if pd.notna(row['category']):
        plt.axvspan(
            row['gap_start'],
            row['gap_end'],
            color=gap_colors[row['category']],
            alpha=0.3
        )

gap_percentages = gaps_df['category'].value_counts(normalize=True) * 100

legend_elements = [
    Patch(
        facecolor=gap_colors[cat],
        alpha=0.3,
        label=f"{cat} ({gap_intervals[cat]}, {gap_percentages.get(cat,0):.1f}%)"
    )
    for cat in gaps_df['category'].cat.categories
]

legend_elements.append(
    plt.Line2D([0], [0], color='tab:blue', lw=1, label='TEMP (QC=1)')
)

plt.legend(handles=legend_elements)
plt.title("TEMP time series with gap categories")
plt.xlabel("Time")
plt.ylabel("Temperature [°C]")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =========================================================
# 5. BUILD REGULAR TIME SERIES (30 min)
# =========================================================
full_index = pd.date_range(
    start=qc_temp_clean.index.min(),
    end=qc_temp_clean.index.max(),
    freq='30min'
)

base = qc_temp_clean.reindex(full_index)

# =========================================================
# 6. BLOCK MEDIUM + LARGE GAPS
# =========================================================
blocked_idx = []

for _, row in gaps_df[gaps_df['category'].isin(['medium', 'large'])].iterrows():
    blocked_idx.extend(
        pd.date_range(row['gap_start'], row['gap_end'], freq='30min')
    )

blocked_idx = pd.DatetimeIndex(blocked_idx)

# =========================================================
# 7. INTERPOLATE ONLY MINIMAL + SMALL
# =========================================================
interp = base.copy()

# Interpolación temporal
interp['TEMP'] = interp['TEMP'].interpolate(method='time')

# Re-imponer NaN en gaps medium + large
interp.loc[blocked_idx, 'TEMP'] = pd.NA

# Máscara clara de valores interpolados
interp['interpolated'] = base['TEMP'].isna() & interp['TEMP'].notna()

r.print(
    f"Interpolated points: {interp['interpolated'].sum()}"
)

# =========================================================
# =========================================================
# 8. PLOT ONLY INTERPOLATED POINTS + MEDIUM/LARGE GAPS
# =========================================================
plt.figure(figsize=(14, 5))

# --- Serie original ---
plt.plot(
    base.index,
    base['TEMP'],
    color='tab:blue',
    lw=1,
    label='Original TEMP'
)

# --- Puntos interpolados (solo minimal + small) ---
plt.scatter(
    interp.index[interp['interpolated']],
    interp.loc[interp['interpolated'], 'TEMP'],
    color='red',
    s=10,
    label='Interpolated (minimal + small)'
)

# --- Bandas para gaps MEDIUM y LARGE ---
for _, row in gaps_df[gaps_df['category'].isin(['medium', 'large'])].iterrows():
    plt.axvspan(
        row['gap_start'],
        row['gap_end'],
        color=gap_colors[row['category']],
        alpha=0.3
    )

# --- Labels & layout ---
plt.xlabel("Time")
plt.ylabel("Temperature [°C]")
plt.title("Interpolated points (minimal + small) with medium/large gaps highlighted")
plt.grid(True, linestyle='--', alpha=0.5)

# --- Legend manual ---
legend_elements = [
    plt.Line2D([0], [0], color='tab:blue', lw=1, label='Original TEMP'),
    plt.Line2D([0], [0], marker='o', color='red', linestyle='', markersize=6,
               label='Interpolated (minimal + small)'),
    Patch(facecolor=gap_colors['medium'], alpha=0.3, label='Medium gaps (12h–1d)'),
    Patch(facecolor=gap_colors['large'], alpha=0.3, label='Large gaps (>1d)')
]

plt.legend(handles=legend_elements)
plt.tight_layout()
plt.show()


# =========================================================
# 9. FINAL COMPARISON PLOT
# =========================================================
plt.figure(figsize=(14, 5))

plt.plot(
    base.index,
    base['TEMP'],
    color='tab:blue',
    lw=1,
    label='Original TEMP'
)

plt.plot(
    interp.index,
    interp['TEMP'],
    color='red',
    lw=1,
    alpha=0.7,
    label='TEMP with minimal+small gaps interpolated'
)

plt.xlabel("Time")
plt.ylabel("Temperature [°C]")
plt.title("TEMP time series with controlled gap interpolation")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


############ SAVE TEMPORAL SERIE INTERPOLATED ###############

# Copia final para exportar
final_ts = interp.copy()

# Opcional: añadir una QC propia
# 1 = observado (QC original)
# 2 = interpolado (minimal + small)
final_ts['TEMP_QC_FINAL'] = 1
final_ts.loc[final_ts['interpolated'], 'TEMP_QC_FINAL'] = 2

output_path = (
    "/home/uripratt/Documents/PhD/OBSEA_data/CTD/"
    "OBSEA_CTD_TEMP_interpolated_minimal_small.csv"
)

final_ts[['TEMP', 'TEMP_QC_FINAL']].to_csv(
    output_path,
    index=True,
    date_format="%Y-%m-%d %H:%M:%S"
)

print(f"Saved final interpolated time series to:\n{output_path}")
