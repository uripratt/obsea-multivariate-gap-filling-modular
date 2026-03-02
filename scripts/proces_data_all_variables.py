import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rich as r
# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv(
    '/home/uripratt/Documents/PhD/OBSEA_data/CTD/exported_data/OBSEA_CTD_all_years.csv'
)
df['TIME'] = pd.to_datetime(df['TIME'])

VARS = ['TEMP', 'PSAL', 'SVEL', 'PRES', 'TEMP_STD']
FREQ = '30min'

qc_df = df[['TIME'] + VARS]
qc_df = qc_df.set_index('TIME')
r.print('df:',df)
r.print('qc_df:',qc_df)
# =========================================================
# 2. REGULAR TIME GRID (30 min)
# =========================================================
time_index = pd.date_range(
    start=qc_df.index.min(),
    end=qc_df.index.max(),
    freq=FREQ
)

base = qc_df.groupby(qc_df.index).mean()
base = base.reindex(time_index)

r.print('--------------------------------------2. REGULAR TIME GRID (30 min)--------------------------------------', base)
# =========================================================
# 3. QUALITY CONTROL FUNCTIONS
# =========================================================
def qc_range(series, min_val, max_val):
    qc_flag = pd.Series(0, index=series.index)
    qc_flag[(series < min_val) | (series > max_val)] = 9
    return qc_flag

def qc_rog(series, max_change):
    qc_flag = pd.Series(0, index=series.index)
    diff = series.diff().abs()
    qc_flag[diff > max_change] = 3
    return qc_flag

def qc_flat(series, n_steps=5):
    qc_flag = pd.Series(0, index=series.index)
    flat = series.rolling(n_steps).apply(lambda x: x.max() - x.min(), raw=True)
    qc_flag[flat == 0] = 4
    return qc_flag

r.print('--------------------------------------3. QUALITY CONTROL FUNCTIONS--------------------------------------')

# =========================================================
# 4. APPLY QC TO EACH VARIABLE
# =========================================================
qc_flags = pd.DataFrame(index=base.index)
r.print('qc_flags:', qc_flags)
for var in VARS:
    if var == 'TEMP':
        flags = qc_range(base[var], -2, 35)
        flags |= qc_rog(base[var], 1)
        flags |= qc_flat(base[var], n_steps=20)
    elif var == 'PSAL':
        flags = qc_range(base[var], 35, 39)
        flags |= qc_rog(base[var], 3)
        flags |= qc_flat(base[var], n_steps=20)
    elif var == 'SVEL':
        flags = qc_range(base[var], 1490, 1560)
        flags |= qc_rog(base[var], 2)
        flags |= qc_flat(base[var], n_steps=5)
    elif var == 'PRES':
        flags = qc_range(base[var], 18.5, 20.5)
        flags |= qc_rog(base[var], 1.5)
    elif var == 'TEMP_STD':
        flags = qc_range(base[var], 0, 5)
    qc_flags[var + '_QC'] = flags
r.print('--------------------------------------4. APPLY QC TO EACH VARIABLE--------------------------------------', qc_flags)

# =========================================================
# 5. SET BAD VALUES TO NaN
# =========================================================
r.print('VARS:',VARS)

for var in VARS:
    base.loc[qc_flags[var + '_QC'] != 0, var] = np.nan

r.print('--------------------------------------5. SET BAD VALUES TO NaN--------------------------------------', base)

# =========================================================
# 6. GAP DETECTION (GLOBAL)
# =========================================================
time_diff = base.index.to_series().diff()
expected_freq = pd.Timedelta(FREQ)

gaps = time_diff[time_diff > expected_freq]

gaps_df = pd.DataFrame({
    'gap_start': gaps.index - gaps.values,
    'gap_end': gaps.index,
    'duration_sec': gaps.dt.total_seconds()
})
gaps_df['duration_hours'] = gaps_df['duration_sec'] / 3600
gaps_df['duration_days'] = gaps_df['duration_hours'] / 24

gaps_df['category'] = pd.cut(
    gaps_df['duration_days'],
    bins=[0, 1/24, 12/24, 1, 1e6],
    labels=['minimal', 'small', 'medium', 'large']
)

r.print('--------------------------------------6. GAP DETECTION (GLOBAL)--------------------------------------', gaps_df.head())

# =========================================================
# 7. INTERPOLATE ONLY MINIMAL + SMALL
# =========================================================
def interpolate_only_short_gaps(series, max_steps):
    """
    Interpola només els gaps de NaN amb longitud <= max_steps.
    Els gaps més llargs queden completament com NaN.
    """
    s = series.copy()

    is_nan = s.isna()

    # Identificar blocs consecutius (NaN / no-NaN)
    groups = (is_nan != is_nan.shift()).cumsum()

    for _, idx in is_nan.groupby(groups).groups.items():

        # idx és un DatetimeIndex
        if is_nan.loc[idx].iloc[0]:  # aquest grup és un gap de NaNs
            gap_len = len(idx)
            # r.print(f"Gap detected from {idx[0]} to {idx[-1]} (length: {gap_len})")

            if gap_len <= max_steps:
                s.loc[idx] = s.interpolate(method='time').loc[idx]
            # else: gap llarg → no interpolar res

    return s
input()
interp = base.copy()
interp['interpolated'] = False

MAX_STEPS = 24  # 12h si freq=30min

for var in VARS:
    interp[var] = interpolate_only_short_gaps(
        base[var],
        max_steps=MAX_STEPS
    )

r.print('--------------------------------------7. INTERPOLATE ONLY MINIMAL + SMALL--------------------------------------')

# =========================================================
# 9. MARK INTERPOLATED VALUES
# =========================================================
for var in VARS:
    interp['interpolated'] |= base[var].isna() & interp[var].notna()

print("Interpolated points:", interp['interpolated'].sum())
print('--------------------------------------9. MARK INTERPOLATED VALUES--------------------------------------')

# =========================================================
# 10. PLOT SUBPLOTS
# =========================================================
n_vars = len(VARS)
fig, axes = plt.subplots(n_vars, 1, figsize=(15, 4*n_vars), sharex=True)
gap_colors = {'medium': 'orange', 'large': 'red'}

for i, var in enumerate(VARS):
    ax = axes[i]
    ax.plot(base.index, base[var], color='tab:blue', lw=1, label=f'Original {var}')
    ax.scatter(interp.index[interp['interpolated']], interp.loc[interp['interpolated'], var],
               color='red', s=10, label='Interpolated (minimal + small)')
    for _, row in gaps_df.iterrows():
        if row['category'] in gap_colors:
            ax.axvspan(row['gap_start'], row['gap_end'], color=gap_colors[row['category']], alpha=0.25)
    ax.set_ylabel(var)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=9)

axes[-1].set_xlabel("Time")
plt.suptitle("OBSEA CTD: QC + Interpolated minimal+small gaps - Time Interpolation Method", fontsize=16)
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()

# =========================================================
# 11. FINAL DATASET
# =========================================================
final_df = interp.copy()
for var in VARS:
    final_df[var + '_QC'] = qc_flags[var + '_QC']

final_df['QC_TEMP_INTERP'] = 0
final_df.loc[final_df['interpolated'], 'QC_TEMP_INTERP'] = 1
final_df.loc[final_df['TEMP'].isna(), 'QC_TEMP_INTERP'] = 9

# =========================================================
# 12. SAVE CSV
# =========================================================
output_path = (
    '/home/uripratt/Documents/PhD/OBSEA_data/CTD/'
    'OBSEA_CTD_multivar_QC_minimal_small_interpolated.csv'
)
final_df.to_csv(output_path)
print("✔ CSV guardado en:", output_path)
