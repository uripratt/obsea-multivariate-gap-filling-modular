import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gsw
import sys
from pathlib import Path

# Add project path so we can import STA connector
sys.path.append('/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts')
from cross_calibrate_ctds import fetch_ctd

output_dir = Path("/home/uripratt/.gemini/antigravity/brain/28a2d98a-3117-41eb-b2cb-235e7bd85937")

print("1. Fetching CTD Data for Golden Window (Spring 2012)...")
d16 = fetch_ctd('2012-03-30', '2012-05-04', 'sbe16')
d37 = fetch_ctd('2012-03-30', '2012-05-04', 'sbe37')

# Require overlapping records for correction baseline
sync = d16.join(d37, how='inner', lsuffix='_16', rsuffix='_37').dropna()

print("2. Calculating the bio-fouling drift vector on Conductivity...")
# The residual error between 16 and 37 over time
sync['CNDC_diff'] = sync['CNDC_16'] - sync['CNDC_37']

# Fit linear regression to model the clogging progression
timesteps = np.arange(len(sync))
slope_cndc, intercept_cndc = np.polyfit(timesteps, sync['CNDC_diff'], 1)

print(f"   [+] Detected CNDC Initial Bias: {intercept_cndc:.4f} S/m")
print(f"   [+] Detected CNDC Biological Drift: {slope_cndc:.6f} S/m per timestamp")

print("3. Applying Inverse Biological Correction to SBE16 CNDC...")
# We reverse the drift slope (biofouling loss) AND the static intercept offset 
# to normalize SBE16 perfectly to the stable internal dynamics of SBE37.
sync['CNDC_16_corrected'] = sync['CNDC_16'] - (intercept_cndc + (slope_cndc * timesteps))

print("4. Re-deriving Absolute Salinity (PSAL) via gsw (TEOS-10 Thermodynamic Equations)...")
# gsw expects Conductivity in mS/cm. Our data is in S/m. 1 S/m = 10 mS/cm.
C_mS_cm = sync['CNDC_16_corrected'] * 10.0

# SP_from_C ( Conductivity [mS/cm], Temperature [°C], Pressure [dbar] )
sync['PSAL_16_corrected'] = gsw.SP_from_C(C_mS_cm, sync['TEMP_16'], sync['PRES_16'])

print("5. Generating Visual Proof of Re-Calibration...")
plt.figure(figsize=(12, 6))

plt.plot(sync.index, sync['PSAL_16'], color='red', alpha=0.4, linestyle='--', lw=1, 
         label='SBE16 Raw (Bio-Fouled & Drifting)')
         
plt.plot(sync.index, sync['PSAL_37'], color='darkorange', alpha=0.9, lw=2, 
         label='SBE37 Raw (Stable / Constant but Offset)')
         
plt.plot(sync.index, sync['PSAL_16_corrected'], color='royalblue', alpha=0.8, lw=3, 
         label='SBE16 TEOS-10 Re-derived (Fouling + Bias Corrected)')

plt.title("Oceanographic Salinity Re-Calibration (TEOS-10)\nCorrection of Biological Fouling & Coefficient Drift")
plt.xlabel("Date (Spring 2012)")
plt.ylabel("Practical Salinity (PSU)")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_path = output_dir / "teos_salinity_correction.png"
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Done! Evaluated correction and saved chart to: {out_path.name}")
