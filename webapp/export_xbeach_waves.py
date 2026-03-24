#!/usr/bin/env python3
"""
Export XBeach simulation results (zs, H) to a binary file optimized for
the Digital Twin web viewer.

Binary format:
  - Header: JSON (null-terminated, padded to 4096 bytes)
  - Data block 1: zs  Float32[frames × ny × nx]  (water surface elevation)
  - Data block 2: H   Float32[frames × ny × nx]  (significant wave height)

Usage:
  python export_xbeach_waves.py
"""

import json
import struct
import numpy as np
import sys

# ── Configuration ──────────────────────────────────────────────────────
NC_PATH = '/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts/xbeach_data_results/xboutput.nc'
OUT_PATH = '/home/uripratt/Documents/PhD/OBSEA_data/CTD/scripts/webapp/assets/data/xbeach_waves.bin'
SUBSAMPLE = 4          # Take every Nth grid point
HEADER_SIZE = 4096     # Fixed header block size in bytes
SKIP_INITIAL = 20      # Skip first N frames (spin-up, all zeros)
# ──────────────────────────────────────────────────────────────────────

def main():
    try:
        import netCDF4 as nc
    except ImportError:
        print("ERROR: netCDF4 not installed. Run: pip install netCDF4")
        sys.exit(1)

    print(f"Opening {NC_PATH} ...")
    ds = nc.Dataset(NC_PATH)

    # ── Read global coordinates ─────────────────────────────────────
    gx_full = ds.variables['globalx'][:]  # (ny, nx)
    gy_full = ds.variables['globaly'][:]
    t_full  = ds.variables['globaltime'][:]  # (ntime,)

    # ── Subsample spatial grid ──────────────────────────────────────
    gx = gx_full[::SUBSAMPLE, ::SUBSAMPLE]
    gy = gy_full[::SUBSAMPLE, ::SUBSAMPLE]
    ny, nx = gx.shape

    # ── Skip spin-up frames ─────────────────────────────────────────
    t = t_full[SKIP_INITIAL:]
    nframes = len(t)

    print(f"Subsampled grid: {nx} × {ny}  ({nx*ny:,} vertices)")
    print(f"Frames: {nframes}  (skipped first {SKIP_INITIAL})")
    print(f"Domain: {gx.min():.1f}–{gx.max():.1f} E,  {gy.min():.1f}–{gy.max():.1f} N")
    print(f"Time: {t[0]:.0f}s – {t[-1]:.0f}s  (dt={t[1]-t[0]:.1f}s)")

    # ── Read & subsample zs and H ───────────────────────────────────
    print("Reading zs ...")
    zs_full = ds.variables['zs'][SKIP_INITIAL:, ::SUBSAMPLE, ::SUBSAMPLE]
    zs = np.asarray(zs_full, dtype=np.float32)
    np.nan_to_num(zs, copy=False, nan=0.0)

    print("Reading H ...")
    H_full = ds.variables['H'][SKIP_INITIAL:, ::SUBSAMPLE, ::SUBSAMPLE]
    H = np.asarray(H_full, dtype=np.float32)
    np.nan_to_num(H, copy=False, nan=0.0)

    ds.close()

    print(f"zs shape: {zs.shape}, range [{zs.min():.4f}, {zs.max():.4f}]")
    print(f"H  shape: {H.shape},  range [{H.min():.4f}, {H.max():.4f}]")

    # ── Build header ─────────────────────────────────────────────────
    # Convert coordinate arrays to lists for x/y edges (first row / first col)
    header = {
        "version": 2,
        "nx": int(nx),
        "ny": int(ny),
        "frames": int(nframes),
        "dt": float(t[1] - t[0]),
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "domain_x_min": float(gx.min()),
        "domain_x_max": float(gx.max()),
        "domain_y_min": float(gy.min()),
        "domain_y_max": float(gy.max()),
        "domain_width_m": float(gx.max() - gx.min()),
        "domain_height_m": float(gy.max() - gy.min()),
        "zs_min": float(zs.min()),
        "zs_max": float(zs.max()),
        "H_min": float(H.min()),
        "H_max": float(H.max()),
        "subsample": SUBSAMPLE,
        "data_order": "zs[frames,ny,nx] then H[frames,ny,nx]",
        "dtype": "float32"
    }

    header_json = json.dumps(header, indent=2)
    header_bytes = header_json.encode('utf-8') + b'\x00'  # null-terminate

    if len(header_bytes) > HEADER_SIZE:
        print(f"ERROR: Header is {len(header_bytes)} bytes, exceeds {HEADER_SIZE}")
        sys.exit(1)

    # Pad to HEADER_SIZE
    header_bytes = header_bytes.ljust(HEADER_SIZE, b'\x00')

    # ── Write binary ─────────────────────────────────────────────────
    import os
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    total_floats = nframes * ny * nx * 2  # zs + H
    expected_size_mb = (HEADER_SIZE + total_floats * 4) / (1024 * 1024)
    print(f"Writing {OUT_PATH}  (expected ~{expected_size_mb:.1f} MB) ...")

    with open(OUT_PATH, 'wb') as f:
        f.write(header_bytes)
        f.write(zs.tobytes())   # C-order: [frames, ny, nx]
        f.write(H.tobytes())

    actual_size = os.path.getsize(OUT_PATH)
    print(f"✅ Done! File size: {actual_size / (1024*1024):.1f} MB")
    print(f"\nHeader summary:")
    for k, v in header.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
