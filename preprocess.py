"""
Preprocess the raw GeoTIFF:
  1. Read all 9 bands
  2. Flatten to pixel rows, drop nodata/NaN
  3. Normalize each band to [0, 1]
  4. Save normalized raster + CSV for ML
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

RAW     = Path('data/raw/Pothohar_Groundwater_Features.tif')
PROC    = Path('data/processed')
PROC.mkdir(parents=True, exist_ok=True)

BANDS = ['Elevation','Slope','Aspect','TWI','NDVI','NDWI','Rainfall','Soil','LST']

print('Reading raster...')
with rasterio.open(RAW) as src:
    data   = src.read()          # shape: (9, rows, cols)
    meta   = src.meta.copy()
    nodata = src.nodata
    transform = src.transform
    n_bands, n_rows, n_cols = data.shape

print(f'Raster shape: {n_bands} bands x {n_rows} rows x {n_cols} cols')

# Flatten to 2D: (pixels, bands)
flat = data.reshape(n_bands, -1).T    # (N_pixels, 9)

# Build valid pixel mask (no NaN, no nodata)
nan_mask  = np.isfinite(flat).all(axis=1)
if nodata is not None:
    nd_mask = ~(flat == nodata).any(axis=1)
    valid   = nan_mask & nd_mask
else:
    valid = nan_mask

print(f'Total pixels : {len(flat):,}')
print(f'Valid pixels : {valid.sum():,}  ({valid.mean()*100:.1f}%)')

valid_data = flat[valid]

# Normalize to [0, 1]
print('Normalizing...')
scaler = MinMaxScaler()
normalized = scaler.fit_transform(valid_data)

# Save as CSV (for ML training)
df = pd.DataFrame(normalized, columns=BANDS)
csv_path = PROC / 'features.csv'
df.to_csv(csv_path, index=False)
print(f'Saved CSV  : {csv_path}  ({len(df):,} rows x {len(BANDS)} cols)')

# Save normalized raster (reconstruct full grid)
norm_full = np.full((n_bands, n_rows * n_cols), np.nan, dtype=np.float32)
norm_full[:, valid] = normalized.T
norm_full = norm_full.reshape(n_bands, n_rows, n_cols)

meta.update(dtype='float32', nodata=np.nan)
tif_path = PROC / 'features_normalized.tif'
with rasterio.open(tif_path, 'w', **meta) as dst:
    dst.write(norm_full)
print(f'Saved TIF  : {tif_path}')

# Save pixel coordinates for spatial reconstruction later
rows_idx, cols_idx = np.where(valid.reshape(n_rows, n_cols))
coords = pd.DataFrame({'row': rows_idx, 'col': cols_idx})
coords.to_csv(PROC / 'pixel_coords.csv', index=False)
print(f'Saved coords: {PROC}/pixel_coords.csv')

print()
print('=== Band Statistics (normalized) ===')
print(df.describe().round(3).to_string())
print()
print('Preprocessing complete. Ready for ML training.')
