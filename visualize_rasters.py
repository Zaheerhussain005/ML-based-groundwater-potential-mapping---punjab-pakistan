"""
Visualize all 9 input feature bands + reconstructed GW Potential label map.
Outputs PNGs for thesis figures and a geo-referenced label GeoTIFF for QGIS.

Reads:
  - data/processed/features_normalized.tif   (9-band, 10062 x 6884, float32, NaN nodata)
  - data/processed/all_labels.npy            (37M int8 labels for valid pixels)

Writes:
  - outputs/figures/layers/01_elevation.png ... 09_lst.png
  - outputs/figures/all_features_grid.png
  - outputs/figures/gw_potential_label_map.png
  - outputs/maps/gw_potential_labels.tif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import rasterio
from rasterio.enums import Resampling
from pathlib import Path

RAW_NORM   = Path('data/processed/features_normalized.tif')
LABELS_NPY = Path('data/processed/all_labels.npy')
FIG_DIR    = Path('outputs/figures')
LAYERS_DIR = FIG_DIR / 'layers'
MAPS_DIR   = Path('outputs/maps')
LAYERS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

# Per-band rendering config: (display_name, colormap, units_hint)
BANDS = [
    ('Elevation', 'terrain',  'normalized'),
    ('Slope',     'magma',    'normalized'),
    ('Aspect',    'twilight', 'normalized'),  # cyclic
    ('TWI',       'Blues',    'normalized'),
    ('NDVI',      'RdYlGn',   'normalized'),
    ('NDWI',      'RdYlBu',   'normalized'),
    ('Rainfall',  'Blues',    'normalized'),
    ('Soil',      'Spectral', 'normalized'),
    ('LST',       'inferno',  'normalized'),
]

DOWNSAMPLE = 5   # factor 5 -> ~2012 x 1376 px PNG (manageable file size)

print('=' * 60)
print('  Raster Visualization')
print('=' * 60)

# ── 1. Read all 9 bands at downsampled resolution ────────────────────────────
print(f'\n[1] Reading {RAW_NORM.name} at 1/{DOWNSAMPLE} resolution ...')
with rasterio.open(RAW_NORM) as src:
    full_h, full_w = src.height, src.width
    out_h = full_h // DOWNSAMPLE
    out_w = full_w // DOWNSAMPLE
    print(f'    Full size: {full_w} x {full_h}  ->  display {out_w} x {out_h}')
    data = src.read(
        out_shape=(src.count, out_h, out_w),
        resampling=Resampling.average,
    )
    transform_full = src.transform
    crs = src.crs
    meta_full = src.meta.copy()
print(f'    Loaded array: {data.shape} dtype={data.dtype}')

# ── 2. Per-band PNGs ─────────────────────────────────────────────────────────
print('\n[2] Rendering per-band PNGs ...')
for i, (name, cmap, units) in enumerate(BANDS):
    arr = data[i]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest')
    ax.set_title(f'{name}  ({units})  -  Pothohar Plateau, 5 districts',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f'{name} (0-1)', fontsize=10)
    plt.tight_layout()
    out = LAYERS_DIR / f'{i+1:02d}_{name.lower()}.png'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'    Saved: {out.relative_to(Path.cwd()) if out.is_absolute() else out}')

# ── 3. Combined 3x3 grid figure ──────────────────────────────────────────────
print('\n[3] Rendering combined 3x3 grid ...')
fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.suptitle('Input Feature Layers - Pothohar Plateau (normalized 0-1)',
             fontsize=15, fontweight='bold', y=0.995)
for ax, (name, cmap, _), arr in zip(axes.flat, BANDS, data):
    im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'all_features_grid.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f'    Saved: outputs/figures/all_features_grid.png')

# ── 4. Reconstruct full-resolution label map ─────────────────────────────────
print('\n[4] Reconstructing full-resolution label map ...')
print('    Reading band 1 at full resolution (for valid mask) ...')
with rasterio.open(RAW_NORM) as src:
    band1_full = src.read(1)  # ~277 MB float32

valid_mask_2d = ~np.isnan(band1_full)
n_valid = int(valid_mask_2d.sum())
print(f'    Valid pixels: {n_valid:,}')

print(f'    Loading {LABELS_NPY.name} ...')
labels = np.load(LABELS_NPY)
print(f'    Labels: {labels.shape} dtype={labels.dtype}')
assert labels.size == n_valid, (
    f'Label count {labels.size:,} does not match valid mask {n_valid:,}'
)

label_grid = np.full(band1_full.shape, -1, dtype=np.int8)
label_grid[valid_mask_2d] = labels
del band1_full   # free memory

# ── 5. Save as geo-referenced GeoTIFF (for QGIS, thesis maps) ────────────────
print('\n[5] Saving label GeoTIFF ...')
meta_lbl = meta_full.copy()
meta_lbl.update(count=1, dtype='int8', nodata=-1, compress='lzw')
tif_out = MAPS_DIR / 'gw_potential_labels.tif'
with rasterio.open(tif_out, 'w', **meta_lbl) as dst:
    dst.write(label_grid[np.newaxis, :, :])
print(f'    Saved: {tif_out}  ({tif_out.stat().st_size / 1e6:.1f} MB)')

# ── 6. Render label map PNG (downsampled with nearest neighbor) ──────────────
print('\n[6] Rendering label map PNG ...')
# Nearest-neighbor decimation - never average class labels.
ds_grid = label_grid[::DOWNSAMPLE, ::DOWNSAMPLE]
print(f'    Display grid: {ds_grid.shape}')

# Mask nodata and render as masked array so it appears white/transparent.
masked = np.ma.masked_where(ds_grid == -1, ds_grid)
class_cmap = ListedColormap(['#d73027', '#fee090', '#1a9850'])  # Low, Med, High
class_cmap.set_bad(color='white', alpha=0.0)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], class_cmap.N)

fig, ax = plt.subplots(figsize=(13, 8))
im = ax.imshow(masked, cmap=class_cmap, norm=norm, interpolation='nearest')
ax.set_title('Groundwater Potential - Pothohar Plateau (weighted-overlay labels)',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xticks([]); ax.set_yticks([])

legend_elems = [
    Patch(facecolor='#d73027', edgecolor='black', label='Low (0)'),
    Patch(facecolor='#fee090', edgecolor='black', label='Medium (1)'),
    Patch(facecolor='#1a9850', edgecolor='black', label='High (2)'),
]
ax.legend(handles=legend_elems, loc='lower right', fontsize=11,
          title='GW Potential', title_fontsize=11, framealpha=0.9)

# Class proportions sub-text
counts = np.bincount(labels.astype(np.int64), minlength=3)
total = counts.sum()
sub = (f'Low: {counts[0]:,} ({counts[0]/total*100:.1f}%)   '
       f'Medium: {counts[1]:,} ({counts[1]/total*100:.1f}%)   '
       f'High: {counts[2]:,} ({counts[2]/total*100:.1f}%)')
fig.text(0.5, 0.02, sub, ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 1])
out_png = FIG_DIR / 'gw_potential_label_map.png'
plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'    Saved: {out_png}')

print('\n' + '=' * 60)
print('  Visualization complete.')
print('=' * 60)
print(f'  Per-band PNGs : outputs/figures/layers/')
print(f'  Combined grid : outputs/figures/all_features_grid.png')
print(f'  Label map PNG : outputs/figures/gw_potential_label_map.png')
print(f'  Label GeoTIFF : outputs/maps/gw_potential_labels.tif  (open in QGIS)')
