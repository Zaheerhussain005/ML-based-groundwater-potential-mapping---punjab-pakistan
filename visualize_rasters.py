"""
Visualize all 9 input feature bands + GW Potential label map.
Produces publication-quality figures: maps, combined map+chart panels,
class distribution charts, correlation chart, feature-class heatmap.

Reads:
  - data/processed/features_normalized.tif   (9-band, float32, NaN nodata)
  - data/processed/all_labels.npy            (37M int8 labels for valid pixels)
  - data/processed/training_samples.csv      (450K rows for per-class stats)

Writes:
  outputs/figures/layers/   - per-band plain PNGs
  outputs/figures/panels/   - per-band map + bar chart combined panels
  outputs/figures/all_features_grid.png
  outputs/figures/gw_potential_label_map.png
  outputs/figures/class_distribution_pub.png
  outputs/figures/feature_class_heatmap.png
  outputs/figures/feature_correlation_pub.png
  outputs/maps/gw_potential_labels.tif       - geo-referenced, opens in QGIS
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from pathlib import Path

RAW_NORM   = Path('data/processed/features_normalized.tif')
LABELS_NPY = Path('data/processed/all_labels.npy')
TRAIN_CSV  = Path('data/processed/training_samples.csv')
FIG_DIR    = Path('outputs/figures')
LAYERS_DIR = FIG_DIR / 'layers'
PANELS_DIR = FIG_DIR / 'panels'
MAPS_DIR   = Path('outputs/maps')
for d in [LAYERS_DIR, PANELS_DIR, MAPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BANDS = [
    ('Elevation', 'terrain',  'Elevation'),
    ('Slope',     'magma',    'Slope'),
    ('Aspect',    'twilight', 'Aspect'),
    ('TWI',       'Blues',    'TWI'),
    ('NDVI',      'RdYlGn',   'NDVI'),
    ('NDWI',      'RdYlBu',   'NDWI'),
    ('Rainfall',  'Blues',    'Rainfall'),
    ('Soil',      'Spectral', 'Soil Texture'),
    ('LST',       'inferno',  'LST'),
]
FEAT_NAMES   = [b[0] for b in BANDS]
CLASS_NAMES  = {0: 'Low', 1: 'Medium', 2: 'High'}
CLASS_COLORS = ['#d73027', '#fee090', '#1a9850']
DOWNSAMPLE   = 5  # 1/5 resolution for display (~2012 x 1376 px)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print('=' * 65)
print('  Raster + Publication Visualization  —  Pothohar Plateau')
print('=' * 65)

# ── Load raster at display resolution ────────────────────────────────────────
print(f'\n[1] Reading {RAW_NORM.name} at 1/{DOWNSAMPLE} res ...')
with rasterio.open(RAW_NORM) as src:
    full_h, full_w = src.height, src.width
    out_h, out_w = full_h // DOWNSAMPLE, full_w // DOWNSAMPLE
    data_ds = src.read(
        out_shape=(src.count, out_h, out_w),
        resampling=Resampling.average,
    )
    meta_full = src.meta.copy()
    crs = src.crs
print(f'    {full_w}x{full_h}  ->  display {out_w}x{out_h}')

# ── Load training samples for stats ──────────────────────────────────────────
print(f'\n[2] Loading {TRAIN_CSV.name} for per-class statistics ...')
df = pd.read_csv(TRAIN_CSV)
df['Class'] = df['GW_Potential'].map(CLASS_NAMES)
means = df.groupby('GW_Potential')[FEAT_NAMES].mean()
corr  = df[FEAT_NAMES + ['GW_Potential']].corr()['GW_Potential'][:-1]
print(f'    {len(df):,} rows loaded.')

# ── Plain per-band PNGs ───────────────────────────────────────────────────────
print('\n[3] Per-band plain PNGs ...')
for i, (name, cmap, label) in enumerate(BANDS):
    arr = data_ds[i]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest')
    ax.set_title(f'{label}  —  Pothohar Plateau (normalized 0-1)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f'{label} (0 = min, 1 = max)', fontsize=9)
    plt.tight_layout()
    out = LAYERS_DIR / f'{i+1:02d}_{name.lower()}.png'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'    {out.name}')

# ── Combined 3x3 features grid ────────────────────────────────────────────────
print('\n[4] Combined 3x3 features grid ...')
fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.suptitle('Input Feature Layers — Pothohar Plateau (normalized 0–1)',
             fontsize=15, fontweight='bold', y=0.998)
for ax, (name, cmap, label), arr in zip(axes.flat, BANDS, data_ds):
    im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest')
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'all_features_grid.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print('    all_features_grid.png')

# ── Panel figures: map + per-class bar chart ──────────────────────────────────
print('\n[5] Map + bar chart panel figures ...')
for i, (name, cmap, label) in enumerate(BANDS):
    arr     = data_ds[i]
    vals    = [means.loc[cls, name] for cls in [0, 1, 2]]
    c_names = [CLASS_NAMES[c] for c in [0, 1, 2]]

    fig = plt.figure(figsize=(16, 6.5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.8, 1], wspace=0.07)

    # Left: raster map
    ax_map = fig.add_subplot(gs[0])
    im = ax_map.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest')
    ax_map.set_title(f'{label} — Pothohar Plateau (normalized 0–1)',
                     fontsize=13, fontweight='bold')
    ax_map.axis('off')
    cbar = plt.colorbar(im, ax=ax_map, fraction=0.025, pad=0.01, shrink=0.92)
    cbar.set_label(f'{label} value', fontsize=9)

    # Right: per-class mean bar chart
    ax_bar = fig.add_subplot(gs[1])
    bars = ax_bar.bar(c_names, vals, color=CLASS_COLORS, edgecolor='black',
                      linewidth=0.7, width=0.5)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel('Mean normalized value', fontsize=10)
    ax_bar.set_title(f'Class mean — {label}', fontsize=11, fontweight='bold')
    ax_bar.set_xlabel('GW Potential class', fontsize=10)
    ax_bar.grid(True, axis='y', alpha=0.35)

    # Pearson r annotation
    r_val = corr[name]
    ax_bar.text(0.97, 0.96, f'Pearson r = {r_val:+.3f}', transform=ax_bar.transAxes,
                ha='right', va='top', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    out = PANELS_DIR / f'{i+1:02d}_{name.lower()}_panel.png'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'    {out.name}')

# ── Reconstruct full-resolution label map ─────────────────────────────────────
print('\n[6] Reconstructing label map ...')
with rasterio.open(RAW_NORM) as src:
    band1_full = src.read(1)
valid_mask_2d = ~np.isnan(band1_full)
labels = np.load(LABELS_NPY)
assert labels.size == valid_mask_2d.sum(), 'Label count mismatch with valid mask'

label_grid = np.full(band1_full.shape, -1, dtype=np.int8)
label_grid[valid_mask_2d] = labels
del band1_full

counts = np.bincount(labels.astype(np.int64), minlength=3)
total  = counts.sum()
print(f'    Low: {counts[0]:,} | Medium: {counts[1]:,} | High: {counts[2]:,}')

# ── Save geo-referenced label GeoTIFF ─────────────────────────────────────────
print('\n[7] Saving label GeoTIFF ...')
meta_lbl = meta_full.copy()
meta_lbl.update(count=1, dtype='int8', nodata=-1, compress='lzw')
tif_out = MAPS_DIR / 'gw_potential_labels.tif'
with rasterio.open(tif_out, 'w', **meta_lbl) as dst:
    dst.write(label_grid[np.newaxis, :, :])
print(f'    {tif_out.name}  ({tif_out.stat().st_size / 1e6:.1f} MB)')

# ── Label map PNG ──────────────────────────────────────────────────────────────
print('\n[8] Rendering label map PNG ...')
ds_grid = label_grid[::DOWNSAMPLE, ::DOWNSAMPLE]
masked  = np.ma.masked_where(ds_grid == -1, ds_grid)
class_cmap = ListedColormap(CLASS_COLORS)
class_cmap.set_bad(color='white', alpha=0.0)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], class_cmap.N)

fig, ax = plt.subplots(figsize=(13, 8))
ax.imshow(masked, cmap=class_cmap, norm=norm, interpolation='nearest')
ax.set_title('Groundwater Potential — Pothohar Plateau\n(knowledge-based weighted-overlay labels)',
             fontsize=14, fontweight='bold', pad=12)
ax.axis('off')
legend_elems = [
    Patch(facecolor=CLASS_COLORS[0], edgecolor='black', label=f'Low  ({counts[0]/total*100:.1f}%)'),
    Patch(facecolor=CLASS_COLORS[1], edgecolor='black', label=f'Medium  ({counts[1]/total*100:.1f}%)'),
    Patch(facecolor=CLASS_COLORS[2], edgecolor='black', label=f'High  ({counts[2]/total*100:.1f}%)'),
]
ax.legend(handles=legend_elems, loc='lower right', fontsize=11,
          title='GW Potential', title_fontsize=11, framealpha=0.9)
sub = (f'Low: {counts[0]:,}  |  Medium: {counts[1]:,}  |  High: {counts[2]:,}'
       f'  |  Total valid: {total:,} pixels  |  Nodata masked white')
fig.text(0.5, 0.02, sub, ha='center', fontsize=9, style='italic')
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(FIG_DIR / 'gw_potential_label_map.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print('    gw_potential_label_map.png')

# ── Publication: Class distribution bar + pie ─────────────────────────────────
print('\n[9] Class distribution publication figure ...')
fig = plt.figure(figsize=(15, 6))
gs  = gridspec.GridSpec(1, 3, wspace=0.35)

# Left: bar chart with pixel counts
ax1 = fig.add_subplot(gs[0])
bars = ax1.bar(list(CLASS_NAMES.values()), counts, color=CLASS_COLORS,
               edgecolor='black', linewidth=0.8, width=0.5)
ax1.set_title('Pixel Count per Class', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Pixels')
ax1.set_ylim(0, max(counts) * 1.18)
for bar, v in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + total * 0.003,
             f'{v:,}', ha='center', fontsize=10, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)

# Middle: pie chart
ax2 = fig.add_subplot(gs[1])
wedge_labels = [f'{CLASS_NAMES[i]}\n{counts[i]/total*100:.1f}%' for i in range(3)]
ax2.pie(counts, labels=wedge_labels, colors=CLASS_COLORS,
        startangle=90, textprops={'fontsize': 11},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8},
        explode=[0.03, 0.03, 0.05])
ax2.set_title('Class Proportion', fontsize=12, fontweight='bold')

# Right: horizontal proportional bar (stacked)
ax3 = fig.add_subplot(gs[2])
pcts = counts / total * 100
left = 0
for i, (name, pct) in enumerate(zip(CLASS_NAMES.values(), pcts)):
    ax3.barh(0, pct, left=left, color=CLASS_COLORS[i], edgecolor='black',
             linewidth=0.7, height=0.45, label=f'{name}: {pct:.1f}%')
    if pct > 5:
        ax3.text(left + pct / 2, 0, f'{pct:.1f}%', ha='center', va='center',
                 fontsize=11, fontweight='bold')
    left += pct
ax3.set_xlim(0, 100)
ax3.set_yticks([])
ax3.set_xlabel('% of valid pixels')
ax3.set_title('Class Split (% of 37M pixels)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)

fig.suptitle('Groundwater Potential Class Distribution — Pothohar Plateau (5 districts)',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(FIG_DIR / 'class_distribution_pub.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print('    class_distribution_pub.png')

# ── Publication: Feature–class heatmap ───────────────────────────────────────
print('\n[10] Feature-class mean heatmap ...')
heat_data = means.rename(index=CLASS_NAMES)[FEAT_NAMES]
fig, ax = plt.subplots(figsize=(13, 4))
im = ax.imshow(heat_data.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(FEAT_NAMES)))
ax.set_xticklabels(FEAT_NAMES, fontsize=10, rotation=15, ha='right')
ax.set_yticks(range(3))
ax.set_yticklabels(['Low', 'Medium', 'High'], fontsize=11, fontweight='bold')
for r in range(3):
    for c in range(len(FEAT_NAMES)):
        v = heat_data.values[r, c]
        ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=9,
                color='black' if 0.3 < v < 0.7 else 'white')
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label='Mean normalized value (0–1)')
ax.set_title('Mean Feature Value per GW Potential Class — Pothohar Plateau',
             fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(FIG_DIR / 'feature_class_heatmap.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print('    feature_class_heatmap.png')

# ── Publication: Feature–target correlation bar chart ────────────────────────
print('\n[11] Feature–target correlation publication chart ...')
corr_sorted = corr.sort_values()
bar_colors  = ['#1a9850' if v > 0 else '#d73027' for v in corr_sorted]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(corr_sorted.index, corr_sorted.values,
               color=bar_colors, edgecolor='black', linewidth=0.7)
ax.axvline(0, color='black', linewidth=0.9)
for bar, val in zip(bars, corr_sorted.values):
    offset = 0.015 if val >= 0 else -0.015
    ha     = 'left' if val >= 0 else 'right'
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center', ha=ha, fontsize=9.5, fontweight='bold')
ax.set_xlabel('Pearson Correlation with GW Potential', fontsize=11)
ax.set_title('Feature Correlation with Groundwater Potential\n'
             '(green = positive predictor, red = negative predictor)',
             fontsize=13, fontweight='bold')
ax.set_xlim(-1.05, 1.05)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
# Legend patches
from matplotlib.patches import Patch as _Patch
ax.legend(handles=[
    _Patch(facecolor='#1a9850', edgecolor='black', label='Positive correlation (higher → better GW)'),
    _Patch(facecolor='#d73027', edgecolor='black', label='Negative correlation (lower → better GW)'),
], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(FIG_DIR / 'feature_correlation_pub.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print('    feature_correlation_pub.png')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('  All outputs saved.')
print('=' * 65)
print()
print('  outputs/figures/layers/        9 per-band plain PNGs')
print('  outputs/figures/panels/        9 map + bar chart panels')
print('  outputs/figures/')
print('    all_features_grid.png        3x3 thesis grid')
print('    gw_potential_label_map.png   Low / Medium / High colored map')
print('    class_distribution_pub.png   bar + pie + stacked bar')
print('    feature_class_heatmap.png    9x3 cell heatmap')
print('    feature_correlation_pub.png  horizontal bar chart')
print()
print('  outputs/maps/')
print('    gw_potential_labels.tif      geo-referenced, open in QGIS')
