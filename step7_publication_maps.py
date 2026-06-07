"""
Publication-quality figures matching Das 2018 (Applied Water Science) style.
Generates 5 thesis-ready maps at 300 DPI.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import rasterio
from rasterio.enums import Resampling
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PRED_TIF  = Path('outputs/maps/gw_prediction_xgboost.tif')
PROB_TIF  = Path('outputs/maps/gw_probability_high.tif')
LABEL_TIF = Path('outputs/maps/gw_potential_labels.tif')
FEAT_TIF  = Path('data/processed/features_normalized.tif')
WELLS_CSV = Path('data/validation/wells_nn_recovered.csv')
OUT       = Path('outputs/figures/publication')
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = ['Elevation','Slope','Aspect','TWI',
                 'NDVI','NDWI','Rainfall','Soil Texture','LST']

CLASS_COLORS  = ['#d7191c', '#fee090', '#1a9850']
CLASS_LABELS  = ['Low Potential', 'Medium Potential', 'High Potential']

CMAP3   = mcolors.ListedColormap(CLASS_COLORS)
BNDS3   = [-0.5, 0.5, 1.5, 2.5]
NORM3   = mcolors.BoundaryNorm(BNDS3, CMAP3.N)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ds(path, factor=4):
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(max(1, src.height // factor),
                       max(1, src.width  // factor)),
            resampling=Resampling.nearest
        ).astype(float)
        nd = src.nodata
        if nd is not None:
            data[data == nd] = np.nan
        return data, src.bounds


def make_hillshade(arr, azimuth=315, altitude=45):
    valid = np.where(np.isnan(arr), 0, arr)
    az  = np.radians(360 - azimuth + 90)
    alt = np.radians(altitude)
    dy, dx = np.gradient(valid)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hs = (np.sin(alt)*np.cos(slope) +
          np.cos(alt)*np.sin(slope)*np.cos(az - aspect))
    return np.clip(hs, 0, 1)


def extent_of(bounds):
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def add_scalebar(ax, bounds, km=50):
    deg = km / 111.0
    x0  = bounds.left   + (bounds.right  - bounds.left)  * 0.06
    y0  = bounds.bottom + (bounds.top    - bounds.bottom) * 0.05
    ax.plot([x0, x0+deg], [y0, y0], 'k-', lw=3)
    ax.plot([x0]*2,       [y0-0.005, y0+0.005], 'k-', lw=2)
    ax.plot([x0+deg]*2,   [y0-0.005, y0+0.005], 'k-', lw=2)
    ax.text(x0+deg/2, y0-0.018, f'{km} km',
            ha='center', va='top', fontsize=8, fontweight='bold')


def add_north(ax, bounds):
    x = bounds.right  - (bounds.right  - bounds.left)  * 0.06
    y = bounds.top    - (bounds.top    - bounds.bottom) * 0.10
    ax.annotate('',
                xy=(x, y+0.05), xytext=(x, y),
                xycoords='data', textcoords='data',
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    ax.text(x, y+0.07, 'N',
            ha='center', va='bottom', fontsize=12, fontweight='bold')


def add_grid(ax, bounds, n=4):
    lons = np.linspace(bounds.left, bounds.right, n+1)
    lats = np.linspace(bounds.bottom, bounds.top, n+1)
    for lon in lons:
        ax.axvline(lon, color='grey', lw=0.4, ls='--', alpha=0.6)
        ax.text(lon, bounds.bottom - (bounds.top-bounds.bottom)*0.015,
                f'{lon:.2f}E', ha='center', va='top', fontsize=7, color='#444')
    for lat in lats:
        ax.axhline(lat, color='grey', lw=0.4, ls='--', alpha=0.6)
        ax.text(bounds.left - (bounds.right-bounds.left)*0.01, lat,
                f'{lat:.2f}N', ha='right', va='center', fontsize=7, color='#444')


def map_ax(ax, data, bounds, title, alpha=0.82, hillshade=None):
    ext = extent_of(bounds)
    if hillshade is not None:
        ax.imshow(hillshade, extent=ext, cmap='gray',
                  vmin=0.3, vmax=1.0, aspect='auto', alpha=0.35, zorder=1)
    m = np.ma.masked_invalid(data)
    ax.imshow(m, extent=ext, cmap=CMAP3, norm=NORM3,
              aspect='auto', alpha=alpha, zorder=2)
    add_grid(ax, bounds, n=4)
    mx, my = (bounds.right-bounds.left)*0.02, (bounds.top-bounds.bottom)*0.02
    ax.set_xlim(bounds.left-mx, bounds.right+mx)
    ax.set_ylim(bounds.bottom-my, bounds.top+my)
    ax.set_facecolor('#b8d4e8')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('Longitude (E)', fontsize=9)
    ax.set_ylabel('Latitude (N)', fontsize=9)
    add_scalebar(ax, bounds)
    add_north(ax, bounds)


# ════════════════════════════════════════════════════════════════════════════════
# FIG 1 — Main GW Prediction Map
# ════════════════════════════════════════════════════════════════════════════════
print('[1] Main prediction map ...')

pred, bounds = load_ds(PRED_TIF, factor=4)
prob, _      = load_ds(PROB_TIF,  factor=4)
hs = make_hillshade(uniform_filter(np.nan_to_num(prob, nan=0)*100, size=8))

fig = plt.figure(figsize=(14, 10), facecolor='white')
gs  = GridSpec(3, 4, figure=fig,
               left=0.07, right=0.85, top=0.92, bottom=0.08,
               hspace=0.05, wspace=0.05)

ax_main = fig.add_subplot(gs[:, :3])
map_ax(ax_main, pred, bounds,
       'Groundwater Potential Zones — Pothohar Plateau\n'
       'XGBoost ML | 37 M pixels | 30 m resolution',
       hillshade=hs)

patches = [mpatches.Patch(color=c, label=l) for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
ax_main.legend(handles=patches, loc='lower left', fontsize=9,
               title='GW Potential Zone', title_fontsize=10,
               framealpha=0.92, edgecolor='black')

# Right panels
ax_r1 = fig.add_subplot(gs[0, 3])
ax_r1.axis('off')
info  = ('Study Area\n'
         'Pothohar Plateau\n'
         'Punjab, Pakistan\n'
         '~25,000 km2\n\n'
         'Districts:\n'
         'Chakwal\n'
         'Rawalpindi\n'
         'Attock\n'
         'Jhelum\n'
         'Mianwali\n\n'
         'Resolution: 30 m\n'
         'Pixels: 37.0 M\n'
         'CRS: EPSG:4326')
ax_r1.text(0.05, 0.97, info, transform=ax_r1.transAxes,
           va='top', ha='left', fontsize=8, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9,
                     edgecolor='grey'))

ax_r2 = fig.add_subplot(gs[1, 3])
valid = pred[~np.isnan(pred)]
pcts  = [(valid == c).sum() / len(valid) * 100 for c in [0, 1, 2]]
ax_r2.pie(pcts, colors=CLASS_COLORS, autopct='%1.1f%%',
          pctdistance=0.72, textprops={'fontsize': 8}, startangle=90,
          wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1))
ax_r2.set_title('Area Distribution', fontsize=9, fontweight='bold', pad=4)

ax_r3 = fig.add_subplot(gs[2, 3])
models    = ['RF', 'XGB*', 'LGB', 'Ens.']
accs      = [98.32, 99.12, 99.11, 99.09]
bar_clrs  = ['#74add1','#f46d43','#74add1','#74add1']
bars = ax_r3.bar(models, accs, color=bar_clrs, edgecolor='black', lw=0.8)
ax_r3.set_ylim(97.5, 99.5)
ax_r3.set_ylabel('Accuracy (%)', fontsize=8)
ax_r3.set_title('Model Comparison', fontsize=9, fontweight='bold')
ax_r3.tick_params(labelsize=8)
ax_r3.yaxis.grid(True, alpha=0.4)
for bar, v in zip(bars, accs):
    ax_r3.text(bar.get_x()+bar.get_width()/2, v+0.02,
               f'{v:.2f}', ha='center', va='bottom', fontsize=7)

out1 = OUT / 'Fig1_GW_Prediction_Map.png'
fig.savefig(out1, dpi=300, facecolor='white')
plt.close()
print(f'   Saved -> {out1}')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 2 — Probability Heatmap
# ════════════════════════════════════════════════════════════════════════════════
print('[2] Probability heatmap ...')

fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')
ext = extent_of(bounds)
mp  = np.ma.masked_invalid(prob)
im2 = ax.imshow(mp, extent=ext, cmap='RdYlGn',
                vmin=0, vmax=1, aspect='auto', zorder=2)
add_grid(ax, bounds, n=4)
mx, my = (bounds.right-bounds.left)*0.02, (bounds.top-bounds.bottom)*0.02
ax.set_xlim(bounds.left-mx, bounds.right+mx)
ax.set_ylim(bounds.bottom-my, bounds.top+my)
ax.set_facecolor('#b8d4e8')
cbar = plt.colorbar(im2, ax=ax, fraction=0.025, pad=0.03, shrink=0.8)
cbar.set_label('P(High GW Potential)', fontsize=10)
cbar.ax.tick_params(labelsize=9)
ax.set_title('Groundwater Potential Probability — Pothohar Plateau\n'
             'XGBoost Continuous Output: P(Class=High)',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Longitude (E)', fontsize=10)
ax.set_ylabel('Latitude (N)', fontsize=10)
add_scalebar(ax, bounds)
add_north(ax, bounds)

out2 = OUT / 'Fig2_GW_Probability_Map.png'
fig.savefig(out2, dpi=300, facecolor='white')
plt.close()
print(f'   Saved -> {out2}')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 3 — 9-Feature Panel
# ════════════════════════════════════════════════════════════════════════════════
print('[3] Feature panel (9 layers) ...')

if FEAT_TIF.exists():
    cmaps = ['terrain','YlOrRd','hsv','Blues',
             'YlGn','RdYlGn_r','YlGnBu','BrBG','RdBu_r']

    fig, axes = plt.subplots(3, 3, figsize=(16, 13), facecolor='white')
    fig.suptitle('Input Feature Layers — Pothohar Plateau, Punjab, Pakistan\n'
                 '9 Remote-Sensing Thematic Maps | Normalized [0, 1] | 30 m',
                 fontsize=13, fontweight='bold', y=0.995)

    with rasterio.open(FEAT_TIF) as src:
        fb = src.bounds
        fe = [fb.left, fb.right, fb.bottom, fb.top]
        for i, ax in enumerate(axes.flat):
            band = src.read(
                i+1,
                out_shape=(src.height // 5, src.width // 5),
                resampling=Resampling.bilinear
            ).astype(float)
            nd = src.nodata
            if nd is not None:
                band[band == nd] = np.nan
            mb = np.ma.masked_invalid(band)
            im_f = ax.imshow(mb, extent=fe, cmap=cmaps[i],
                             aspect='auto', interpolation='bilinear')
            ax.set_title(FEATURE_NAMES[i], fontsize=10, fontweight='bold', pad=4)
            ax.set_facecolor('#cccccc')
            ax.tick_params(labelsize=7)
            if i % 3 == 0:
                ax.set_ylabel('Lat (N)', fontsize=8)
            if i >= 6:
                ax.set_xlabel('Lon (E)', fontsize=8)
            cb = plt.colorbar(im_f, ax=ax, fraction=0.04, pad=0.02, shrink=0.8)
            cb.ax.tick_params(labelsize=7)
            cb.set_label('Norm.', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out3 = OUT / 'Fig3_Feature_Maps_Panel.png'
    fig.savefig(out3, dpi=300, facecolor='white')
    plt.close()
    print(f'   Saved -> {out3}')
else:
    print('   SKIPPED -- features_normalized.tif not found')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 4 — Weighted Overlay vs ML side-by-side
# ════════════════════════════════════════════════════════════════════════════════
print('[4] Overlay vs ML comparison ...')

label_ds, _  = load_ds(LABEL_TIF, factor=4)
pred_ds2, b2 = load_ds(PRED_TIF,  factor=4)
ext2 = extent_of(b2)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 8), facecolor='white')
fig.suptitle('Knowledge-Based Weighted Overlay vs XGBoost ML Prediction\n'
             'Pothohar Plateau, Punjab, Pakistan',
             fontsize=13, fontweight='bold')

for ax, data, title in [
    (ax_a, label_ds, '(a) Weighted Overlay Labels\n(Training label source)'),
    (ax_b, pred_ds2, '(b) XGBoost ML Prediction\n(99.12% accuracy, 37 M pixels)')
]:
    m = np.ma.masked_invalid(data)
    ax.imshow(m, extent=ext2, cmap=CMAP3, norm=NORM3, aspect='auto', alpha=0.85, zorder=2)
    add_grid(ax, b2, n=3)
    mx2, my2 = (b2.right-b2.left)*0.02, (b2.top-b2.bottom)*0.02
    ax.set_xlim(b2.left-mx2, b2.right+mx2)
    ax.set_ylim(b2.bottom-my2, b2.top+my2)
    ax.set_facecolor('#b8d4e8')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel('Longitude (E)', fontsize=9)
    ax.set_ylabel('Latitude (N)', fontsize=9)
    add_scalebar(ax, b2)

ax_b.text(0.97, 0.03,
          'Agreement:\n99.05%',
          transform=ax_b.transAxes, ha='right', va='bottom',
          fontsize=10, fontweight='bold',
          bbox=dict(boxstyle='round', facecolor='white',
                    alpha=0.88, edgecolor='#1a9850'))

patches_cmp = [mpatches.Patch(color=c, label=l)
               for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
fig.legend(handles=patches_cmp, loc='lower center', ncol=3,
           fontsize=10, title='Groundwater Potential', title_fontsize=11,
           framealpha=0.95, edgecolor='black', bbox_to_anchor=(0.5, 0.005))

plt.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.97)
out4 = OUT / 'Fig4_Overlay_vs_ML_Comparison.png'
fig.savefig(out4, dpi=300, facecolor='white')
plt.close()
print(f'   Saved -> {out4}')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 5 — Validation Map + AUC curve + well breakdown  (Das 2018 Fig 13 style)
# ════════════════════════════════════════════════════════════════════════════════
print('[5] Validation map with wells + AUC ...')

if WELLS_CSV.exists():
    df_w = pd.read_csv(WELLS_CSV).dropna(subset=['pred_class'])
    df_w['pc'] = df_w['pred_class'].astype(int)
    prod = df_w[df_w['has_water'] == 1]
    dry  = df_w[df_w['has_water'] == 0]

    fig = plt.figure(figsize=(14, 10), facecolor='white')
    gs5 = GridSpec(3, 4, figure=fig,
                   left=0.07, right=0.85, top=0.92, bottom=0.08,
                   hspace=0.05, wspace=0.05)

    ax_v = fig.add_subplot(gs5[:, :3])
    map_ax(ax_v, pred, bounds,
           'Real-World Validation: GW Potential Zones vs Field Wells\n'
           'Pothohar Plateau — Sensitivity Analysis')

    if len(prod) > 0:
        ax_v.scatter(prod['lon'], prod['lat'],
                     c='#1f78b4', s=65, marker='^',
                     edgecolors='white', lw=0.8, zorder=6,
                     label=f'Productive wells (n={len(prod)})')
    if len(dry) > 0:
        ax_v.scatter(dry['lon'], dry['lat'],
                     c='#e31a1c', s=75, marker='X',
                     edgecolors='white', lw=0.8, zorder=6,
                     label=f'Dry references (n={len(dry)})')

    prod_in_mh = (prod['pc'] >= 1).sum()
    sens = prod_in_mh / len(prod) * 100 if len(prod) else 0

    gw_p = [mpatches.Patch(color=c, label=l) for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
    w_handles, w_labels = ax_v.get_legend_handles_labels()
    ax_v.legend(handles=gw_p + w_handles,
                labels=CLASS_LABELS + w_labels,
                loc='lower left', fontsize=8, title='Legend',
                title_fontsize=9, framealpha=0.92, edgecolor='black')

    ax_v.text(0.97, 0.97,
              f'Sensitivity: {sens:.0f}%\n({prod_in_mh}/{len(prod)} in Med+High)\nSpecificity: 33%',
              transform=ax_v.transAxes, ha='right', va='top',
              fontsize=9, fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='white',
                        alpha=0.9, edgecolor='black'))

    # Info panel
    ax_vi = fig.add_subplot(gs5[0, 3])
    ax_vi.axis('off')
    ax_vi.text(0.05, 0.97,
               ('Validation Data\n'
                f'Total wells: {len(df_w)}\n'
                f'Productive: {len(prod)}\n'
                f'Dry refs: {len(dry)}\n\n'
                f'Sensitivity: {sens:.1f}%\n\n'
                'Sources:\n'
                'Rana 2022\n'
                'Naz 2023\n'
                'GSP refs'),
               transform=ax_vi.transAxes, va='top', ha='left',
               fontsize=8, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f5f5f5',
                         alpha=0.9, edgecolor='grey'))

    # AUC curve
    ax_auc = fig.add_subplot(gs5[1, 3])
    zone_order = [2, 1, 0]
    cum_a, cum_w = [0.0], [0.0]
    tot_px = (~np.isnan(pred)).sum()
    tot_wl = len(df_w)
    for z in zone_order:
        cum_a.append(cum_a[-1] + (pred[~np.isnan(pred)] == z).sum() / tot_px)
        cum_w.append(cum_w[-1] + (df_w['pc'] == z).sum() / tot_wl)

    auc_v = np.trapezoid(cum_w, cum_a)
    ax_auc.plot(cum_a, cum_w, 'b-o', ms=5, lw=2)
    ax_auc.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
    ax_auc.fill_between(cum_a, cum_w, alpha=0.12, color='blue')
    ax_auc.text(0.55, 0.12, f'AUC={auc_v:.2f}',
                transform=ax_auc.transAxes,
                fontsize=10, fontweight='bold', color='blue')
    ax_auc.set_xlabel('Cum. % Area', fontsize=8)
    ax_auc.set_ylabel('Cum. % Wells', fontsize=8)
    ax_auc.set_title('AUC Curve', fontsize=9, fontweight='bold')
    ax_auc.tick_params(labelsize=7)
    ax_auc.set_xlim(0, 1); ax_auc.set_ylim(0, 1)
    ax_auc.set_facecolor('#f8f9fa')
    ax_auc.grid(True, alpha=0.35)

    # Well class breakdown
    ax_vb = fig.add_subplot(gs5[2, 3])
    prod_c = [(prod['pc'] == c).sum() for c in [0,1,2]]
    dry_c  = [(dry['pc']  == c).sum() for c in [0,1,2]]
    bot_p = bot_d = 0
    for ci in range(3):
        ax_vb.bar(0, prod_c[ci], bottom=bot_p,
                  color=CLASS_COLORS[ci], edgecolor='white', lw=0.8)
        ax_vb.bar(1, dry_c[ci], bottom=bot_d,
                  color=CLASS_COLORS[ci], edgecolor='white', lw=0.8)
        bot_p += prod_c[ci]; bot_d += dry_c[ci]
    ax_vb.set_xticks([0,1])
    ax_vb.set_xticklabels(['Productive\nWells', 'Dry\nRefs'], fontsize=8)
    ax_vb.set_ylabel('Count', fontsize=8)
    ax_vb.set_title('Wells by Zone', fontsize=9, fontweight='bold')
    ax_vb.tick_params(labelsize=7)
    ax_vb.yaxis.grid(True, alpha=0.3)

    out5 = OUT / 'Fig5_Validation_Map_Wells_AUC.png'
    fig.savefig(out5, dpi=300, facecolor='white')
    plt.close()
    print(f'   Saved -> {out5}')
else:
    print('   SKIPPED -- wells_nn_recovered.csv not found')


print('\n' + '='*60)
print('DONE  ->  outputs/figures/publication/')
print('='*60)
print('  Fig1_GW_Prediction_Map.png')
print('  Fig2_GW_Probability_Map.png')
print('  Fig3_Feature_Maps_Panel.png')
print('  Fig4_Overlay_vs_ML_Comparison.png')
print('  Fig5_Validation_Map_Wells_AUC.png')
print('\nCopy to: C:\\Users\\ADNAN\\THESIS\\figures\\')
