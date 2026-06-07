"""
Step 7: Rain Gauge Station Spatial Validation
Uses 22 independent rain gauge stations from Khan et al. (2023)
as proxy validators for model spatial consistency.

Rationale: These stations are entirely independent from training data.
Higher-altitude, wetter stations should cluster in Medium/High GW zones;
known alluvial basin stations (Rawalpindi, Taxila, Gujjar Khan) should
be classified High.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.transform import rowcol
from scipy.stats import spearmanr, pointbiserialr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Output paths ─────────────────────────────────────────────────────────────
OUT_FIG = Path('outputs/figures/validation')
OUT_DATA = Path('data/validation')
OUT_FIG.mkdir(parents=True, exist_ok=True)

PRED_TIF = Path('outputs/maps/gw_prediction_xgboost.tif')
PROB_TIF = Path('outputs/maps/gw_probability_high.tif')

CLASS_COLORS = {0: '#d73027', 1: '#fee090', 2: '#1a9850'}
CLASS_NAMES  = {0: 'Low', 1: 'Medium', 2: 'High'}

print('=' * 65)
print('  STEP 7: RAIN GAUGE SPATIAL VALIDATION (Khan et al. 2023)')
print('=' * 65)

# ── 22 stations from Table 2, Khan 2023 (Atmosphere, 14, 452) ───────────────
# Station 17 (Mulhal Mughlan) had lat/lon swapped in PDF — corrected here.
# Stations outside study area extent excluded: Abbottabad, Nathiagali, Nara,
# Murree, Kotli Sattian, Mohra Sharif (all > 33.9°N or in KPK).
stations_raw = [
    # name,             lat,     lon,    alt_m, basin, zone_expected
    ('Rawalpindi',      33.5651, 73.0169, 540,  'Soan',         'alluvial'),
    ('Chakwal',         32.9328, 72.8630, 522,  'Soan+Bunha',   'mixed'),
    ('Fateh Jung',      33.5635, 72.6375, 514,  'Soan+Haro',    'alluvial'),
    ('Talagang',        32.9172, 72.4081, 457,  'Soan',         'mixed'),
    ('Gujjar Khan',     33.2616, 73.3058, 458,  'Soan',         'alluvial'),
    ('Pendigheb',       33.2452, 72.2660, 310,  'Soan',         'alluvial'),
    ('Taxila',          33.7463, 72.8397, 549,  'Soan+Haro',    'alluvial'),
    ('Khanpur Dam',     33.8018, 72.9305, 545,  'Soan',         'alluvial'),
    ('Jabbri',          33.9045, 73.1733, 923,  'Haro',         'hilly'),
    ('Wah Cantt',       33.7843, 72.7388, 471,  'Haro',         'alluvial'),
    ('Hattar',          33.8521, 72.8501, 513,  'Haro',         'alluvial'),
    ('Mulhal Mughlan',  33.0027, 73.1497, 523,  'Kuhan+Bunha',  'mixed'),
    ('Khokhar Bala',    32.7716, 72.8146, 853,  'Kuhan+Bunha',  'hilly'),
    ('Wagh',            32.7343, 73.3966, 272,  'Kuhan+Bunha',  'alluvial'),
    ('Sohawa',          33.1171, 73.4149, 440,  'Kuhan+Bunha',  'mixed'),
    ('Dina',            33.0306, 73.6069, 281,  'Kuhan+Bunha',  'alluvial'),
]

df = pd.DataFrame(stations_raw,
                  columns=['station', 'lat', 'lon', 'altitude_m', 'basin', 'zone_expected'])
print(f'\nUsing {len(df)} stations within Pothohar study area bounds.')

# ── Load rasters ─────────────────────────────────────────────────────────────
print('\nLoading prediction rasters ...')
with rasterio.open(PRED_TIF) as src:
    pred_data      = src.read(1).astype(float)
    pred_nodata    = src.nodata
    pred_transform = src.transform
    pred_bounds    = src.bounds
    pred_data[pred_data == pred_nodata] = np.nan

with rasterio.open(PROB_TIF) as src:
    prob_data   = src.read(1).astype(float)
    prob_nodata = src.nodata
    if prob_nodata is not None:
        prob_data[prob_data == prob_nodata] = np.nan

print(f'Raster bounds: {pred_bounds}')

# ── Extract predictions at station locations ──────────────────────────────────
print('\nExtracting predictions at station coordinates ...')
records = []
for _, row in df.iterrows():
    lon, lat = row['lon'], row['lat']
    try:
        r, c = rowcol(pred_transform, lon, lat)
        if 0 <= r < pred_data.shape[0] and 0 <= c < pred_data.shape[1]:
            cls  = pred_data[r, c]
            prob = prob_data[r, c]
            if not np.isnan(cls):
                records.append({'station': row['station'], 'lat': lat, 'lon': lon,
                                'altitude_m': row['altitude_m'], 'basin': row['basin'],
                                'zone_expected': row['zone_expected'],
                                'pred_class': int(cls), 'pred_label': CLASS_NAMES[int(cls)],
                                'prob_high': float(prob)})
            else:
                records.append({'station': row['station'], 'lat': lat, 'lon': lon,
                                'altitude_m': row['altitude_m'], 'basin': row['basin'],
                                'zone_expected': row['zone_expected'],
                                'pred_class': np.nan, 'pred_label': 'NoData',
                                'prob_high': np.nan})
        else:
            records.append({'station': row['station'], 'lat': lat, 'lon': lon,
                            'altitude_m': row['altitude_m'], 'basin': row['basin'],
                            'zone_expected': row['zone_expected'],
                            'pred_class': np.nan, 'pred_label': 'OutOfBounds',
                            'prob_high': np.nan})
    except Exception as e:
        records.append({'station': row['station'], 'lat': lat, 'lon': lon,
                        'altitude_m': row['altitude_m'], 'basin': row['basin'],
                        'zone_expected': row['zone_expected'],
                        'pred_class': np.nan, 'pred_label': 'Error',
                        'prob_high': np.nan})

result_df = pd.DataFrame(records)
valid = result_df[result_df['pred_label'].isin(['Low','Medium','High'])].copy()
print(f'  Valid extractions: {len(valid)}/{len(result_df)}')

# ── Statistics ────────────────────────────────────────────────────────────────
print('\n--- Results per station ---')
print(result_df[['station','altitude_m','zone_expected','pred_label','prob_high']].to_string(index=False))

# Spearman correlation: altitude vs GW probability
# (Higher altitude = more rainfall in Pothohar = expected higher recharge)
if len(valid) >= 6:
    rho, p_alt = spearmanr(valid['altitude_m'], valid['prob_high'])
    print(f'\nSpearman rho (altitude vs P_high): {rho:.3f}  (p={p_alt:.3f})')

    # Alluvial vs non-alluvial
    alluvial_prob = valid[valid['zone_expected'] == 'alluvial']['prob_high'].values
    other_prob    = valid[valid['zone_expected'] != 'alluvial']['prob_high'].values
    from scipy.stats import mannwhitneyu
    if len(alluvial_prob) >= 3 and len(other_prob) >= 3:
        u, p_mw = mannwhitneyu(alluvial_prob, other_prob, alternative='greater')
        print(f'Mann-Whitney (alluvial > other zones): U={u:.0f}, p={p_mw:.3f}')
        print(f'  Alluvial median prob_high : {np.median(alluvial_prob):.3f}  (n={len(alluvial_prob)})')
        print(f'  Non-alluvial median prob  : {np.median(other_prob):.3f}  (n={len(other_prob)})')

    # Class distribution
    dist = valid['pred_label'].value_counts()
    print(f'\nClass distribution at rain gauge stations:')
    for cls, cnt in dist.items():
        pct = 100*cnt/len(valid)
        print(f'  {cls:8s}: {cnt:2d} ({pct:.0f}%)')

# ── Figure ────────────────────────────────────────────────────────────────────
print('\nGenerating validation figure ...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Rain Gauge Station Spatial Validation\n(Khan et al. 2023 — 22 independent stations)',
             fontsize=12, fontweight='bold')

# 1. Map of stations colored by prediction
ax = axes[0]
if len(valid) > 0:
    # Downsample prediction raster for background
    step = 8
    rows_ax = np.arange(0, pred_data.shape[0], step)
    cols_ax = np.arange(0, pred_data.shape[1], step)
    bg = pred_data[np.ix_(rows_ax, cols_ax)]
    xmin, ymin, xmax, ymax = pred_bounds
    ax.imshow(bg, extent=[xmin, xmax, ymin, ymax], origin='upper',
              cmap=plt.cm.RdYlGn, vmin=0, vmax=2, alpha=0.4, aspect='auto')
    for _, r in result_df.iterrows():
        if r['pred_label'] in CLASS_NAMES.values():
            c = CLASS_COLORS[int(r['pred_class'])]
            ax.scatter(r['lon'], r['lat'], c=c, s=80, zorder=5,
                       edgecolors='black', linewidths=0.6)
            ax.annotate(r['station'], (r['lon'], r['lat']),
                        textcoords='offset points', xytext=(4, 3), fontsize=5.5)
        elif r['pred_label'] == 'NoData':
            ax.scatter(r['lon'], r['lat'], c='gray', s=50, marker='x', zorder=5)
patches = [mpatches.Patch(color=CLASS_COLORS[k], label=f'{CLASS_NAMES[k]}')
           for k in [0,1,2]]
patches.append(mpatches.Patch(color='gray', label='NoData'))
ax.legend(handles=patches, fontsize=7, loc='lower left')
ax.set_xlabel('Longitude', fontsize=9)
ax.set_ylabel('Latitude', fontsize=9)
ax.set_title('Predicted GW Class at\nRain Gauge Stations', fontsize=10)

# 2. Altitude vs prob_high scatter
ax2 = axes[1]
if len(valid) >= 4:
    colors_s = [CLASS_COLORS.get(int(c), 'gray') for c in valid['pred_class']]
    ax2.scatter(valid['altitude_m'], valid['prob_high'], c=colors_s,
                s=80, edgecolors='black', linewidths=0.6, zorder=4)
    for _, r in valid.iterrows():
        ax2.annotate(r['station'], (r['altitude_m'], r['prob_high']),
                     textcoords='offset points', xytext=(3, 2), fontsize=5.5)
    m, b = np.polyfit(valid['altitude_m'], valid['prob_high'], 1)
    xs = np.linspace(valid['altitude_m'].min(), valid['altitude_m'].max(), 50)
    ax2.plot(xs, m*xs + b, 'k--', lw=1.2, alpha=0.6)
    if len(valid) >= 6:
        ax2.set_title(f'Altitude vs P(High GW)\nSpearman ρ={rho:.3f}, p={p_alt:.3f}', fontsize=10)
    else:
        ax2.set_title('Altitude vs P(High GW)', fontsize=10)
ax2.set_xlabel('Station Altitude (m)', fontsize=9)
ax2.set_ylabel('Predicted P(High GW)', fontsize=9)
ax2.set_ylim(-0.05, 1.05)

# 3. Bar chart: mean prob_high by zone_expected
ax3 = axes[2]
if len(valid) >= 4:
    zone_summary = valid.groupby('zone_expected')['prob_high'].agg(['mean','sem','count'])
    zone_order = ['alluvial','mixed','hilly']
    zone_summary = zone_summary.reindex([z for z in zone_order if z in zone_summary.index])
    bar_colors = {'alluvial': '#1a9850', 'mixed': '#fee090', 'hilly': '#d73027'}
    bars = ax3.bar(zone_summary.index,
                   zone_summary['mean'],
                   yerr=zone_summary['sem'],
                   color=[bar_colors.get(z,'#999') for z in zone_summary.index],
                   edgecolor='black', linewidth=0.8, capsize=5)
    for bar, (idx, row_s) in zip(bars, zone_summary.iterrows()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + row_s['sem'] + 0.02,
                 f'n={int(row_s["count"])}', ha='center', va='bottom', fontsize=8)
ax3.set_xlabel('Expected Zone Type', fontsize=9)
ax3.set_ylabel('Mean P(High GW)', fontsize=9)
ax3.set_ylim(0, 1.0)
ax3.set_title('Mean P(High GW) by\nExpected Hydrogeology', fontsize=10)
ax3.axhline(0.5, color='gray', linestyle=':', lw=0.8)

plt.tight_layout()
out_png = OUT_FIG / 'raingauge_spatial_validation.png'
fig.savefig(out_png, dpi=200, facecolor='white')
plt.close()
print(f'Saved: {out_png}')

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_csv = OUT_DATA / 'raingauge_validation.csv'
result_df.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')
print('\nDone.')
