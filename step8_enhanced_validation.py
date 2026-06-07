"""
Step 8: Enhanced Real-World Validation
Three improvements over step5/step6:
  A. Expand NN search for failed dry wells (up to 15 pixel radius)
  B. ROC-AUC with continuous probability (publication-quality)
  C. Large-n sensitivity using full 146 Naz 2023 wells
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.transform import rowcol
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              classification_report)
from scipy.stats import mannwhitneyu, fisher_exact
from pathlib import Path
import json, warnings
warnings.filterwarnings('ignore')

PRED_TIF  = Path('outputs/maps/gw_prediction_xgboost.tif')
PROB_TIF  = Path('outputs/maps/gw_probability_high.tif')
WELLS_CSV = Path('data/validation/validation_wells_combined.csv')
NAZ_CSV   = Path('data/validation/naz2023_wells_pothohar.csv')
OUT_FIG   = Path('outputs/figures/validation')
OUT_DATA  = Path('data/validation')
OUT_FIG.mkdir(parents=True, exist_ok=True)

print('=' * 65)
print('  STEP 8: ENHANCED REAL-WORLD VALIDATION')
print('=' * 65)

# ── Load rasters ──────────────────────────────────────────────────────────────
with rasterio.open(PRED_TIF) as src:
    pred       = src.read(1).astype(float)
    nd_pred    = src.nodata
    transform  = src.transform
    bounds     = src.bounds
    pred[pred == nd_pred] = np.nan

with rasterio.open(PROB_TIF) as src:
    prob    = src.read(1).astype(float)
    nd_prob = src.nodata
    if nd_prob is not None:
        prob[prob == nd_prob] = np.nan

print(f'Raster shape: {pred.shape}')

# ── Helper: extract value with NN fallback ────────────────────────────────────
def extract_with_nn(arr_pred, arr_prob, t, lon, lat, max_radius=15):
    """Return (pred_class, prob_high, nn_dist) with nearest-neighbor fallback."""
    try:
        r0, c0 = rowcol(t, lon, lat)
    except:
        return np.nan, np.nan, -1

    # Try direct hit first
    if 0 <= r0 < arr_pred.shape[0] and 0 <= c0 < arr_pred.shape[1]:
        if not np.isnan(arr_pred[r0, c0]):
            return arr_pred[r0, c0], arr_prob[r0, c0], 0

    # Spiral search up to max_radius pixels
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                r, c = r0 + dr, c0 + dc
                if 0 <= r < arr_pred.shape[0] and 0 <= c < arr_pred.shape[1]:
                    if not np.isnan(arr_pred[r, c]):
                        return arr_pred[r, c], arr_prob[r, c], radius
    return np.nan, np.nan, -1

# ── PART A: Extract all 85 validation wells with expanded NN ──────────────────
print('\n[A] Extracting all 85 wells (NN radius up to 15 px) ...')
wells = pd.read_csv(WELLS_CSV)
records = []
for _, row in wells.iterrows():
    cls, pr, nn = extract_with_nn(pred, prob, transform,
                                  row['lon'], row['lat'], max_radius=15)
    records.append({
        'lon': row['lon'], 'lat': row['lat'],
        'source': row['source'], 'district': row['district'],
        'has_water': row['has_water'],
        'pred_class': cls, 'prob_high': pr, 'nn_px': nn
    })

df_all = pd.DataFrame(records)
df_valid = df_all[df_all['nn_px'] >= 0].copy()
df_direct = df_valid[df_valid['nn_px'] == 0]
df_nn     = df_valid[df_valid['nn_px'] > 0]
df_failed = df_all[df_all['nn_px'] < 0]

print(f'  Direct hits   : {len(df_direct)}')
print(f'  NN recovered  : {len(df_nn)}')
print(f'  Still failed  : {len(df_failed)}')

# Breakdown by dry/productive for recovered wells
prod_valid = df_valid[df_valid['has_water'] == 1]
dry_valid  = df_valid[df_valid['has_water'] == 0]
print(f'  Valid productive: {len(prod_valid)}, Valid dry: {len(dry_valid)}')

# Binary: Med+High = predicted positive (GW present)
df_valid['pred_pos'] = (df_valid['pred_class'] >= 1).astype(int)

TP = ((df_valid['has_water'] == 1) & (df_valid['pred_pos'] == 1)).sum()
FN = ((df_valid['has_water'] == 1) & (df_valid['pred_pos'] == 0)).sum()
TN = ((df_valid['has_water'] == 0) & (df_valid['pred_pos'] == 0)).sum()
FP = ((df_valid['has_water'] == 0) & (df_valid['pred_pos'] == 1)).sum()

sens = TP / (TP + FN) if (TP + FN) > 0 else 0
spec = TN / (TN + FP) if (TN + FP) > 0 else 0
ppv  = TP / (TP + FP) if (TP + FP) > 0 else 0
ba   = (sens + spec) / 2

print(f'\n  Sensitivity  : {100*sens:.1f}%  (TP={TP}, FN={FN})')
print(f'  Specificity  : {100*spec:.1f}%  (TN={TN}, FP={FP})')
print(f'  PPV          : {100*ppv:.1f}%')
print(f'  Balanced Acc : {100*ba:.1f}%')

# Fisher exact test on 2x2 contingency
oddsratio, p_fisher = fisher_exact([[TP, FN], [FP, TN]], alternative='greater')
print(f'  Fisher exact p (TPR > chance): {p_fisher:.4f}')

# ── PART B: ROC-AUC with continuous probability ───────────────────────────────
print('\n[B] ROC-AUC analysis ...')
roc_df = df_valid.dropna(subset=['prob_high', 'has_water'])

if len(roc_df['has_water'].unique()) == 2:
    fpr, tpr, thresholds = roc_curve(roc_df['has_water'], roc_df['prob_high'])
    auc_score = roc_auc_score(roc_df['has_water'], roc_df['prob_high'])
    print(f'  ROC-AUC : {auc_score:.3f}  (n={len(roc_df)}: {roc_df["has_water"].sum()} prod, '
          f'{(roc_df["has_water"]==0).sum()} dry)')

    # Mann-Whitney: productive wells have higher prob than dry wells
    prod_prob = roc_df[roc_df['has_water'] == 1]['prob_high'].values
    dry_prob  = roc_df[roc_df['has_water'] == 0]['prob_high'].values
    if len(dry_prob) >= 2:
        u_stat, p_mw = mannwhitneyu(prod_prob, dry_prob, alternative='greater')
        print(f'  Mann-Whitney (prod > dry): U={u_stat:.0f}, p={p_mw:.4f}')
        print(f'  Productive median prob_high : {np.median(prod_prob):.3f}')
        print(f'  Dry        median prob_high : {np.median(dry_prob):.3f}')
else:
    print('  Only one class — cannot compute ROC-AUC')
    auc_score = None

# ── PART C: Large-n sensitivity — full 146 Naz 2023 wells ────────────────────
print('\n[C] Large-n sensitivity (Naz 2023, 146 wells, all productive) ...')
naz = pd.read_csv(NAZ_CSV)
naz_records = []
for _, row in naz.iterrows():
    cls, pr, nn = extract_with_nn(pred, prob, transform,
                                  row['lon'], row['lat'], max_radius=10)
    naz_records.append({'lon': row['lon'], 'lat': row['lat'],
                        'district': row.get('district', '?'),
                        'pred_class': cls, 'prob_high': pr, 'nn_px': nn})

naz_df = pd.DataFrame(naz_records)
naz_valid = naz_df[naz_df['nn_px'] >= 0].copy()
naz_valid['pred_pos'] = (naz_valid['pred_class'] >= 1).astype(int)

naz_sens = naz_valid['pred_pos'].mean()
print(f'  Valid Naz wells: {len(naz_valid)}/{len(naz_df)}')
print(f'  Large-n sensitivity (Med+High): {100*naz_sens:.1f}%')

# Per-district
for dist in naz_valid['district'].unique():
    sub = naz_valid[naz_valid['district'] == dist]
    s = sub['pred_pos'].mean()
    print(f'    {dist:20s}: {100*s:.0f}%  (n={len(sub)})')

# ── FIGURE ────────────────────────────────────────────────────────────────────
print('\nGenerating validation figure ...')
fig = plt.figure(figsize=(16, 5))
gs  = fig.add_gridspec(1, 4, wspace=0.35)

CLASS_COLORS = {0: '#d73027', 1: '#fee090', 2: '#1a9850'}
WELL_EDGE    = {1: 'blue', 0: 'red'}

# -- Panel 1: spatial map of all 85 wells
ax1 = fig.add_subplot(gs[0, 0])
step = 8
bg   = pred[::step, ::step]
xmin, ymin, xmax, ymax = bounds
ax1.imshow(bg, extent=[xmin, xmax, ymin, ymax], origin='upper',
           cmap=plt.cm.RdYlGn, vmin=0, vmax=2, alpha=0.35, aspect='auto')
for _, r in df_all.iterrows():
    if r['nn_px'] >= 0:
        c = CLASS_COLORS.get(int(r['pred_class']), 'gray')
        ec = 'blue' if r['has_water'] == 1 else 'red'
        ms = 8 if r['nn_px'] == 0 else 6
        mk = 'o' if r['has_water'] == 1 else 'x'
        ax1.scatter(r['lon'], r['lat'], c=c, s=ms**2, marker=mk,
                    edgecolors=ec, linewidths=0.8, zorder=5)
    else:
        ax1.scatter(r['lon'], r['lat'], c='lightgray', s=25, marker='+',
                    linewidths=0.6, zorder=3)
patches = [mpatches.Patch(color=CLASS_COLORS[k], label=['Low','Med','High'][k])
           for k in [0, 1, 2]]
patches += [plt.Line2D([0],[0], marker='o', color='w', markeredgecolor='blue',
                        markerfacecolor='none', label='Productive', markersize=7),
            plt.Line2D([0],[0], marker='x', color='red', label='Dry', markersize=7)]
ax1.legend(handles=patches, fontsize=6, loc='lower left')
ax1.set_title(f'Well Locations\n(n={len(df_valid)} valid, {len(df_failed)} failed)',
              fontsize=9)
ax1.set_xlabel('Lon', fontsize=8); ax1.set_ylabel('Lat', fontsize=8)

# -- Panel 2: ROC curve
ax2 = fig.add_subplot(gs[0, 1])
if auc_score is not None:
    ax2.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC={auc_score:.3f})')
    ax2.fill_between(fpr, tpr, alpha=0.15)
ax2.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random (AUC=0.5)')
ax2.set_xlabel('False Positive Rate', fontsize=8)
ax2.set_ylabel('True Positive Rate', fontsize=8)
ax2.set_title('ROC Curve\n(Productive vs Dry Wells)', fontsize=9)
ax2.legend(fontsize=8)
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

# -- Panel 3: prob_high boxplot productive vs dry
ax3 = fig.add_subplot(gs[0, 2])
prod_p = roc_df[roc_df['has_water'] == 1]['prob_high'].values
dry_p  = roc_df[roc_df['has_water'] == 0]['prob_high'].values
bp = ax3.boxplot([prod_p, dry_p], labels=['Productive\n(n=%d)' % len(prod_p),
                                           'Dry\n(n=%d)' % len(dry_p)],
                 patch_artist=True, notch=False,
                 medianprops=dict(color='black', lw=2))
bp['boxes'][0].set_facecolor('#2166ac')
if len(bp['boxes']) > 1:
    bp['boxes'][1].set_facecolor('#d73027')
ax3.set_ylabel('P(High GW Potential)', fontsize=8)
ax3.set_title('Probability Distribution\nProductive vs Dry Wells', fontsize=9)
ax3.set_ylim(-0.05, 1.05)
if len(dry_p) >= 2:
    ax3.text(0.5, 0.95, f'MW p={p_mw:.3f}', transform=ax3.transAxes,
             ha='center', va='top', fontsize=8,
             color='green' if p_mw < 0.05 else 'gray')

# -- Panel 4: summary bar chart
ax4 = fig.add_subplot(gs[0, 3])
metrics = {
    'Sensitivity\n(n=%d wells)' % len(prod_valid): 100*sens,
    'Specificity\n(n=%d wells)' % len(dry_valid):  100*spec,
    'Bal. Accuracy': 100*ba,
    'Large-n Sens.\n(n=%d Naz)' % len(naz_valid): 100*naz_sens,
}
colors = ['#2166ac', '#d73027', '#4dac26', '#756bb1']
bars = ax4.barh(list(metrics.keys()), list(metrics.values()),
                color=colors, edgecolor='black', linewidth=0.7)
for bar, val in zip(bars, metrics.values()):
    ax4.text(val + 1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=8)
ax4.set_xlim(0, 110)
ax4.set_xlabel('Percentage (%)', fontsize=8)
ax4.set_title('Validation Summary', fontsize=9)
ax4.axvline(50, color='gray', linestyle=':', lw=0.8)

fig.suptitle('Real-World Well Validation — XGBoost Groundwater Potential Model\n'
             'Pothohar Plateau, Punjab, Pakistan',
             fontsize=11, fontweight='bold')

out_png = OUT_FIG / 'enhanced_validation.png'
fig.savefig(out_png, dpi=200, facecolor='white')
plt.close()
print(f'Saved: {out_png}')

# ── Save summary JSON ─────────────────────────────────────────────────────────
summary = {
    'part_A_well_validation': {
        'total_wells': len(wells), 'direct_hits': int(len(df_direct)),
        'nn_recovered': int(len(df_nn)), 'failed': int(len(df_failed)),
        'productive': int(len(prod_valid)), 'dry': int(len(dry_valid)),
        'TP': int(TP), 'FN': int(FN), 'TN': int(TN), 'FP': int(FP),
        'sensitivity_pct': round(100*sens, 1),
        'specificity_pct': round(100*spec, 1),
        'balanced_accuracy_pct': round(100*ba, 1),
        'ppv_pct': round(100*ppv, 1),
        'fisher_exact_p': round(float(p_fisher), 4),
    },
    'part_B_roc_auc': {
        'auc': round(float(auc_score), 3) if auc_score else None,
        'n_wells': int(len(roc_df)),
        'mann_whitney_p': round(float(p_mw), 4) if len(dry_p) >= 2 else None,
        'prod_median_prob': round(float(np.median(prod_prob)), 3),
        'dry_median_prob': round(float(np.median(dry_prob)), 3) if len(dry_p) > 0 else None,
    },
    'part_C_large_n': {
        'naz2023_wells_valid': int(len(naz_valid)),
        'sensitivity_pct': round(100*naz_sens, 1),
    }
}
out_json = OUT_DATA / 'enhanced_validation_report.json'
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Saved: {out_json}')
print('\nDone.')
