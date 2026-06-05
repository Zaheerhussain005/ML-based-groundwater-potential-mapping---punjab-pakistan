"""
Step 5: Real-World Validation of GW Potential Prediction Map

Combined dataset:
  - Rana et al. 2022   (7 pts, Rawalpindi, groundwater quality surveys)
  - Naz et al. 2023    (57 pts, Mianwali+Jhelum alluvial + 10 Rawalpindi)
  - GSP geology refs   (12 dry pts, Salt Range + Kala Chitta hard rock)
  - Template curated   (16 pts, includes 2 dry wells)
Total: ~88 pts with both productive (has_water=1) and dry (has_water=0) wells.

Statistical tests:
  - Sensitivity (TPR) + Specificity (TNR) + Balanced Accuracy
  - ROC-AUC from predicted probability at well locations
  - Mann-Whitney U: P(High) at productive vs dry wells
  - Chi-square: independence of has_water vs predicted class
  - Point-biserial correlation: has_water vs P(High)

Outputs:
  outputs/figures/validation/well_overlay_map.png
  outputs/figures/validation/validation_statistics.png
  outputs/figures/validation/probability_distribution.png
  outputs/figures/validation/roc_well_points.png
  outputs/figures/validation/validation_report.json
  data/validation/well_points_classified.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import rasterio
from rasterio.plot import show as rshow
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# scipy / sklearn for statistics
from scipy.stats import mannwhitneyu, chi2_contingency, pointbiserialr
from sklearn.metrics import roc_auc_score, roc_curve

# -- Paths ---------------------------------------------------------------------
PRED_TIF   = Path('outputs/maps/gw_prediction_xgboost.tif')
PROB_TIF   = Path('outputs/maps/gw_probability_high.tif')
WELLS_CSV  = Path('data/validation/validation_wells_combined.csv')
OUT_FIG    = Path('outputs/figures/validation')
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DATA   = Path('data/validation')

CLASS_NAMES  = {0: 'Low', 1: 'Medium', 2: 'High'}
CLASS_COLORS = {0: '#d73027', 1: '#fee090', 2: '#1a9850'}
WELL_COLORS  = {1: '#1f78b4', 0: '#e31a1c'}

print('=' * 70)
print('  STEP 5: REAL-WORLD VALIDATION (improved)')
print('=' * 70)

# -- Load rasters --------------------------------------------------------------
print('\n[1] Loading prediction rasters ...')
with rasterio.open(PRED_TIF) as src:
    pred_data      = src.read(1).astype(float)
    pred_nodata    = src.nodata
    pred_data[pred_data == pred_nodata] = np.nan
    pred_meta      = src.meta
    pred_bounds    = src.bounds
    pred_transform = src.transform

with rasterio.open(PROB_TIF) as src:
    prob_data = src.read(1).astype(float)
    prob_nodata = src.nodata
    if prob_nodata is not None:
        prob_data[prob_data == prob_nodata] = np.nan

print(f'   Prediction raster : {pred_data.shape}, bounds: {pred_bounds}')

# -- Load well points ----------------------------------------------------------
print('\n[2] Loading combined validation dataset ...')
df = pd.read_csv(WELLS_CSV)
print(f'   Loaded {len(df)} points from {WELLS_CSV.name}')
print(f'   Sources     : {df["source"].value_counts().to_dict()}')
print(f'   Districts   : {df["district"].value_counts().to_dict()}')
print(f'   Hydro zones : {df["hydro_zone"].value_counts().to_dict()}')
print(f'   Productive  : {df["has_water"].sum()}  |  Dry: {(df["has_water"]==0).sum()}')

# Bounds check
in_bounds = (
    (df['lon'] >= pred_bounds.left)   & (df['lon'] <= pred_bounds.right) &
    (df['lat'] >= pred_bounds.bottom) & (df['lat'] <= pred_bounds.top)
)
n_out = (~in_bounds).sum()
if n_out:
    print(f'   WARNING: {n_out} points outside raster extent -- skipped')
df_valid = df[in_bounds].copy()

# -- Sample rasters at each well -----------------------------------------------
print('\n[3] Sampling rasters at well locations ...')

def sample_raster(rdata, transform, lon, lat):
    row, col = rasterio.transform.rowcol(transform, lon, lat)
    r, c = rdata.shape
    if 0 <= row < r and 0 <= col < c:
        v = rdata[row, col]
        return float(v) if not np.isnan(v) else None
    return None

pred_vals = [sample_raster(pred_data, pred_transform, r['lon'], r['lat'])
             for _, r in df_valid.iterrows()]
prob_vals = [sample_raster(prob_data, pred_transform, r['lon'], r['lat'])
             for _, r in df_valid.iterrows()]

df_valid['pred_class']     = pred_vals
df_valid['pred_class_int'] = df_valid['pred_class'].apply(
    lambda x: int(x) if (x is not None and not pd.isna(x)) else np.nan)
df_valid['pred_label']     = df_valid['pred_class_int'].map(CLASS_NAMES)
df_valid['pred_prob_high'] = prob_vals

# Remove points on NoData pixels
n_before = len(df_valid)
df_valid = df_valid.dropna(subset=['pred_class']).copy()
n_nodata = n_before - len(df_valid)
print(f'   Total within bounds : {n_before}')
print(f'   Hit NoData pixels   : {n_nodata}  (outside raster/district polygons)')
print(f'   Successfully sampled: {len(df_valid)}')
print(f'   Productive sampled  : {(df_valid["has_water"]==1).sum()}')
print(f'   Dry sampled         : {(df_valid["has_water"]==0).sum()}')

productive = df_valid[df_valid['has_water'] == 1]
dry        = df_valid[df_valid['has_water'] == 0]

# -- Core validation metrics ---------------------------------------------------
print('\n[4] Computing validation metrics ...')

# Sensitivity (True Positive Rate): productive wells predicted Med or High
tp = (productive['pred_class_int'] >= 1).sum()
fn = (productive['pred_class_int'] == 0).sum()
sensitivity = tp / len(productive) * 100 if len(productive) else 0

# Specificity (True Negative Rate): dry wells predicted Low
tn = (dry['pred_class_int'] == 0).sum() if len(dry) else 0
fp = (dry['pred_class_int'] >= 1).sum() if len(dry) else 0
specificity = tn / len(dry) * 100 if len(dry) else 0

balanced_acc = (sensitivity + specificity) / 2

# High zone capture: productive wells in High zone
high_capture = (productive['pred_class_int'] == 2).sum()
high_rate    = high_capture / len(productive) * 100 if len(productive) else 0

# False positive rate: dry wells in High zone
fp_high = (dry['pred_class_int'] == 2).sum() if len(dry) else 0
fpr_high = fp_high / len(dry) * 100 if len(dry) else 0

print(f'\n   --- Core Metrics ---')
print(f'   Sensitivity (TPR)    : {sensitivity:.1f}%  ({tp}/{len(productive)} productive -> Med+High)')
print(f'   Specificity (TNR)    : {specificity:.1f}%  ({tn}/{len(dry)} dry -> Low)')
print(f'   Balanced Accuracy    : {balanced_acc:.1f}%')
print(f'   High Zone Capture    : {high_rate:.1f}%')
print(f'   False Positive Rate  : {fpr_high:.1f}%  (dry wells wrongly in High zone)')

# -- Statistical tests ---------------------------------------------------------
print('\n[5] Running statistical tests ...')

prod_probs = productive['pred_prob_high'].dropna()
dry_probs  = dry['pred_prob_high'].dropna() if len(dry) else pd.Series([], dtype=float)

stat_results = {}

# Mann-Whitney U: productive vs dry well probabilities
if len(dry_probs) >= 3:
    u_stat, p_mw = mannwhitneyu(prod_probs, dry_probs, alternative='greater')
    stat_results['mann_whitney_U']       = float(u_stat)
    stat_results['mann_whitney_p']       = float(p_mw)
    stat_results['mann_whitney_sig']     = bool(p_mw < 0.05)
    print(f'   Mann-Whitney U test  : U={u_stat:.0f}, p={p_mw:.4f}  '
          f'({"SIGNIFICANT" if p_mw < 0.05 else "not significant"} at alpha=0.05)')
else:
    print('   Mann-Whitney U       : skipped (fewer than 3 dry wells sampled)')

# Point-biserial correlation: has_water vs P(High)
pb_sub = df_valid.dropna(subset=['pred_prob_high'])
if len(pb_sub) >= 10:
    r_pb, p_pb = pointbiserialr(pb_sub['has_water'], pb_sub['pred_prob_high'])
    stat_results['pointbiserial_r']  = float(r_pb)
    stat_results['pointbiserial_p']  = float(p_pb)
    print(f'   Point-biserial r     : r={r_pb:.3f}, p={p_pb:.4f}  '
          f'({"SIGNIFICANT" if p_pb < 0.05 else "not significant"})')

# Chi-square: independence of has_water vs predicted class
ct = pd.crosstab(df_valid['has_water'], df_valid['pred_class_int'].dropna())
if ct.shape == (2, 3) or (ct.shape[0] == 2 and ct.shape[1] >= 2):
    chi2, p_chi, dof, expected = chi2_contingency(ct)
    stat_results['chi2_statistic']   = float(chi2)
    stat_results['chi2_p']           = float(p_chi)
    stat_results['chi2_dof']         = int(dof)
    print(f'   Chi-square test      : chi2={chi2:.2f}, df={dof}, p={p_chi:.4f}  '
          f'({"SIGNIFICANT" if p_chi < 0.05 else "not significant"})')

# ROC-AUC from well-point probabilities
roc_sub = df_valid.dropna(subset=['pred_prob_high'])
if len(roc_sub) >= 10 and roc_sub['has_water'].nunique() == 2:
    auc_val = roc_auc_score(roc_sub['has_water'], roc_sub['pred_prob_high'])
    fpr_arr, tpr_arr, thresholds = roc_curve(roc_sub['has_water'],
                                             roc_sub['pred_prob_high'])
    stat_results['roc_auc_well_points'] = float(auc_val)
    print(f'   ROC-AUC (well pts)   : {auc_val:.3f}')
else:
    fpr_arr = tpr_arr = None
    auc_val = None
    print('   ROC-AUC              : skipped (insufficient class balance)')

# -- Per-district statistics ---------------------------------------------------
print(f'\n   --- Per-District Summary ---')
district_stats = []
for dist in sorted(df_valid['district'].unique()):
    sub      = df_valid[df_valid['district'] == dist]
    prod_sub = sub[sub['has_water'] == 1]
    dry_sub  = sub[sub['has_water'] == 0]
    sr = (prod_sub['pred_class_int'] >= 1).sum() / len(prod_sub) * 100 if len(prod_sub) else None
    sp = (dry_sub['pred_class_int'] == 0).sum() / len(dry_sub) * 100 if len(dry_sub) else None
    district_stats.append({
        'district': dist,
        'n_productive': len(prod_sub),
        'n_dry': len(dry_sub),
        'sensitivity_pct': round(sr, 1) if sr is not None else None,
        'specificity_pct': round(sp, 1) if sp is not None else None,
    })
    label = f'sens={sr:.0f}%' if sr is not None else 'no productive'
    if sp is not None:
        label += f', spec={sp:.0f}%'
    print(f'   {dist:15s}: {label}  (n_prod={len(prod_sub)}, n_dry={len(dry_sub)})')

district_df = pd.DataFrame(district_stats)

# -- Per-hydro-zone statistics -------------------------------------------------
print(f'\n   --- Per Hydro Zone ---')
zone_stats = {}
for zone in df_valid['hydro_zone'].unique():
    sub = df_valid[df_valid['hydro_zone'] == zone]
    prod_sub = sub[sub['has_water'] == 1]
    dry_sub  = sub[sub['has_water'] == 0]
    sr = (prod_sub['pred_class_int'] >= 1).sum() / len(prod_sub) * 100 if len(prod_sub) else None
    sp = (dry_sub['pred_class_int'] == 0).sum() / len(dry_sub) * 100 if len(dry_sub) else None
    zone_stats[zone] = {'n_productive': len(prod_sub), 'n_dry': len(dry_sub),
                        'sensitivity_pct': round(sr, 1) if sr else None,
                        'specificity_pct': round(sp, 1) if sp else None}
    lbl = ''
    if sr is not None: lbl += f'sens={sr:.0f}%  '
    if sp is not None: lbl += f'spec={sp:.0f}%'
    print(f'   {zone:12s}: {lbl}  (n={len(sub)})')

# -- Save classified CSV -------------------------------------------------------
out_csv = OUT_DATA / 'well_points_classified.csv'
df_valid.to_csv(out_csv, index=False)
print(f'\n   Saved classified wells -> {out_csv}')

# ==============================================================================
# FIGURES
# ==============================================================================
print('\n[6] Generating figures ...')

# -- Figure 1: Well overlay on prediction map ----------------------------------
fig, ax = plt.subplots(figsize=(14, 10))

cmap  = mcolors.ListedColormap([CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2]])
bds   = [-0.5, 0.5, 1.5, 2.5]
norm  = mcolors.BoundaryNorm(bds, cmap.N)
rshow(pred_data, ax=ax, transform=pred_transform, cmap=cmap, norm=norm, alpha=0.80)

markers = {'alluvial': 'o', 'mixed': 's', 'hard_rock': '^'}
for (has_water, zone), grp in df_valid.groupby(['has_water', 'hydro_zone']):
    m = markers.get(zone, 'o')
    c = WELL_COLORS[has_water]
    label = f'{"Productive" if has_water else "Dry"} ({zone})'
    ax.scatter(grp['lon'], grp['lat'], c=c, s=90, marker=m,
               edgecolors='black', linewidths=0.7, zorder=5, label=label, alpha=0.9)

zone_patches = [mpatches.Patch(color=CLASS_COLORS[i],
                label=f'{CLASS_NAMES[i]} Potential') for i in [2, 1, 0]]
ax.legend(handles=zone_patches + ax.get_legend_handles_labels()[0],
          loc='lower left', fontsize=8, framealpha=0.9, ncol=2)

title_line2 = (f'Sensitivity={sensitivity:.1f}%  |  '
               f'Specificity={specificity:.1f}%  |  '
               f'Balanced Accuracy={balanced_acc:.1f}%')
if auc_val is not None:
    title_line2 += f'  |  ROC-AUC={auc_val:.3f}'

ax.set_title(f'Groundwater Potential Map -- Real-World Validation\n'
             f'Pothohar Plateau, Punjab, Pakistan   (n={len(df_valid)} wells)\n'
             f'{title_line2}',
             fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('Longitude (E)', fontsize=10)
ax.set_ylabel('Latitude (N)', fontsize=10)

stats_text = (f'Sensitivity (TPR) : {sensitivity:.1f}%\n'
              f'Specificity (TNR) : {specificity:.1f}%\n'
              f'Balanced Accuracy : {balanced_acc:.1f}%\n'
              f'High Capture Rate : {high_rate:.1f}%')
if auc_val is not None:
    stats_text += f'\nROC-AUC          : {auc_val:.3f}'
ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
        fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.88))

plt.tight_layout()
fig.savefig(OUT_FIG / 'well_overlay_map.png', dpi=150, bbox_inches='tight')
plt.close()
print('   Saved: well_overlay_map.png')

# -- Figure 2: Statistics panel ------------------------------------------------
fig = plt.figure(figsize=(16, 5))
gs  = GridSpec(1, 4, figure=fig, wspace=0.40)

# Panel A: Productive wells by predicted zone (bar)
ax1 = fig.add_subplot(gs[0])
zone_order  = ['High', 'Medium', 'Low']
zone_counts = [productive['pred_label'].eq(z).sum() for z in zone_order]
colors_bar  = [CLASS_COLORS[2], CLASS_COLORS[1], CLASS_COLORS[0]]
bars = ax1.bar(zone_order, zone_counts, color=colors_bar,
               edgecolor='black', linewidth=0.8)
for bar, count in zip(bars, zone_counts):
    pct = count / len(productive) * 100 if len(productive) else 0
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{count}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=9)
ax1.set_title('Productive Wells\nby Predicted Zone', fontsize=11, fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_ylim(0, max(zone_counts) * 1.30 if zone_counts else 1)

# Panel B: Dry wells by predicted zone
ax2 = fig.add_subplot(gs[1])
if len(dry):
    dry_counts = [dry['pred_label'].eq(z).sum() for z in zone_order]
    bars2 = ax2.bar(zone_order, dry_counts, color=colors_bar,
                    edgecolor='black', linewidth=0.8)
    for bar, count in zip(bars2, dry_counts):
        pct = count / len(dry) * 100 if len(dry) else 0
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{count}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=9)
    ax2.set_title('Dry Reference Points\nby Predicted Zone', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.set_ylim(0, max(dry_counts) * 1.30 if dry_counts else 1)
else:
    ax2.text(0.5, 0.5, 'No dry\nwells', ha='center', va='center',
             transform=ax2.transAxes, fontsize=12, color='gray')
    ax2.set_title('Dry Reference Points\nby Predicted Zone', fontsize=11, fontweight='bold')

# Panel C: Per-district sensitivity
ax3 = fig.add_subplot(gs[2])
plot_df = district_df[district_df['sensitivity_pct'].notna()]
if not plot_df.empty:
    bars3 = ax3.barh(plot_df['district'], plot_df['sensitivity_pct'],
                     color='#4393c3', edgecolor='black', linewidth=0.8)
    for bar, row in zip(bars3, plot_df.itertuples()):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{row.sensitivity_pct:.0f}% (n={row.n_productive})',
                 va='center', ha='left', fontsize=8)
    ax3.axvline(x=60, color='green', linestyle='--', linewidth=1.2,
                label='60% benchmark')
    ax3.set_xlim(0, 120)
    ax3.set_title('Sensitivity by District\n(productive wells -> Med+High)',
                  fontsize=10, fontweight='bold')
    ax3.set_xlabel('Sensitivity (%)')
    ax3.legend(fontsize=8)

# Panel D: Sensitivity vs Specificity scatter
ax4 = fig.add_subplot(gs[3])
s_vals = [d['sensitivity_pct'] for d in district_stats if d['sensitivity_pct'] is not None]
sp_vals = [d['specificity_pct'] for d in district_stats if d['specificity_pct'] is not None]
d_labels = [d['district'] for d in district_stats if d['sensitivity_pct'] is not None]
if s_vals:
    ax4.scatter(s_vals, [0]*len(s_vals), s=80, zorder=5, label='Sensitivity',
                c='#4393c3', edgecolors='black')
    ax4.axvline(x=sensitivity, color='#4393c3', linewidth=2,
                label=f'Overall sens={sensitivity:.0f}%')
    ax4.axvline(x=specificity, color='#d73027', linewidth=2, linestyle='--',
                label=f'Overall spec={specificity:.0f}%')
    ax4.axvline(x=balanced_acc, color='purple', linewidth=2, linestyle=':',
                label=f'Balanced={balanced_acc:.0f}%')
    ax4.set_xlim(0, 105)
    ax4.set_yticks([])
    ax4.set_xlabel('Score (%)', fontsize=10)
    ax4.set_title('Sensitivity / Specificity\nOverview', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=7, loc='center left')

plt.suptitle('Real-World Validation Statistics -- Pothohar Plateau GW Potential Map',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(OUT_FIG / 'validation_statistics.png', dpi=150, bbox_inches='tight')
plt.close()
print('   Saved: validation_statistics.png')

# -- Figure 3: Probability distribution at well locations ----------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax_hist = axes[0]
prod_probs_plot = productive['pred_prob_high'].dropna()
dry_probs_plot  = dry['pred_prob_high'].dropna() if len(dry) else pd.Series([], dtype=float)

bins = np.linspace(0, 1, 22)
ax_hist.hist(prod_probs_plot, bins=bins, alpha=0.65, color='#1a9850',
             edgecolor='black', linewidth=0.5,
             label=f'Productive wells (n={len(prod_probs_plot)}, '
                   f'mean={prod_probs_plot.mean():.2f})')
if len(dry_probs_plot):
    ax_hist.hist(dry_probs_plot, bins=bins, alpha=0.65, color='#d73027',
                 edgecolor='black', linewidth=0.5,
                 label=f'Dry wells (n={len(dry_probs_plot)}, '
                       f'mean={dry_probs_plot.mean():.2f})')
ax_hist.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5,
                label='P=0.5 threshold')
ax_hist.set_xlabel('P(High GW Potential)', fontsize=11)
ax_hist.set_ylabel('Number of Wells', fontsize=11)

stat_anno = ''
if 'mann_whitney_p' in stat_results:
    p = stat_results['mann_whitney_p']
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    stat_anno += f'Mann-Whitney: p={p:.4f} {sig}\n'
if 'pointbiserial_r' in stat_results:
    stat_anno += f'Point-biserial r={stat_results["pointbiserial_r"]:.3f}\n'
if stat_anno:
    ax_hist.text(0.02, 0.97, stat_anno.strip(), transform=ax_hist.transAxes,
                 fontsize=9, va='top', ha='left',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax_hist.set_title('P(High GW) at Well Locations\nProductve > Dry = model captures real signal',
                  fontsize=10, fontweight='bold')
ax_hist.legend(fontsize=9)

# ROC curve from well points
ax_roc = axes[1]
if fpr_arr is not None and tpr_arr is not None:
    ax_roc.plot(fpr_arr, tpr_arr, color='#1f78b4', linewidth=2.5,
                label=f'ROC (AUC={auc_val:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.2,
                label='Random (AUC=0.5)')
    ax_roc.fill_between(fpr_arr, tpr_arr, alpha=0.12, color='#1f78b4')
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
    ax_roc.set_title('ROC Curve from Well-Point Probabilities\n'
                     '(productive=1 vs dry=0 using P(High) as score)',
                     fontsize=10, fontweight='bold')
    ax_roc.legend(fontsize=10)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.grid(True, alpha=0.3)
else:
    ax_roc.text(0.5, 0.5, 'ROC not computed\n(need both classes)',
                ha='center', va='center', transform=ax_roc.transAxes,
                fontsize=12, color='gray')
    ax_roc.set_title('ROC Curve', fontsize=10)

plt.tight_layout()
fig.savefig(OUT_FIG / 'probability_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('   Saved: probability_distribution.png')

# -- Save JSON report ----------------------------------------------------------
report = {
    'validation_summary': {
        'n_wells_total'            : len(df_valid),
        'n_productive'             : int(len(productive)),
        'n_dry'                    : int(len(dry)),
        'sensitivity_pct'          : round(sensitivity, 2),
        'specificity_pct'          : round(specificity, 2),
        'balanced_accuracy_pct'    : round(balanced_acc, 2),
        'high_zone_capture_pct'    : round(high_rate, 2),
        'false_positive_rate_pct'  : round(fpr_high, 2),
    },
    'statistical_tests'     : stat_results,
    'productive_by_zone'    : {z: int(productive['pred_label'].eq(z).sum())
                                for z in ['High', 'Medium', 'Low']},
    'dry_by_zone'           : {z: int(dry['pred_label'].eq(z).sum())
                                for z in ['High', 'Medium', 'Low']} if len(dry) else {},
    'per_district'          : district_df.to_dict(orient='records'),
    'per_hydro_zone'        : zone_stats,
    'data_sources'          : df_valid['source'].value_counts().to_dict(),
    'model'                 : 'XGBoost (best_model.joblib)',
    'prediction_map'        : str(PRED_TIF),
    'well_data'             : str(WELLS_CSV),
}

report_path = OUT_FIG / 'validation_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print('   Saved: validation_report.json')

# -- Final summary -------------------------------------------------------------
print('\n' + '=' * 70)
print('  VALIDATION COMPLETE')
print('=' * 70)
print(f'\n  Dataset : {len(df_valid)} wells ({len(productive)} productive, {len(dry)} dry)')
print(f'\n  Sensitivity (TPR) : {sensitivity:.1f}%  -- productive wells in Med+High zone')
print(f'  Specificity (TNR) : {specificity:.1f}%  -- dry wells correctly in Low zone')
print(f'  Balanced Accuracy : {balanced_acc:.1f}%')
print(f'  High Capture Rate : {high_rate:.1f}%')
if auc_val is not None:
    print(f'  ROC-AUC           : {auc_val:.3f}')
if 'mann_whitney_p' in stat_results:
    p = stat_results['mann_whitney_p']
    print(f'  Mann-Whitney p    : {p:.4f}  '
          f'({"SIGNIFICANT" if p < 0.05 else "not significant"})')
print(f'\n  Figures -> outputs/figures/validation/')
print(f'  Report  -> outputs/figures/validation/validation_report.json')
print('=' * 70)
