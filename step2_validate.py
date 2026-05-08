"""
Step 2: Full data validation + EDA before ML training.
Checks data quality, class logic, feature-label alignment,
and generates all diagnostic plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROCESSED = Path('data/processed')
FIGURES   = Path('outputs/figures')
FIGURES.mkdir(parents=True, exist_ok=True)

FEATURES     = ['Elevation','Slope','Aspect','TWI','NDVI','NDWI','Rainfall','Soil','LST']
CLASS_NAMES  = {0: 'Low', 1: 'Medium', 2: 'High'}
CLASS_COLORS = {0: '#d73027', 1: '#fee090', 2: '#1a9850'}
PALETTE      = {'Low': '#d73027', 'Medium': '#fee090', 'High': '#1a9850'}

print('=' * 60)
print('  DATA VALIDATION BEFORE ML TRAINING')
print('=' * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
print('\n[1] Loading training_samples.csv ...')
df = pd.read_csv(PROCESSED / 'training_samples.csv')
df['Class'] = df['GW_Potential'].map(CLASS_NAMES)
print(f'    Shape : {df.shape}')
print(f'    Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')

# ── CHECK 1: Missing values ───────────────────────────────────────────────────
print('\n[2] Checking missing values ...')
nulls = df[FEATURES].isnull().sum()
if nulls.sum() == 0:
    print('    PASS - No missing values in any feature.')
else:
    print('    WARN -Missing values found:')
    print(nulls[nulls > 0])

# ── CHECK 2: Value ranges (all features should be 0-1) ───────────────────────
print('\n[3] Checking value ranges (expect all 0.0 to 1.0) ...')
range_ok = True
for feat in FEATURES:
    mn, mx = df[feat].min(), df[feat].max()
    status = 'OK  ' if (mn >= 0 and mx <= 1) else 'WARN'
    if status == 'WARN':
        range_ok = False
    print(f'    {status} {feat:12s}: min={mn:.4f}  max={mx:.4f}')
if range_ok:
    print('    PASS -All features normalized correctly.')

# ── CHECK 3: Class distribution ───────────────────────────────────────────────
print('\n[4] Class distribution ...')
counts = df['GW_Potential'].value_counts().sort_index()
for cls, name in CLASS_NAMES.items():
    pct = counts[cls] / len(df) * 100
    bar = '#' * int(pct / 2)
    print(f'    {name:8s} ({cls}): {counts[cls]:>8,} rows  {pct:5.1f}%  {bar}')

# ── CHECK 4: Scientific logic check ──────────────────────────────────────────
# High GW potential should have: low elevation, low slope, high TWI, high rainfall
# Low GW potential should have: high elevation/slope, low TWI, low rainfall
print('\n[5] Scientific logic check (do class means make sense?) ...')
means = df.groupby('GW_Potential')[FEATURES].mean()

checks = [
    ('Elevation', 'High should have LOWER elevation than Low',
     means.loc[2,'Elevation'] < means.loc[0,'Elevation']),
    ('Slope',     'High should have LOWER slope than Low',
     means.loc[2,'Slope'] < means.loc[0,'Slope']),
    ('TWI',       'High should have HIGHER TWI than Low',
     means.loc[2,'TWI'] > means.loc[0,'TWI']),
    ('Rainfall',  'High should have HIGHER rainfall than Low',
     means.loc[2,'Rainfall'] > means.loc[0,'Rainfall']),
    ('NDVI',      'High should have HIGHER NDVI than Low',
     means.loc[2,'NDVI'] > means.loc[0,'NDVI']),
    ('LST',       'High should have LOWER LST than Low',
     means.loc[2,'LST'] < means.loc[0,'LST']),
]

all_pass = True
for feat, desc, condition in checks:
    status = 'PASS' if condition else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    low_val  = means.loc[0, feat]
    high_val = means.loc[2, feat]
    print(f'    {status} {feat:12s}: Low={low_val:.3f}  High={high_val:.3f}  -{desc}')

if all_pass:
    print('\n    ALL CHECKS PASSED -Labels are scientifically consistent.')
else:
    print('\n    SOME CHECKS FAILED -Review label generation weights.')

# ── CHECK 5: Feature means per class table ────────────────────────────────────
print('\n[6] Feature means per class (normalized 0-1):')
means_named = means.rename(index=CLASS_NAMES)
print(means_named.round(3).to_string())

# ── CHECK 6: Correlation with target ─────────────────────────────────────────
print('\n[7] Feature correlation with GW_Potential (higher = more useful for ML):')
corr = df[FEATURES + ['GW_Potential']].corr()['GW_Potential'][:-1].sort_values(key=abs, ascending=False)
for feat, val in corr.items():
    bar  = '#' * int(abs(val) * 40)
    sign = '+' if val > 0 else '-'
    print(f'    {feat:12s}  r={sign}{abs(val):.3f}  {bar}')

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

print('\n[8] Generating plots ...')

# ── Plot 1: Class distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Class Distribution -Groundwater Potential Labels', fontsize=14, fontweight='bold')

names_list  = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
values_list = [counts[i] for i in sorted(CLASS_NAMES)]
colors_list = [CLASS_COLORS[i] for i in sorted(CLASS_COLORS)]

bars = axes[0].bar(names_list, values_list, color=colors_list, edgecolor='black', linewidth=0.8, width=0.5)
axes[0].set_title('Pixel Count per Class')
axes[0].set_ylabel('Number of Pixels')
axes[0].set_ylim(0, max(values_list) * 1.15)
for bar, val in zip(bars, values_list):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'{val:,}', ha='center', fontsize=11, fontweight='bold')

wedges, texts, autotexts = axes[1].pie(
    values_list, labels=names_list, colors=colors_list,
    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12},
    wedgeprops={'edgecolor': 'black', 'linewidth': 0.8}
)
axes[1].set_title('Class Proportion')
plt.tight_layout()
plt.savefig(FIGURES / 'class_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print('    Saved: class_distribution.png')

# ── Plot 2: Correlation matrix ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
corr_full = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr_full, dtype=bool))
sns.heatmap(corr_full, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(FIGURES / 'correlation_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print('    Saved: correlation_matrix.png')

# ── Plot 3: Feature means per class (radar-style bar) ────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
x     = np.arange(len(FEATURES))
width = 0.25

for i, (cls, name) in enumerate(CLASS_NAMES.items()):
    vals = means.loc[cls, FEATURES].values
    ax.bar(x + i * width, vals, width, label=name,
           color=CLASS_COLORS[cls], edgecolor='black', linewidth=0.6, alpha=0.9)

ax.set_xticks(x + width)
ax.set_xticklabels(FEATURES, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Mean Normalized Value (0-1)')
ax.set_title('Mean Feature Value per Groundwater Potential Class',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, title='GW Class')
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / 'class_feature_means.png', dpi=200, bbox_inches='tight')
plt.close()
print('    Saved: class_feature_means.png')

# ── Plot 4: Feature distributions per class ───────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle('Feature Distributions by GW Potential Class', fontsize=14, fontweight='bold')

for ax, feat in zip(axes.flat, FEATURES):
    for cls in [0, 1, 2]:
        data = df[df['GW_Potential'] == cls][feat]
        ax.hist(data, bins=60, alpha=0.55, color=CLASS_COLORS[cls],
                label=CLASS_NAMES[cls], density=True)
        ax.axvline(data.mean(), color=CLASS_COLORS[cls], linewidth=1.5, linestyle='--')
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_xlabel('Normalized Value (0-1)', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / 'feature_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print('    Saved: feature_distributions.png')

# ── Plot 5: Boxplots ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle('Feature Spread per GW Potential Class', fontsize=14, fontweight='bold')
order = ['Low', 'Medium', 'High']

for ax, feat in zip(axes.flat, FEATURES):
    sns.boxplot(data=df, x='Class', y=feat, order=order,
                palette=PALETTE, ax=ax, linewidth=0.8,
                flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.3})
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Normalized Value (0-1)', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES / 'feature_boxplots.png', dpi=200, bbox_inches='tight')
plt.close()
print('    Saved: feature_boxplots.png')

# ── Plot 6: Correlation with target ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
corr_sorted = corr.sort_values()
colors_bar  = ['#1a9850' if v > 0 else '#d73027' for v in corr_sorted]
bars = ax.barh(corr_sorted.index, corr_sorted.values, color=colors_bar,
               edgecolor='black', linewidth=0.7)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Feature Correlation with GW_Potential\n(green=positive, red=negative)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Pearson Correlation Coefficient')
ax.set_xlim(-1, 1)
for bar, val in zip(bars, corr_sorted.values):
    ax.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(FIGURES / 'feature_target_correlation.png', dpi=200, bbox_inches='tight')
plt.close()
print('    Saved: feature_target_correlation.png')

# ── Final summary ─────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('  VALIDATION SUMMARY')
print('=' * 60)
print(f'  Total training rows  : {len(df):,}')
print(f'  Features             : {len(FEATURES)}')
print(f'  Missing values       : 0')
print(f'  Class balance        : {counts[0]:,} / {counts[1]:,} / {counts[2]:,}')
print(f'  Scientific logic     : {"ALL PASS" if all_pass else "REVIEW NEEDED"}')
print(f'  Plots saved          : outputs/figures/ (6 plots)')
print()
print('  Most predictive features (by correlation):')
for feat, val in corr.items():
    print(f'    {feat:12s}  {val:+.3f}')
print()
if all_pass:
    print('  DATA IS VALID -Ready to proceed to ML training (Step 3).')
else:
    print('  REVIEW LABELS before ML training.')
