"""
Step 2: Exploratory Data Analysis on training_samples.csv
Outputs:
  - outputs/figures/class_distribution.png
  - outputs/figures/correlation_matrix.png
  - outputs/figures/feature_distributions.png
  - outputs/figures/feature_boxplots.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

PROCESSED = Path('data/processed')
FIGURES   = Path('outputs/figures')
FIGURES.mkdir(parents=True, exist_ok=True)

FEATURES = ['Elevation','Slope','Aspect','TWI','NDVI','NDWI','Rainfall','Soil','LST']
CLASS_NAMES  = {0: 'Low', 1: 'Medium', 2: 'High'}
CLASS_COLORS = {0: '#d73027', 1: '#fee090', 2: '#1a9850'}

print('=' * 55)
print('  Step 2: Exploratory Data Analysis')
print('=' * 55)

print('\nLoading training_samples.csv ...')
df = pd.read_csv(PROCESSED / 'training_samples.csv')
print(f'Loaded: {len(df):,} rows')

# ── Plot 1: Class Distribution ────────────────────────────────────────────────
print('\nPlot 1: Class distribution ...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Groundwater Potential — Class Distribution', fontsize=14, fontweight='bold')

counts = df['GW_Potential'].value_counts().sort_index()
labels = [CLASS_NAMES[i] for i in counts.index]
colors = [CLASS_COLORS[i] for i in counts.index]

axes[0].bar(labels, counts.values, color=colors, edgecolor='black', linewidth=0.8)
axes[0].set_title('Pixel Count per Class')
axes[0].set_ylabel('Number of Pixels')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=10)

axes[1].pie(counts.values, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Class Proportion')

plt.tight_layout()
plt.savefig(FIGURES / 'class_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: outputs/figures/class_distribution.png')

# ── Plot 2: Correlation Matrix ────────────────────────────────────────────────
print('Plot 2: Correlation matrix ...')
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(FIGURES / 'correlation_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: outputs/figures/correlation_matrix.png')

# ── Plot 3: Feature Distributions per Class ───────────────────────────────────
print('Plot 3: Feature distributions per class ...')
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Feature Distributions by Groundwater Potential Class', fontsize=14, fontweight='bold')

for ax, feat in zip(axes.flat, FEATURES):
    for cls in [0, 1, 2]:
        data = df[df['GW_Potential'] == cls][feat]
        ax.hist(data, bins=50, alpha=0.6, color=CLASS_COLORS[cls],
                label=CLASS_NAMES[cls], density=True)
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_xlabel('Normalized Value (0-1)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / 'feature_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: outputs/figures/feature_distributions.png')

# ── Plot 4: Feature Boxplots by Class ────────────────────────────────────────
print('Plot 4: Feature boxplots ...')
df_plot = df.copy()
df_plot['Class'] = df_plot['GW_Potential'].map(CLASS_NAMES)

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Feature Spread by Groundwater Potential Class', fontsize=14, fontweight='bold')

palette = {v: CLASS_COLORS[k] for k, v in CLASS_NAMES.items()}
order   = ['Low', 'Medium', 'High']

for ax, feat in zip(axes.flat, FEATURES):
    sns.boxplot(data=df_plot, x='Class', y=feat, order=order,
                palette=palette, ax=ax, linewidth=0.8)
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Normalized Value')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES / 'feature_boxplots.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: outputs/figures/feature_boxplots.png')

# ── Print summary stats ───────────────────────────────────────────────────────
print('\n── Summary Statistics per Class ──')
print(df.groupby('GW_Potential')[FEATURES].mean().round(3).rename(index=CLASS_NAMES).to_string())

print('\n── Feature Correlation with GW_Potential ──')
corr_target = df[FEATURES + ['GW_Potential']].corr()['GW_Potential'][:-1].sort_values(ascending=False)
for feat, val in corr_target.items():
    bar = '█' * int(abs(val) * 30)
    sign = '+' if val > 0 else '-'
    print(f'  {feat:12s} {sign}{abs(val):.3f}  {bar}')

print('\n✓ Step 2 complete — check outputs/figures/ for all plots.')
