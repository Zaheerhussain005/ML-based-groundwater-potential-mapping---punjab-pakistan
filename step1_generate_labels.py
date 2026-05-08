"""
Step 1: Generate groundwater potential labels via weighted overlay.
Maps each pixel to: 0=Low, 1=Medium, 2=High groundwater potential.
Output:
  - data/processed/training_samples.csv  (balanced sample for ML)
  - data/processed/all_labels.npy        (labels for all 37M pixels)
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED = Path('data/processed')

# ── Weights (must sum to 1.0) ────────────────────────────────────────────────
# Mapped from reference paper: Geology/LULC=45%, Drainage=25%, Rain=20%, Slope=10%
WEIGHTS = {
    'Soil':      0.25,   # proxy for geology — most important
    'TWI':       0.20,   # proxy for drainage density
    'Rainfall':  0.20,   # recharge driver
    'NDVI':      0.10,   # vegetation/LULC proxy
    'Slope':     0.10,   # infiltration rate
    'NDWI':      0.05,   # water body proximity
    'Elevation': 0.05,   # terrain context
    'LST':       0.03,   # evapotranspiration proxy
    'Aspect':    0.02,   # solar exposure (minor)
}

# +1 = higher value means better GW potential (direct)
# -1 = lower value means better GW potential (inverse)
DIRECTION = {
    'Soil':      +1,   # higher class (4-9) = coarser = better permeability
    'TWI':       +1,   # higher wetness index = better
    'Rainfall':  +1,   # more rain = more recharge
    'NDVI':      +1,   # denser vegetation = better recharge zones
    'Slope':     -1,   # flatter terrain = more infiltration time
    'NDWI':      +1,   # more water presence = better
    'Elevation': -1,   # lower elevation = alluvial plains = better
    'LST':       -1,   # cooler = less evaporation = better
    'Aspect':    +1,   # neutral (small weight)
}

SAMPLE_PER_CLASS = 150_000   # 150K x 3 classes = 450K training rows

print('=' * 55)
print('  Step 1: Generating Groundwater Potential Labels')
print('=' * 55)
print(f'\nLoading features.csv ...')

df = pd.read_csv(PROCESSED / 'features.csv', dtype='float32')
print(f'Loaded  : {len(df):,} rows x {len(df.columns)} features')

# ── Compute weighted overlay score ───────────────────────────────────────────
print('\nComputing weighted overlay scores ...')
score = np.zeros(len(df), dtype='float32')
for feat, weight in WEIGHTS.items():
    if DIRECTION[feat] == +1:
        score += df[feat].values * weight
    else:
        score += (1.0 - df[feat].values) * weight

print(f'Score stats: min={score.min():.3f}  max={score.max():.3f}  mean={score.mean():.3f}')

# ── Quantile-based classification ────────────────────────────────────────────
# Low=bottom 40%, Medium=middle 40%, High=top 20%  (mirrors paper's distribution)
p40 = np.percentile(score, 40)
p80 = np.percentile(score, 80)
print(f'\nThresholds: Low < {p40:.3f} | Medium {p40:.3f}-{p80:.3f} | High >= {p80:.3f}')

labels = np.where(score >= p80, 2,
         np.where(score >= p40, 1, 0)).astype('int8')

# ── Class distribution ────────────────────────────────────────────────────────
counts = pd.Series(labels).value_counts().sort_index()
total  = len(labels)
print(f'\nClass distribution (all {total:,} pixels):')
print(f'  Low    (0): {counts[0]:>12,}  ({counts[0]/total*100:.1f}%)')
print(f'  Medium (1): {counts[1]:>12,}  ({counts[1]/total*100:.1f}%)')
print(f'  High   (2): {counts[2]:>12,}  ({counts[2]/total*100:.1f}%)')

# ── Save all labels (for prediction map reconstruction later) ─────────────────
labels_path = PROCESSED / 'all_labels.npy'
np.save(labels_path, labels)
print(f'\nSaved all labels : {labels_path}  ({labels.nbytes / 1e6:.0f} MB)')

# ── Create balanced sample for ML training ────────────────────────────────────
print(f'\nSampling {SAMPLE_PER_CLASS:,} pixels per class ...')
df['GW_Potential'] = labels.astype('int8')

sampled = pd.concat([
    df[df['GW_Potential'] == cls].sample(
        n=min(SAMPLE_PER_CLASS, counts[cls]),
        random_state=42
    )
    for cls in [0, 1, 2]
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f'Training set size: {len(sampled):,} rows')
print(f'Class balance: {sampled["GW_Potential"].value_counts().sort_index().to_dict()}')

out_path = PROCESSED / 'training_samples.csv'
sampled.to_csv(out_path, index=False)
print(f'Saved training samples: {out_path}')

print('\n✓ Step 1 complete — ready for EDA and ML training.')
