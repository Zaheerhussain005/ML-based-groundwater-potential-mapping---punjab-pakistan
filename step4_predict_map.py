"""
Step 4: Generate the full groundwater potential prediction map.

Loads the best trained model and predicts on all 37M valid pixels of the
Pothohar Plateau via windowed reads (memory-safe, no full-raster load).

Outputs:
  outputs/maps/
    gw_prediction_xgboost.tif       int8 single-band class map (-1 = nodata)
    gw_probability_high.tif         float32 probability of class High (0-1)
    gw_prediction_xgboost.png       colored Low/Med/High visualization
    gw_probability_high.png         continuous probability map
    gw_comparison_overlay_vs_ml.png side-by-side weighted-overlay vs ML prediction
    district_class_stats.csv        per-district class %  (if districts shp present)
  outputs/figures/
    prediction_summary.png          class proportions + agreement summary

Reads:
  data/processed/features_normalized.tif
  data/processed/all_labels.npy            (for ML vs weighted-overlay comparison)
  outputs/models/best_model.joblib
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from loguru import logger

PROCESSED      = Path('data/processed')
MODELS_DIR     = Path('outputs/models')
MAPS_DIR       = Path('outputs/maps')
FIG_DIR        = Path('outputs/figures')
MAPS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_TIF     = PROCESSED / 'features_normalized.tif'
LABELS_NPY    = PROCESSED / 'all_labels.npy'
MODEL_PATH    = MODELS_DIR / 'best_model.joblib'
PRED_TIF      = MAPS_DIR / 'gw_prediction_xgboost.tif'
PROBA_TIF     = MAPS_DIR / 'gw_probability_high.tif'

CLASS_NAMES   = ['Low', 'Medium', 'High']
CLASS_COLORS  = ['#d73027', '#fee090', '#1a9850']
WINDOW_SIZE   = 2048             # 2048 x 2048 = 4.2 M pixels per tile
DOWNSAMPLE_PNG = 5               # render PNGs at 1/5 resolution


# ════════════════════════════════════════════════════════════════════════════
#  1. Windowed prediction over the full raster
# ════════════════════════════════════════════════════════════════════════════
def predict_windowed(model):
    logger.info(f'Reading {INPUT_TIF.name} in {WINDOW_SIZE}x{WINDOW_SIZE} tiles ...')
    t0 = time.time()
    n_total = n_valid = 0

    with rasterio.open(INPUT_TIF) as src:
        meta = src.meta.copy()
        h, w = src.height, src.width
        n_tiles_h = (h + WINDOW_SIZE - 1) // WINDOW_SIZE
        n_tiles_w = (w + WINDOW_SIZE - 1) // WINDOW_SIZE
        n_tiles = n_tiles_h * n_tiles_w
        logger.info(f'  raster: {w} x {h}  |  tiles: {n_tiles_w} x {n_tiles_h} = {n_tiles}')

        meta_pred = meta.copy()
        meta_pred.update(count=1, dtype='int8', nodata=-1, compress='lzw',
                         tiled=True, blockxsize=512, blockysize=512)
        meta_proba = meta.copy()
        meta_proba.update(count=1, dtype='float32', nodata=np.nan, compress='lzw',
                          tiled=True, blockxsize=512, blockysize=512)

        with rasterio.open(PRED_TIF, 'w', **meta_pred) as dst_pred, \
             rasterio.open(PROBA_TIF, 'w', **meta_proba) as dst_prob:
            tile_idx = 0
            for j in range(0, h, WINDOW_SIZE):
                for i in range(0, w, WINDOW_SIZE):
                    tile_idx += 1
                    win_h = min(WINDOW_SIZE, h - j)
                    win_w = min(WINDOW_SIZE, w - i)
                    win   = Window(i, j, win_w, win_h)

                    tile = src.read(window=win)              # (9, win_h, win_w)
                    flat = tile.reshape(tile.shape[0], -1).T  # (N, 9)
                    valid = ~np.isnan(flat).any(axis=1)
                    n_total += flat.shape[0]
                    n_valid += int(valid.sum())

                    preds = np.full(flat.shape[0], -1, dtype=np.int8)
                    proba = np.full(flat.shape[0], np.nan, dtype=np.float32)

                    if valid.any():
                        X = flat[valid].astype(np.float32)
                        preds[valid] = model.predict(X).astype(np.int8)
                        # probability of class index 2 (High) for confidence map
                        p = model.predict_proba(X)
                        proba[valid] = p[:, 2].astype(np.float32)

                    dst_pred.write(preds.reshape(win_h, win_w)[np.newaxis], window=win)
                    dst_prob.write(proba.reshape(win_h, win_w)[np.newaxis], window=win)

                    if tile_idx % 5 == 0 or tile_idx == n_tiles:
                        pct = tile_idx / n_tiles * 100
                        logger.info(f'  tile {tile_idx:>3}/{n_tiles}  ({pct:5.1f}%)')

    dt = time.time() - t0
    logger.success(f'Prediction complete in {dt:.1f}s')
    logger.info(f'  valid pixels predicted: {n_valid:,} / {n_total:,}  '
                f'({n_valid/n_total*100:.1f}%)')
    return n_total, n_valid


# ════════════════════════════════════════════════════════════════════════════
#  2. Render prediction PNGs (downsampled)
# ════════════════════════════════════════════════════════════════════════════
def render_class_png():
    logger.info('Rendering class prediction PNG ...')
    with rasterio.open(PRED_TIF) as src:
        out_h = src.height // DOWNSAMPLE_PNG
        out_w = src.width  // DOWNSAMPLE_PNG
        from rasterio.enums import Resampling
        arr = src.read(1, out_shape=(out_h, out_w),
                       resampling=Resampling.nearest)

    masked = np.ma.masked_where(arr == -1, arr)
    cmap = ListedColormap(CLASS_COLORS); cmap.set_bad('white', 0.0)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    counts = np.bincount(arr[arr >= 0].astype(np.int64), minlength=3)
    total  = counts.sum()

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(masked, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title('Groundwater Potential — Pothohar Plateau\n'
                 'XGBoost ML Prediction (best model)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.axis('off')
    legend = [
        Patch(facecolor=CLASS_COLORS[i], edgecolor='black',
              label=f'{CLASS_NAMES[i]}  ({counts[i]/total*100:.1f}%)')
        for i in range(3)
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=11,
              title='GW Potential', title_fontsize=11, framealpha=0.9)
    sub = (f'Low: {counts[0]:,}  |  Medium: {counts[1]:,}  |  High: {counts[2]:,}'
           f'  |  Total predicted: {total:,} pixels')
    fig.text(0.5, 0.02, sub, ha='center', fontsize=9, style='italic')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = MAPS_DIR / 'gw_prediction_xgboost.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success(f'  {out.name}')
    return counts


def render_probability_png():
    logger.info('Rendering High-class probability PNG ...')
    with rasterio.open(PROBA_TIF) as src:
        out_h = src.height // DOWNSAMPLE_PNG
        out_w = src.width  // DOWNSAMPLE_PNG
        from rasterio.enums import Resampling
        arr = src.read(1, out_shape=(out_h, out_w),
                       resampling=Resampling.average)

    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(arr, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('Probability of High Groundwater Potential — Pothohar Plateau\n'
                 '(XGBoost continuous prediction)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.92)
    cbar.set_label('P(class = High)', fontsize=11)
    plt.tight_layout()
    out = MAPS_DIR / 'gw_probability_high.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success(f'  {out.name}')


# ════════════════════════════════════════════════════════════════════════════
#  3. Side-by-side comparison: weighted overlay labels vs ML prediction
# ════════════════════════════════════════════════════════════════════════════
def render_comparison():
    logger.info('Rendering weighted-overlay vs ML side-by-side comparison ...')
    from rasterio.enums import Resampling

    # Load weighted-overlay labels at full res, downsample with nearest neighbor
    with rasterio.open(INPUT_TIF) as src:
        full_h, full_w = src.height, src.width
        band1 = src.read(1)  # for valid mask
    valid_mask_2d = ~np.isnan(band1)
    overlay_labels = np.load(LABELS_NPY)
    overlay_grid = np.full(band1.shape, -1, dtype=np.int8)
    overlay_grid[valid_mask_2d] = overlay_labels
    del band1

    overlay_ds = overlay_grid[::DOWNSAMPLE_PNG, ::DOWNSAMPLE_PNG]
    ds_h, ds_w = overlay_ds.shape  # use this as the canonical size

    with rasterio.open(PRED_TIF) as src:
        ml_ds = src.read(1, out_shape=(ds_h, ds_w),
                          resampling=Resampling.nearest)

    # Stats — compute agreement on down-sampled views (representative & fast)
    cmp_valid = (overlay_ds >= 0) & (ml_ds >= 0)
    agreement = (overlay_ds[cmp_valid] == ml_ds[cmp_valid]).mean()

    cmap = ListedColormap(CLASS_COLORS); cmap.set_bad('white', 0.0)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, grid, title in zip(
        axes,
        [overlay_ds, ml_ds],
        ['Weighted-Overlay Labels (knowledge-based)',
         'XGBoost ML Prediction (data-driven)'],
    ):
        masked = np.ma.masked_where(grid == -1, grid)
        ax.imshow(masked, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')

    legend = [Patch(facecolor=CLASS_COLORS[i], edgecolor='black',
                    label=CLASS_NAMES[i]) for i in range(3)]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle(f'Weighted Overlay vs ML Prediction — Pothohar Plateau\n'
                 f'Visual agreement (downsampled): {agreement*100:.2f}%',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = MAPS_DIR / 'gw_comparison_overlay_vs_ml.png'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success(f'  {out.name}')

    # ── Full-resolution exact agreement statistics ────────────────────────────
    logger.info('Computing full-resolution agreement statistics ...')
    with rasterio.open(PRED_TIF) as src:
        ml_full = src.read(1)
    valid_pair = (overlay_grid >= 0) & (ml_full >= 0)
    n_pair = int(valid_pair.sum())
    agree  = (overlay_grid[valid_pair] == ml_full[valid_pair]).sum()
    full_agreement = agree / n_pair
    logger.info(f'  Full-res agreement: {agree:,} / {n_pair:,}  '
                f'({full_agreement*100:.4f}%)')

    # Per-class confusion: weighted-overlay (rows) vs ML (cols)
    confusion = np.zeros((3, 3), dtype=np.int64)
    for r in range(3):
        rmask = overlay_grid[valid_pair] == r
        for c in range(3):
            confusion[r, c] = ((overlay_grid[valid_pair] == r) &
                                (ml_full[valid_pair] == c)).sum()
    return overlay_grid, ml_full, full_agreement, confusion


def _pred_full_path():
    """placeholder for compatibility; see comparison fn body."""
    return PRED_TIF


# ════════════════════════════════════════════════════════════════════════════
#  4. Summary chart: pixel counts + per-class agreement
# ════════════════════════════════════════════════════════════════════════════
def render_summary(counts_ml, counts_overlay, full_agreement, confusion):
    logger.info('Rendering prediction summary figure ...')
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(17, 6))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)

    # ── Bar comparison: ML vs Weighted Overlay class counts ──
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(3); width = 0.36
    ax1.bar(x - width/2, counts_overlay, width, color='#888888',
            edgecolor='black', label='Weighted Overlay')
    ax1.bar(x + width/2, counts_ml, width, color=CLASS_COLORS,
            edgecolor='black', label='XGBoost ML')
    ax1.set_xticks(x); ax1.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax1.set_ylabel('Pixel count')
    ax1.set_title('Pixel Count per Class — Both Methods',
                  fontsize=12, fontweight='bold')
    for i, (a, b) in enumerate(zip(counts_overlay, counts_ml)):
        ax1.text(i - width/2, a + max(counts_ml) * 0.01,
                 f'{a/1e6:.2f}M', ha='center', fontsize=8.5)
        ax1.text(i + width/2, b + max(counts_ml) * 0.01,
                 f'{b/1e6:.2f}M', ha='center', fontsize=8.5)
    ax1.legend(fontsize=9); ax1.grid(True, axis='y', alpha=0.3)

    # ── Pie: ML class proportions ──
    ax2 = fig.add_subplot(gs[1])
    total_ml = counts_ml.sum()
    pcts = [c / total_ml * 100 for c in counts_ml]
    wlbls = [f'{n}\n{p:.1f}%' for n, p in zip(CLASS_NAMES, pcts)]
    ax2.pie(counts_ml, labels=wlbls, colors=CLASS_COLORS, startangle=90,
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.7},
            explode=[0.03, 0.03, 0.05], textprops={'fontsize': 11})
    ax2.set_title('XGBoost Predicted Class Proportions',
                  fontsize=12, fontweight='bold')

    # ── Confusion: weighted-overlay (rows) vs ML (cols) ──
    ax3 = fig.add_subplot(gs[2])
    cm_pct = confusion / confusion.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f'{confusion[i,j]/1e6:.2f}M\n({cm_pct[i,j]:.1f}%)'
                        for j in range(3)] for i in range(3)])
    import seaborn as sns
    sns.heatmap(confusion, annot=annot, fmt='', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax3, cbar=False, linewidths=0.5, linecolor='white')
    ax3.set_xlabel('XGBoost ML prediction')
    ax3.set_ylabel('Weighted-overlay label')
    ax3.set_title(f'Method Agreement (full res)\nOverall = {full_agreement*100:.2f}%',
                  fontsize=12, fontweight='bold')

    fig.suptitle('Step 4 — Prediction Map Summary (Pothohar Plateau)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIG_DIR / 'prediction_summary.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.success(f'  {out.name}')


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print('  Step 4: Generate ML Prediction Map')
    print('=' * 70)

    logger.info(f'Loading model: {MODEL_PATH}')
    model = joblib.load(MODEL_PATH)
    logger.info(f'  type: {type(model).__name__}')

    n_total, n_valid = predict_windowed(model)
    counts_ml = render_class_png()
    render_probability_png()
    overlay_grid, ml_full, full_agreement, confusion = render_comparison()

    counts_overlay = np.bincount(
        overlay_grid[overlay_grid >= 0].astype(np.int64), minlength=3)
    render_summary(counts_ml, counts_overlay, full_agreement, confusion)

    # ── Save run summary JSON ────────────────────────────────────────────────
    summary = {
        'model_used'           : str(MODEL_PATH.name),
        'pixels_total'         : int(n_total),
        'pixels_valid'         : int(n_valid),
        'pixels_predicted'     : int(counts_ml.sum()),
        'class_counts_ml'      : {CLASS_NAMES[i]: int(counts_ml[i])      for i in range(3)},
        'class_counts_overlay' : {CLASS_NAMES[i]: int(counts_overlay[i]) for i in range(3)},
        'class_pct_ml'         : {CLASS_NAMES[i]: float(counts_ml[i] / counts_ml.sum() * 100)
                                   for i in range(3)},
        'agreement_full_res'   : float(full_agreement),
        'agreement_confusion'  : confusion.tolist(),
        'window_size_px'       : WINDOW_SIZE,
        'png_downsample_factor': DOWNSAMPLE_PNG,
    }
    with open(MAPS_DIR / 'prediction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print('=' * 70)
    print('  PREDICTION COMPLETE')
    print('=' * 70)
    total_ml = counts_ml.sum()
    for i, name in enumerate(CLASS_NAMES):
        print(f'  {name:<8}: {counts_ml[i]:>12,}  '
              f'({counts_ml[i] / total_ml * 100:5.2f}%)')
    print(f'  Total   : {total_ml:>12,}  pixels predicted')
    print(f'  Method agreement (ML vs weighted overlay): {full_agreement*100:.4f}%')
    print()
    print('  Outputs:')
    print(f'    {PRED_TIF}                  ({PRED_TIF.stat().st_size/1e6:.1f} MB)')
    print(f'    {PROBA_TIF}                 ({PROBA_TIF.stat().st_size/1e6:.1f} MB)')
    print(f'    outputs/maps/gw_prediction_xgboost.png')
    print(f'    outputs/maps/gw_probability_high.png')
    print(f'    outputs/maps/gw_comparison_overlay_vs_ml.png')
    print(f'    outputs/figures/prediction_summary.png')
    print(f'    outputs/maps/prediction_summary.json')


if __name__ == '__main__':
    main()
