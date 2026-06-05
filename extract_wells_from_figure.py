"""
Auto-digitizer for Naz et al. 2023 Figure 1
Detects red sampling points on the Punjab groundwater map
and converts pixel positions to GPS coordinates using the
figure's built-in coordinate grid.

Usage:
  1. Save Figure 1 from Naz et al. 2023 as:
       data/validation/naz2023_figure1.png
  2. Run: python extract_wells_from_figure.py
  3. Step A: click 4 grid-line intersections to calibrate
  4. Script auto-detects all red dots and exports CSV

Output:
  data/validation/naz2023_wells_pothohar.csv
  data/validation/naz2023_wells_all.csv
  outputs/figures/validation/digitized_wells_check.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import cv2
import pandas as pd
import json
import sys

IMG_PATH  = Path('data/validation/naz2023_figure1.png')
OUT_DIR   = Path('data/validation')
FIG_DIR   = Path('outputs/figures/validation')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Pothohar bounding box (your study area)
POTHOHAR_BOUNDS = {
    'lon_min': 71.107, 'lon_max': 73.819,
    'lat_min': 32.162, 'lat_max': 34.017,
}

print('=' * 60)
print('  FIGURE DIGITIZER — Naz et al. 2023 Figure 1')
print('=' * 60)

if not IMG_PATH.exists():
    print(f'\nERROR: Image not found at {IMG_PATH}')
    print('\nSteps to fix:')
    print('  1. Open Naz et al. 2023 PDF:')
    print('     https://www.mdpi.com/2073-4441/16/1/63/pdf')
    print('  2. Screenshot Figure 1 (the map with red dots)')
    print(f'  3. Save as: {IMG_PATH.absolute()}')
    print('  4. Re-run this script')
    sys.exit(1)

# ── Load image ────────────────────────────────────────────────────────────────
img_bgr = cv2.imread(str(IMG_PATH))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f'\nImage loaded: {W} x {H} pixels')

# ── STEP A: Calibration — click 4 known geo-reference points ──────────────────
print('\n[STEP A] CALIBRATION')
print('  The figure has a coordinate grid with known lat/lon lines.')
print('  You will click 4 points where grid lines cross.')
print('  Use the MAIN MAP (right panel) only — ignore the insets.')
print()
print('  Suggested clicks (pick visible grid intersections):')
print('    Point 1: where 71E grid line meets 34N grid line')
print('    Point 2: where 75E grid line meets 34N grid line')
print('    Point 3: where 71E grid line meets 30N grid line')
print('    Point 4: where 75E grid line meets 30N grid line')
print()
print('  Click the 4 points on the image window that will open.')
print('  Press ENTER after clicking all 4.')

GEO_REFS = [
    (71.0, 34.0),
    (75.0, 34.0),
    (71.0, 30.0),
    (75.0, 30.0),
]

clicked_pixels = []

def on_click(event):
    if event.inaxes and event.button == 1:
        if len(clicked_pixels) < 4:
            clicked_pixels.append((event.xdata, event.ydata))
            idx = len(clicked_pixels)
            geo = GEO_REFS[idx - 1]
            ax.plot(event.xdata, event.ydata, 'y+', markersize=18,
                    markeredgewidth=2.5)
            ax.text(event.xdata + 8, event.ydata - 8,
                    f'P{idx}: {geo[0]}E, {geo[1]}N',
                    color='yellow', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.6, pad=2))
            print(f'  Clicked P{idx}: pixel ({event.xdata:.0f}, {event.ydata:.0f})'
                  f'  -> geo ({geo[0]}E, {geo[1]}N)')
            fig_cal.canvas.draw()
            if len(clicked_pixels) == 4:
                print('\n  4 points captured. Close the window to continue.')

fig_cal, ax = plt.subplots(figsize=(14, 9))
ax.imshow(img_rgb)
ax.set_title('CALIBRATION: Click the 4 corner grid intersections\n'
             '(71E/34N), (75E/34N), (71E/30N), (75E/30N) — then close window',
             fontsize=10, color='red', fontweight='bold')
fig_cal.canvas.mpl_connect('button_press_event', on_click)
plt.tight_layout()
plt.show()

if len(clicked_pixels) < 4:
    print('\nERROR: Need exactly 4 calibration points. Re-run and click 4 points.')
    sys.exit(1)

# ── Build affine transform pixel -> lon/lat ────────────────────────────────────
print('\n[STEP B] Building coordinate transform ...')

px = np.array([[p[0], p[1], 1] for p in clicked_pixels], dtype=float)
lon_ref = np.array([g[0] for g in GEO_REFS], dtype=float)
lat_ref = np.array([g[1] for g in GEO_REFS], dtype=float)

# Least-squares affine fit
coeffs_lon, _, _, _ = np.linalg.lstsq(px, lon_ref, rcond=None)
coeffs_lat, _, _, _ = np.linalg.lstsq(px, lat_ref, rcond=None)

def pixel_to_geo(px_x, px_y):
    v = np.array([px_x, px_y, 1.0])
    return float(coeffs_lon @ v), float(coeffs_lat @ v)

# Verify calibration
print('  Calibration check (should match reference):')
for i, (px_pt, geo_ref) in enumerate(zip(clicked_pixels, GEO_REFS)):
    lon_c, lat_c = pixel_to_geo(*px_pt)
    print(f'    P{i+1}: predicted ({lon_c:.3f}E, {lat_c:.3f}N) | '
          f'actual ({geo_ref[0]}E, {geo_ref[1]}N) | '
          f'error ({abs(lon_c-geo_ref[0]):.3f}, {abs(lat_c-geo_ref[1]):.3f}) deg')

# Save calibration
cal_data = {
    'clicked_pixels': clicked_pixels,
    'geo_refs': GEO_REFS,
    'coeffs_lon': coeffs_lon.tolist(),
    'coeffs_lat': coeffs_lat.tolist(),
}
with open(OUT_DIR / 'calibration.json', 'w') as f:
    json.dump(cal_data, f, indent=2)
print('  Calibration saved -> data/validation/calibration.json')

# ── STEP C: Detect red dots ────────────────────────────────────────────────────
print('\n[STEP C] Detecting red dots ...')

# Convert to HSV for robust red detection
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Red hue range in HSV (red wraps around 0/180)
mask1 = cv2.inRange(img_hsv, (0,  120, 100), (10,  255, 255))
mask2 = cv2.inRange(img_hsv, (165, 120, 100), (180, 255, 255))
red_mask = cv2.bitwise_or(mask1, mask2)

# Morphological cleanup — remove noise, keep dot-sized blobs
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find connected components (each = one dot)
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    red_mask, connectivity=8)

print(f'  Raw blobs detected: {n_labels - 1}')

# Filter by blob size (dots should be small — 5 to 200 pixels area)
dots = []
for i in range(1, n_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    cx, cy = centroids[i]
    if 5 <= area <= 300:
        lon, lat = pixel_to_geo(cx, cy)
        dots.append({'pixel_x': cx, 'pixel_y': cy, 'area_px': area,
                     'lon': lon, 'lat': lat})

print(f'  Valid dots after size filter: {len(dots)}')

df_all = pd.DataFrame(dots)

# ── Filter for Pothohar bounds ────────────────────────────────────────────────
b = POTHOHAR_BOUNDS
df_pothohar = df_all[
    (df_all['lon'] >= b['lon_min']) & (df_all['lon'] <= b['lon_max']) &
    (df_all['lat'] >= b['lat_min']) & (df_all['lat'] <= b['lat_max'])
].copy()

df_pothohar['source']    = 'Naz_2023'
df_pothohar['has_water'] = 1         # all PCRWR monitoring wells are productive
df_pothohar['well_type'] = 'monitoring_well'
df_pothohar['notes']     = 'Auto-digitized from Naz et al. 2023 Fig.1 PCRWR network'

print(f'\n  Total dots in image     : {len(df_all)}')
print(f'  Dots in Pothohar bounds : {len(df_pothohar)}')
print(f'  Lon range (Pothohar)    : {df_pothohar["lon"].min():.3f} - {df_pothohar["lon"].max():.3f}')
print(f'  Lat range (Pothohar)    : {df_pothohar["lat"].min():.3f} - {df_pothohar["lat"].max():.3f}')

# ── Save outputs ──────────────────────────────────────────────────────────────
df_all.to_csv(OUT_DIR / 'naz2023_wells_all.csv', index=False)
df_pothohar[['lon','lat','source','has_water','well_type','notes']].to_csv(
    OUT_DIR / 'naz2023_wells_pothohar.csv', index=False)
print(f'\n  Saved: data/validation/naz2023_wells_all.csv ({len(df_all)} points)')
print(f'  Saved: data/validation/naz2023_wells_pothohar.csv ({len(df_pothohar)} points)')

# ── Verification figure ────────────────────────────────────────────────────────
print('\n[STEP D] Generating verification figure ...')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: all detected dots on original image
ax0 = axes[0]
ax0.imshow(img_rgb)
if len(df_all):
    ax0.scatter(df_all['pixel_x'], df_all['pixel_y'],
                s=12, c='cyan', edgecolors='none', alpha=0.7, label='Detected')
if len(df_pothohar):
    ax0.scatter(df_pothohar['pixel_x'], df_pothohar['pixel_y'],
                s=20, c='yellow', edgecolors='black', linewidths=0.5,
                alpha=0.9, label='In Pothohar')
ax0.set_title(f'Detection check\n{len(df_all)} total | {len(df_pothohar)} in Pothohar',
              fontsize=10)
ax0.legend(fontsize=8, loc='lower right')
ax0.axis('off')

# Right: geo-space scatter of Pothohar wells
ax1 = axes[1]
ax1.scatter(df_all['lon'], df_all['lat'],
            s=8, c='lightgray', alpha=0.4, label='All Punjab points')
ax1.scatter(df_pothohar['lon'], df_pothohar['lat'],
            s=25, c='red', edgecolors='black', linewidths=0.5,
            alpha=0.85, label=f'Pothohar (n={len(df_pothohar)})')

# Draw Pothohar bounding box
rect = patches.Rectangle(
    (b['lon_min'], b['lat_min']),
    b['lon_max'] - b['lon_min'],
    b['lat_max'] - b['lat_min'],
    linewidth=2, edgecolor='blue', facecolor='none', linestyle='--',
    label='Pothohar bounds')
ax1.add_patch(rect)

ax1.set_xlabel('Longitude (E)', fontsize=10)
ax1.set_ylabel('Latitude (N)', fontsize=10)
ax1.set_title('Extracted well coordinates\n(verify alignment looks correct)',
              fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

plt.suptitle('Digitization Verification — Naz et al. 2023 Figure 1',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(FIG_DIR / 'digitized_wells_check.png', dpi=150, bbox_inches='tight')
plt.show()
print('  Saved: outputs/figures/validation/digitized_wells_check.png')

# ── Final instructions ─────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('  DIGITIZATION COMPLETE')
print('=' * 60)
print(f'\n  Extracted {len(df_pothohar)} productive monitoring wells')
print(f'  from Naz et al. 2023 within your Pothohar study area.')
print()
print('  NEXT STEP: Feed into validation script')
print('  Copy naz2023_wells_pothohar.csv into well_points_template.csv')
print('  (or add a district column and run step5_realworld_validation.py)')
print()
print('  ALSO: If dots look misaligned in the verification figure,')
print('  re-run and click more precise calibration points.')
print('=' * 60)
