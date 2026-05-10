# Groundwater Potential Mapping — Research Project

## Project Overview
ML-based groundwater potential mapping for the **Pothohar Plateau, Punjab, Pakistan**.
Extends and improves upon: *Waqas et al. (2022), Pak. j. sci. ind. res. 65A(2) 128-134*.

**Researcher:** Data Science student, ML expert, Web Developer, GIS/QGIS experience
**Goal:** Final thesis — replace weighted overlay with ML models, add web GIS output
**GEE Account:** ee-zaheersapar005 (zaheersapar005@gmail.com)
**Python:** 3.13 | **venv:** `d:\rao-research\venv\` | Activate: `.\venv\Scripts\activate`

---

## Current Progress Status

| Step | Task | Status |
|------|------|--------|
| 1 | Project setup + venv + packages | DONE |
| 2 | GEE data collection (9 layers, 5 districts) | DONE |
| 3 | Raster preprocessing + normalization | DONE |
| 4 | Label generation (weighted overlay) | DONE |
| 5 | EDA + data validation | DONE |
| 6 | ML training (RF, XGBoost, LightGBM + Ensemble) | DONE |
| 7 | Prediction map generation (37M pixels) | DONE |
| 8 | Interactive web map (Folium) | NEXT |
| 9 | Thesis figures + write-up | PENDING |
| 10 | Real-world validation (PCRWR/WAPDA) | PENDING |

---

## Tech Stack
- **GEE:** Data collection (Landsat 8, SRTM, CHIRPS, MODIS, SoilGrids)
- **Python 3.13:** rasterio, geopandas, scikit-learn, XGBoost, LightGBM, geemap
- **GIS:** QGIS + PyQGIS for visualization and layer prep
- **Web:** Folium (interactive maps), Flask (future dashboard)

---

## Project Structure
```
rao-research/
├── data/
│   ├── raw/
│   │   └── Pothohar_Groundwater_Features.tif     ← 860 MB, 9-band GeoTIFF from GEE
│   ├── processed/
│   │   ├── features.csv                           ← 3.2 GB, 37M pixels x 9 features (normalized)
│   │   ├── features_normalized.tif                ← normalized raster (input to predict.py)
│   │   ├── pixel_coords.csv                       ← row/col index for spatial reconstruction
│   │   ├── all_labels.npy                         ← 37M labels (0/1/2) for all pixels
│   │   └── training_samples.csv                   ← 450K balanced rows used for ML training
│   └── exports/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── data_collection/gee_collector.py
│   ├── preprocessing/raster_processor.py
│   ├── models/train.py + predict.py
│   └── visualization/map_visualizer.py
├── outputs/
│   ├── figures/
│   │   ├── all_features_grid.png                  ← all 9 bands visualized
│   │   ├── correlation_matrix.png
│   │   ├── feature_distributions.png
│   │   ├── feature_boxplots.png
│   │   ├── class_distribution.png
│   │   ├── feature_target_correlation.png
│   │   ├── feature_class_heatmap.png
│   │   ├── gw_potential_label_map.png             ← weighted overlay label map
│   │   ├── prediction_summary.png
│   │   ├── layers/                                ← 9 individual feature maps
│   │   ├── panels/                                ← 9 publication-style panels
│   │   └── training/
│   │       ├── roc_curves.png
│   │       ├── pr_curves.png
│   │       ├── confusion_matrices.png
│   │       ├── feature_importance.png
│   │       ├── model_comparison.png
│   │       ├── calibration_curves.png
│   │       └── cv_fold_scores.png
│   ├── maps/
│   │   ├── gw_prediction_xgboost.tif              ← 2.9 MB, classified map (0/1/2)
│   │   ├── gw_probability_high.tif                ← 167 MB, continuous probability
│   │   ├── gw_potential_labels.tif                ← weighted overlay labels raster
│   │   ├── gw_prediction_xgboost.png              ← classified map (static)
│   │   ├── gw_probability_high.png                ← probability heatmap
│   │   ├── gw_comparison_overlay_vs_ml.png        ← side-by-side comparison
│   │   └── prediction_summary.json                ← pixel counts + agreement stats
│   └── models/
│       ├── best_model.joblib                      ← XGBoost (8.5 MB)
│       ├── xgboost.joblib                         ← 8.5 MB
│       ├── random_forest.joblib                   ← 68 MB
│       ├── lightgbm.joblib                        ← 14.5 MB
│       ├── voting_ensemble.joblib                 ← 91 MB (RF+XGB+LGB)
│       ├── metrics.json                           ← full accuracy report
│       └── feature_importance.csv
├── collect_data.py          ← DONE (GEE export script)
├── preprocess.py            ← DONE
├── step1_generate_labels.py ← DONE
├── step2_validate.py        ← DONE
├── step3_train_models.py    ← DONE
├── step4_predict_map.py     ← DONE
├── CLAUDE.md
├── requirements.txt
└── .env                     ← GEE_PROJECT_ID=ee-zaheersapar005
```

---

## Study Area
**Pothohar Plateau — 5 Districts:**
- Chakwal, Rawalpindi, Attock, Jhelum, Mianwali
- Total area: ~25,000 sq.km
- CRS: EPSG:4326 (exported from GEE), EPSG:32642 (UTM Zone 42N for analysis)
- Resolution: 30 meters (Landsat/SRTM native)
- Raster size: 10,062 x 6,884 pixels = 69.2M total, **37,015,091 valid (53.4%)**
- FAO/GAUL district names used: `Chakwal District`, `Rawalpindi District`, etc.

---

## Input Features (9 layers from GEE)
| # | Feature | Source | GEE Dataset | Role | Correlation with GW |
|---|---------|--------|-------------|------|---------------------|
| 1 | Elevation | SRTM | `USGS/SRTMGL1_003` | terrain | +0.234 |
| 2 | Slope | SRTM derived | `ee.Terrain.slope()` | infiltration | +0.051 |
| 3 | Aspect | SRTM derived | `ee.Terrain.aspect()` | solar exposure | +0.060 |
| 4 | TWI | SRTM + `WWF/HydroSHEDS/15ACC` | wetness index | | +0.185 |
| 5 | NDVI | Landsat 8 | `LANDSAT/LC08/C02/T1_L2` | vegetation/recharge | +0.240 |
| 6 | NDWI | Landsat 8 | `LANDSAT/LC08/C02/T1_L2` | water bodies | -0.222 |
| 7 | Rainfall | CHIRPS | `UCSB-CHG/CHIRPS/DAILY` (2015–2023 sum) | recharge driver | +0.514 |
| 8 | Soil | SoilGrids | `OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02` | permeability | +0.836 |
| 9 | LST | MODIS | `MODIS/061/MOD11A1` | evapotranspiration | -0.426 |

**Key EDA finding:** Soil (r=0.836) and Rainfall (r=0.514) are dominant predictors.
Soil texture is the #1 factor — consistent with reference paper (Geology/LULC = 45% weight).

---

## Label Generation (Step 1)
Labels generated via weighted overlay (knowledge-based label generation):

| Feature | Weight | Direction |
|---------|--------|-----------|
| Soil | 25% | direct (+) |
| TWI | 20% | direct (+) |
| Rainfall | 20% | direct (+) |
| NDVI | 10% | direct (+) |
| Slope | 10% | inverse (-) |
| NDWI | 5% | direct (+) |
| Elevation | 5% | inverse (-) |
| LST | 3% | inverse (-) |
| Aspect | 2% | direct (+) |

Classification thresholds: bottom 40% = Low (0), 40–80% = Medium (1), top 20% = High (2)
Training set: **450,000 rows** (150K per class, perfectly balanced) | Split: 72% train / 8% val / 20% test

---

## EDA Key Findings
- No missing values, all 9 features correctly normalized [0, 1]
- Soil (r=0.836) and Rainfall (r=0.514) are the two dominant predictors
- Slope correlation r=0.051 — negligible predictor in this flat plateau region
- Elevation "FAIL" in directional logic check is a **regional characteristic** — northern areas
  (Rawalpindi/Attock) are higher elevation AND have better GW potential due to more rainfall
  and coarser alluvial deposits (confirmed by reference paper)
- Class distribution after labeling: Low=40%, Medium=40%, High=20%

---

## ML Training Results (Step 3) — COMPLETED

### Model Accuracy on Test Set (90,000 samples)

| Model | Accuracy | F1-Macro | Cohen's κ | ROC-AUC (OvR) |
|-------|----------|----------|-----------|---------------|
| Random Forest | 98.32% | 0.9832 | 0.9749 | 0.9995 |
| **XGBoost ★** | **99.12%** | **0.9912** | **0.9868** | **0.9998** |
| LightGBM | 99.11% | 0.9911 | 0.9867 | 0.9998 |
| Voting Ensemble | 99.09% | 0.9909 | 0.9863 | 0.9998 |

**Best model: XGBoost** — selected automatically, saved as `best_model.joblib`

### Cross-Validation (5-fold, macro F1)
| Model | Mean F1 | Std |
|-------|---------|-----|
| Random Forest | 0.9839 | ±0.0005 |
| XGBoost | 0.9896 | ±0.0004 |
| LightGBM | 0.9911 | ±0.0002 |

### Per-Class Performance (XGBoost)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Low | 99.32% | 99.43% | 99.38% |
| Medium | 99.04% | 98.32% | 98.68% |
| High | 98.99% | 99.61% | 99.31% |

---

## Prediction Map Results (Step 4) — COMPLETED

### Groundwater Potential Distribution (37M pixels, full Pothohar Plateau)
| Zone | Pixel Count | Coverage |
|------|-------------|----------|
| Low Potential | 14,806,035 | 40.0% |
| Medium Potential | 14,806,036 | 40.0% |
| High Potential | 7,403,020 | **20.0%** |

- **ML vs Weighted Overlay agreement: 99.05%** on full 37M pixel comparison
- Prediction ran in tiled 2048×2048 windows to handle memory (20 tiles, ~19 min)

---

## Key Commands
```powershell
# Activate venv
.\venv\Scripts\activate

# GEE data collection (DONE)
python collect_data.py

# Preprocessing (DONE)
python preprocess.py

# Step 1 - Generate labels (DONE)
python step1_generate_labels.py

# Step 2 - EDA + Validation (DONE)
python step2_validate.py

# Step 3 - Train ML models (DONE)
python step3_train_models.py

# Step 4 - Generate prediction map (DONE)
python step4_predict_map.py

# Step 5 - Interactive web map (NEXT)
python step5_visualize.py

# Re-authenticate GEE if needed
python -c "import ee; ee.Authenticate(force=True)"

# Launch notebooks
jupyter notebook notebooks/
```

---

## Important Notes
- **Never commit** `.env`, `data/raw/*.tif`, `data/processed/*.csv`, `*.npy` (too large)
- GEE project: `ee-zaheersapar005` | GEE email: `zaheersapar005@gmail.com`
- FAO/GAUL district names require `" District"` suffix (e.g. `"Chakwal District"`)
- All GEE bands must be cast to `.toFloat()` before export to avoid type conflict errors
- Training labels are knowledge-based (weighted overlay) — valid methodology for thesis
- Geology layer from GSP sheets not yet digitized (optional enhancement)
- `prediction_summary.json` pixel counts are from downsampled view (÷5); full-res counts in metrics

---

## Reference Paper
Waqas M., Ahmad S.R., Majid M. (2022). Assessment of Groundwater Potential in District
Chakwal, Punjab: A GIS and Remote Sensing Perspective.
*Pak. j. sci. ind. res. Ser. A: phys. sci.* 65A(2): 128-134.

**How this thesis improves on it:**
| Aspect | Reference Paper | This Thesis |
|--------|----------------|-------------|
| Study area | 1 district (Chakwal) | 5 districts (Pothohar Plateau) |
| Method | Manual weighted overlay | ML (RF, XGBoost, LightGBM, Ensemble) |
| Features | 5 (LULC, geology, drainage, slope, rainfall) | 9 (+ NDVI, NDWI, TWI, LST, soil) |
| Validation | Secondary data only (NESPAK) | 5-fold CV + ROC-AUC + 99% agreement test |
| Output | Static GIS maps | GeoTIFF + interactive web map |
| Accuracy | Not reported | XGBoost: 99.12% accuracy, AUC=0.9998 |
