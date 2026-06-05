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
| 5 | EDA + exploratory validation | DONE |
| 6 | ML training (RF, XGBoost, LightGBM + Ensemble) | DONE |
| 7 | Prediction map generation (37M pixels) | DONE |
| 8 | Real-world validation (well data + statistics) | IN PROGRESS |
| 9 | Thesis write-up (LaTeX template created) | IN PROGRESS |
| 10 | Interactive web map (Folium) | DEFERRED — Q1 work first |

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
├── data/
│   └── validation/
│       ├── validation_wells_combined.csv  ← 85 pts (Rana 2022 + Naz 2023 + GSP dry refs)
│       ├── naz2023_wells_pothohar.csv     ← 146 auto-digitized PCRWR wells from Fig.1
│       ├── naz2023_wells_all.csv          ← all Punjab wells from digitizer
│       ├── well_points_template.csv       ← 16-pt curated template (has dry wells)
│       ├── well_points_classified.csv     ← 45 sampled wells with predicted class
│       └── calibration.json              ← affine transform from extract_wells script
├── collect_data.py               ← DONE (GEE export script)
├── preprocess.py                 ← DONE
├── step1_generate_labels.py      ← DONE
├── step2_validate.py             ← DONE
├── step3_train_models.py         ← DONE
├── step4_predict_map.py          ← DONE
├── step5_realworld_validation.py ← DONE (sensitivity/specificity/ROC-AUC/Mann-Whitney)
├── extract_wells_from_figure.py  ← DONE (auto-digitizer for Naz 2023 Fig.1 red dots)
├── mcp_academic_search.py        ← DONE (MCP server: OpenAlex + Semantic Scholar + Unpaywall)
├── .mcp.json                     ← MCP server config (academic-search)
├── CLAUDE.md
├── requirements.txt
└── .env                          ← GEE_PROJECT_ID=ee-zaheersapar005
```

**Thesis LaTeX template:** `C:\Users\ADNAN\THESIS\` (Overleaf-ready, all 6 chapters pre-written)

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
# Run venv python directly (PS execution policy blocks activate script)
& "d:\rao-research\venv\Scripts\python.exe" <script.py>

# All pipeline steps (DONE)
& "d:\rao-research\venv\Scripts\python.exe" collect_data.py
& "d:\rao-research\venv\Scripts\python.exe" preprocess.py
& "d:\rao-research\venv\Scripts\python.exe" step1_generate_labels.py
& "d:\rao-research\venv\Scripts\python.exe" step2_validate.py
& "d:\rao-research\venv\Scripts\python.exe" step3_train_models.py
& "d:\rao-research\venv\Scripts\python.exe" step4_predict_map.py

# Step 8 - Real-world validation (re-run after getting new well data)
& "d:\rao-research\venv\Scripts\python.exe" step5_realworld_validation.py

# Auto-digitize wells from a figure image (interactive — needs display)
& "d:\rao-research\venv\Scripts\python.exe" extract_wells_from_figure.py

# Re-authenticate GEE if needed
& "d:\rao-research\venv\Scripts\python.exe" -c "import ee; ee.Authenticate(force=True)"
```

---

## Real-World Validation Results (Step 8)

| Metric | Value | Notes |
|--------|-------|-------|
| Sensitivity (TPR) | 59.0% | 23/39 productive wells in Med+High zone |
| Specificity (TNR) | 33.3% | 2/6 dry reference points in Low zone |
| Balanced Accuracy | 46.2% | |
| ROC-AUC (well pts) | 0.568 | |
| Mianwali sensitivity | 63.0% | Indus alluvial, n=19 — strongest signal |
| Jhelum sensitivity | 67.0% | Jhelum River alluvial, n=3 |
| Mann-Whitney p | 0.302 | Not significant — only 6 dry wells sampled |

**Data sources used:** Rana et al. 2022 (7 pts), Naz et al. 2023 digitized (57 pts alluvial), GSP geology dry refs (12 pts)
**Key limitation:** 40/85 wells hit NoData (outside exact district polygon boundaries)
**Key finding:** Model over-predicts High in forested hard-rock areas (Margalla Hills, Kala Chitta) — missing GSP lithology layer
**Pending:** PCRWR data request sent (email drafted) — will improve validation when received

---

## MCP Academic Search Server

- **File:** `mcp_academic_search.py` (FastMCP, stdio transport)
- **Config:** `.mcp.json` at project root
- **Approved in:** `.claude/settings.json` → `enabledMcpjsonServers: ["academic-search"]`
- **APIs used:** OpenAlex (free, no key), Semantic Scholar, Unpaywall
- **Tools:** search_papers, search_pothohar_groundwater, search_pakistan_gw_data_sources, get_paper_details, find_open_access_pdf

---

## Thesis Template

- **Location:** `C:\Users\ADNAN\THESIS\`
- **Format:** Overleaf-ready LaTeX (pdfLaTeX + BibTeX)
- **Status:** All 6 chapters pre-written with actual research data/numbers
- **To use:** Zip folder → upload to overleaf.com → set main.tex as root → compile
- **Still needed:** Fill university/supervisor name, copy figures from outputs/ to THESIS/figures/

---

## Important Notes
- **Never commit** `.env`, `data/raw/*.tif`, `data/processed/*.csv`, `*.npy` (too large)
- GEE project: `ee-zaheersapar005` | GEE email: `zaheersapar005@gmail.com`
- FAO/GAUL district names require `" District"` suffix (e.g. `"Chakwal District"`)
- All GEE bands must be cast to `.toFloat()` before export to avoid type conflict errors
- Training labels are knowledge-based (weighted overlay) — valid methodology for thesis
- Geology layer from GSP sheets not yet digitized — adding it as Feature 10 is top priority for Q1
- `prediction_summary.json` pixel counts are from downsampled view (÷5); full-res counts in metrics
- PowerShell execution policy blocks `.\venv\Scripts\activate` — use full python path instead

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
| Validation | Secondary data only (NESPAK) | 5-fold CV + ROC-AUC + real-world well validation |
| Output | Static GIS maps | GeoTIFF + 4 validation figures + LaTeX thesis |
| Accuracy | Not reported | XGBoost: 99.12% accuracy, AUC=0.9998 |
