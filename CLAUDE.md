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
| 6 | ML training (RF, XGBoost, LightGBM) | NEXT |
| 7 | Prediction map generation | PENDING |
| 8 | Visualization (static + interactive) | PENDING |
| 9 | Accuracy validation | PENDING |

---

## Tech Stack
- **GEE:** Data collection (Landsat 8, SRTM, CHIRPS, MODIS, SoilGrids)
- **Python:** rasterio, geopandas, scikit-learn, XGBoost, LightGBM, geemap
- **GIS:** QGIS + PyQGIS for visualization and layer prep
- **Web:** Folium (interactive maps), Flask (future dashboard)

---

## Project Structure
```
rao-research/
├── data/
│   ├── raw/
│   │   └── Pothohar_Groundwater_Features.tif   ← 860 MB, 9-band GeoTIFF from GEE
│   ├── processed/
│   │   ├── features.csv                         ← 3.2 GB, 37M pixels x 9 features (normalized)
│   │   ├── features_normalized.tif              ← normalized raster version
│   │   ├── pixel_coords.csv                     ← row/col index for spatial reconstruction
│   │   ├── all_labels.npy                       ← 37M labels (0/1/2) for all pixels
│   │   └── training_samples.csv                 ← 450K balanced rows for ML training
│   └── exports/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── data_collection/
│   │   └── gee_collector.py
│   ├── preprocessing/
│   │   └── raster_processor.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── visualization/
│       └── map_visualizer.py
├── outputs/
│   ├── maps/         ← prediction GeoTIFFs go here
│   ├── models/       ← trained .joblib models go here
│   └── figures/      ← EDA plots already saved here (6 plots)
├── step1_generate_labels.py    ← DONE
├── step2_validate.py           ← DONE
├── collect_data.py             ← DONE (GEE export script)
├── preprocess.py               ← DONE
├── CLAUDE.md
├── requirements.txt
└── .env                        ← GEE_PROJECT_ID=ee-zaheersapar005
```

---

## Study Area
**Pothohar Plateau — 5 Districts:**
- Chakwal, Rawalpindi, Attock, Jhelum, Mianwali
- Total area: ~25,000 sq.km
- CRS: EPSG:4326 (exported from GEE)
- Resolution: 30 meters
- Raster size: 10,062 x 6,884 pixels = 69.2M total, 37M valid (53.4%)

---

## Input Features (9 layers from GEE)
| # | Feature | Source | Role | Correlation with GW |
|---|---------|--------|------|---------------------|
| 1 | Elevation | SRTM | terrain | +0.234 |
| 2 | Slope | SRTM derived | infiltration | +0.051 |
| 3 | Aspect | SRTM derived | solar exposure | +0.060 |
| 4 | TWI | SRTM + flow acc | wetness index | +0.185 |
| 5 | NDVI | Landsat 8 | vegetation/recharge | +0.240 |
| 6 | NDWI | Landsat 8 | water bodies | -0.222 |
| 7 | Rainfall | CHIRPS | recharge driver | +0.514 |
| 8 | Soil | SoilGrids | permeability | +0.836 |
| 9 | LST | MODIS | evapotranspiration | -0.426 |

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

Classification: bottom 40% = Low (0), 40-80% = Medium (1), top 20% = High (2)
Training set: **450,000 rows** (150K per class, perfectly balanced)
Full label set: **37,015,091 labels** saved in `all_labels.npy`

---

## EDA Key Findings
- No missing values, all features correctly normalized [0, 1]
- Elevation "FAIL" in logic check is a **regional characteristic** — in Pothohar,
  northern areas (Rawalpindi/Attock) are higher elevation AND have better GW potential
  due to more rainfall and coarser soil deposits (confirmed by reference paper)
- Slope correlation r=0.051 — negligible predictor in this region
- **Data validated and ready for ML training**

---

## ML Models — NEXT STEP
- **Random Forest** — baseline, feature importance
- **XGBoost** — target best accuracy
- **LightGBM** — fast training, handles class patterns well
- Input: `data/processed/training_samples.csv` (450K rows, 9 features)
- Target: `GW_Potential` (0=Low, 1=Medium, 2=High)
- Validation: 80/20 split + 5-fold cross-validation + ROC-AUC
- Output: `outputs/models/` (.joblib files)
- Next script to write: `step3_train_models.py`

---

## Key Commands
```powershell
# Activate venv
.\venv\Scripts\activate

# Step 1 - Generate labels (DONE)
python step1_generate_labels.py

# Step 2 - EDA + Validation (DONE)
python step2_validate.py

# Step 3 - Train ML models (NEXT)
python step3_train_models.py

# Step 4 - Generate prediction map
python step4_predict_map.py

# Step 5 - Visualize
python step5_visualize.py

# Launch notebooks
jupyter notebook notebooks/
```

---

## Important Notes
- **Never commit** `.env`, `data/raw/*.tif`, `data/processed/*.csv` (too large)
- GEE project: `ee-zaheersapar005` | re-auth: `python -c "import ee; ee.Authenticate(force=True)"`
- Training labels are knowledge-based (weighted overlay) — valid methodology for thesis
- Geology layer from GSP sheets not yet digitized (optional enhancement)
- All 37M pixel labels stored in `all_labels.npy` for final map reconstruction

---

## Reference Paper
Waqas M., Ahmad S.R., Majid M. (2022). Assessment of Groundwater Potential in District
Chakwal, Punjab: A GIS and Remote Sensing Perspective.
*Pak. j. sci. ind. res. Ser. A: phys. sci.* 65A(2): 128-134.

**How this thesis improves on it:**
- 5 districts vs 1 (Chakwal only)
- ML models vs manual weighted overlay
- 9 features vs 5 features
- Proper cross-validation vs secondary data only
- Interactive web map output
