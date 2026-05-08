"""
Run trained model on full raster to produce groundwater potential map.
"""

import numpy as np
import rasterio
import joblib
from pathlib import Path
from loguru import logger

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("outputs/models")
MAPS_DIR = Path("outputs/maps")
MAPS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["Elevation", "Slope", "Aspect", "TWI", "NDVI", "NDWI", "Rainfall", "Soil", "LST"]


def predict_raster(model_name: str = "xgboost"):
    model = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    raster_path = PROCESSED_DIR / "features_normalized.tif"

    with rasterio.open(raster_path) as src:
        data = src.read()
        meta = src.meta.copy()
        nodata = src.nodata

    bands, rows, cols = data.shape
    flat = data.reshape(bands, -1).T

    # Mask nodata
    if nodata is not None:
        valid_mask = ~(flat == nodata).any(axis=1)
    else:
        valid_mask = ~np.isnan(flat).any(axis=1)

    predictions = np.full(rows * cols, -1, dtype=np.int8)
    predictions[valid_mask] = model.predict(flat[valid_mask])

    pred_map = predictions.reshape(rows, cols)

    meta.update(count=1, dtype="int8", nodata=-1)
    output_path = MAPS_DIR / f"groundwater_potential_{model_name}.tif"
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(pred_map[np.newaxis, :, :])

    logger.info(f"Prediction map saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    predict_raster("xgboost")
