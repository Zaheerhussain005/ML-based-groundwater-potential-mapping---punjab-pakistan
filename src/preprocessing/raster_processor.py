"""
Raster preprocessing: normalization, resampling, sampling points for ML training.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from loguru import logger


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raster(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read()
        meta = src.meta.copy()
    return data, meta


def normalize_bands(data: np.ndarray) -> np.ndarray:
    bands, rows, cols = data.shape
    flat = data.reshape(bands, -1).T
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(flat).T
    return normalized.reshape(bands, rows, cols)


def save_raster(data: np.ndarray, meta: dict, output_path: str):
    meta.update(count=data.shape[0], dtype="float32")
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data.astype("float32"))
    logger.info(f"Saved: {output_path}")


def raster_to_dataframe(raster_path: str, band_names: list[str]) -> pd.DataFrame:
    """Convert multi-band raster to flat DataFrame for ML training."""
    with rasterio.open(raster_path) as src:
        data = src.read()
        nodata = src.nodata

    bands, rows, cols = data.shape
    flat = data.reshape(bands, -1).T
    df = pd.DataFrame(flat, columns=band_names)

    if nodata is not None:
        df = df[~(df == nodata).any(axis=1)]

    df = df.dropna()
    logger.info(f"DataFrame shape: {df.shape}")
    return df


BAND_NAMES = [
    "Elevation", "Slope", "Aspect", "TWI",
    "NDVI", "NDWI",
    "Rainfall", "Soil", "LST"
]


if __name__ == "__main__":
    raw_tif = RAW_DIR / "Pothohar_Groundwater_Features.tif"
    if raw_tif.exists():
        data, meta = load_raster(str(raw_tif))
        normalized = normalize_bands(data)
        save_raster(normalized, meta, str(PROCESSED_DIR / "features_normalized.tif"))
        df = raster_to_dataframe(str(PROCESSED_DIR / "features_normalized.tif"), BAND_NAMES)
        df.to_csv(PROCESSED_DIR / "features.csv", index=False)
        logger.info("Preprocessing complete.")
    else:
        logger.warning(f"Raw raster not found: {raw_tif}")
