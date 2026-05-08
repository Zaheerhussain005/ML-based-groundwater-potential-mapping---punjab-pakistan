"""
Google Earth Engine data collection for groundwater potential mapping.
Collects: DEM, slope, TWI, NDVI, NDWI, rainfall, soil, LST layers.
"""

import ee
import geemap
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ID = os.getenv("GEE_PROJECT_ID")
DISTRICTS = os.getenv("DISTRICTS", "Chakwal,Rawalpindi,Attock,Jhelum,Mianwali").split(",")
RESOLUTION = int(os.getenv("RESOLUTION_METERS", 30))
EXPORT_FOLDER = "GW_Research"


def init_gee():
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    logger.info(f"GEE initialized — project: {PROJECT_ID}")


def get_study_area() -> ee.Geometry:
    pakistan = ee.FeatureCollection("FAO/GAUL/2015/level2") \
        .filter(ee.Filter.eq("ADM1_NAME", "Punjab")) \
        .filter(ee.Filter.inList("ADM2_NAME", DISTRICTS))
    return pakistan.geometry()


def get_dem_layers(aoi: ee.Geometry) -> ee.Image:
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    slope = ee.Terrain.slope(dem).rename("Slope")
    aspect = ee.Terrain.aspect(dem).rename("Aspect")
    flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").clip(aoi)
    slope_rad = slope.multiply(3.14159 / 180)
    twi = flow_acc.divide(slope_rad.tan().add(0.001)).log().rename("TWI")
    return dem.rename("Elevation").addBands(slope).addBands(aspect).addBands(twi)


def get_landsat_layers(aoi: ee.Geometry) -> ee.Image:
    def mask_clouds(image):
        qa = image.select("QA_PIXEL")
        return image.updateMask(qa.bitwiseAnd(1 << 3).eq(0))

    landsat = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate("2020-01-01", "2023-12-31")
        .filter(ee.Filter.lt("CLOUD_COVER", 10))
        .map(mask_clouds)
        .median()
        .clip(aoi)
    )
    optical = landsat.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]) \
        .multiply(0.0000275).add(-0.2)

    ndvi = optical.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    ndwi = optical.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
    return ndvi.addBands(ndwi)


def get_rainfall(aoi: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate("2015-01-01", "2023-12-31")
        .sum()
        .clip(aoi)
        .rename("Rainfall")
    )


def get_soil(aoi: ee.Geometry) -> ee.Image:
    return (
        ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
        .select("b0")
        .clip(aoi)
        .rename("Soil")
    )


def get_lst(aoi: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterBounds(aoi)
        .filterDate("2020-01-01", "2023-12-31")
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02).subtract(273.15)
        .clip(aoi)
        .rename("LST")
    )


def collect_all_layers() -> ee.Image:
    logger.info("Collecting all GEE layers...")
    aoi = get_study_area()
    dem_layers = get_dem_layers(aoi)
    landsat_layers = get_landsat_layers(aoi)
    rainfall = get_rainfall(aoi)
    soil = get_soil(aoi)
    lst = get_lst(aoi)

    stacked = dem_layers \
        .addBands(landsat_layers) \
        .addBands(rainfall) \
        .addBands(soil) \
        .addBands(lst)

    logger.info(f"Bands collected: {stacked.bandNames().getInfo()}")
    return stacked, aoi


def export_to_drive(image: ee.Image, aoi: ee.Geometry):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description="Pothohar_Groundwater_Features",
        folder=EXPORT_FOLDER,
        region=aoi,
        scale=RESOLUTION,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    logger.info(f"Export started → Google Drive / {EXPORT_FOLDER} (~10-20 min)")
    return task


if __name__ == "__main__":
    init_gee()
    image, aoi = collect_all_layers()
    export_to_drive(image, aoi)
