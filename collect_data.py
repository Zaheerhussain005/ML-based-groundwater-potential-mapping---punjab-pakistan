import ee
from dotenv import load_dotenv
import os

load_dotenv()
ee.Initialize(project='ee-zaheersapar005')

# Pothohar Plateau - 5 districts
aoi = ee.FeatureCollection('FAO/GAUL/2015/level2') \
    .filter(ee.Filter.eq('ADM0_NAME', 'Pakistan')) \
    .filter(ee.Filter.eq('ADM1_NAME', 'Punjab')) \
    .filter(ee.Filter.inList('ADM2_NAME', [
        'Chakwal District', 'Rawalpindi District', 'Attock District',
        'Jhelum District', 'Mianwali District'
    ])).geometry()
print('Study area loaded')

# 1. DEM + Terrain
dem    = ee.Image('USGS/SRTMGL1_003').clip(aoi)
slope  = ee.Terrain.slope(dem).rename('Slope')
aspect = ee.Terrain.aspect(dem).rename('Aspect')
flow   = ee.Image('WWF/HydroSHEDS/15ACC').clip(aoi)
twi    = flow.divide(slope.multiply(3.14159 / 180).tan().add(0.001)).log().rename('TWI')
print('DEM layers ready')

# 2. Landsat 8 - NDVI, NDWI
def mask_clouds(img):
    return img.updateMask(img.select('QA_PIXEL').bitwiseAnd(1 << 3).eq(0))

ls = (
    ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(aoi)
    .filterDate('2020-01-01', '2023-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 10))
    .map(mask_clouds)
    .median()
    .clip(aoi)
)
opt  = ls.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']).multiply(0.0000275).add(-0.2)
ndvi = opt.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
ndwi = opt.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
print('Landsat layers ready')

# 3. Rainfall - CHIRPS annual sum
rain = (
    ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterBounds(aoi)
    .filterDate('2015-01-01', '2023-12-31')
    .sum()
    .clip(aoi)
    .rename('Rainfall')
)
print('Rainfall ready')

# 4. Soil texture
soil = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02') \
    .select('b0').clip(aoi).rename('Soil')
print('Soil ready')

# 5. Land Surface Temperature - MODIS
lst = (
    ee.ImageCollection('MODIS/061/MOD11A1')
    .filterBounds(aoi)
    .filterDate('2020-01-01', '2023-12-31')
    .select('LST_Day_1km')
    .mean()
    .multiply(0.02).subtract(273.15)
    .clip(aoi)
    .rename('LST')
)
print('LST ready')

# Stack all 9 bands — cast everything to Float32 for consistent export
stack = dem.rename('Elevation').toFloat() \
    .addBands(slope.toFloat()) \
    .addBands(aspect.toFloat()) \
    .addBands(twi.toFloat()) \
    .addBands(ndvi.toFloat()) \
    .addBands(ndwi.toFloat()) \
    .addBands(rain.toFloat()) \
    .addBands(soil.toFloat()) \
    .addBands(lst.toFloat())

print('All bands:', stack.bandNames().getInfo())

# Export to Google Drive
task = ee.batch.Export.image.toDrive(
    image=stack,
    description='Pothohar_Groundwater_Features',
    folder='GW_Research',
    region=aoi,
    scale=30,
    maxPixels=1e13,
    fileFormat='GeoTIFF'
)
task.start()
status = task.status()

print()
print('EXPORT TASK STARTED!')
print('Task ID  :', task.id)
print('Status   :', status['state'])
print()
print('Monitor  : https://code.earthengine.google.com/tasks')
print('Output   : Google Drive > GW_Research > Pothohar_Groundwater_Features.tif')
print('Wait time: ~15-30 minutes')
