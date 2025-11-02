import ee
import time
import sys

from ee import ComputedObject

# --- 1. CONFIGURATION ---
# !! UPDATE ALL THESE PATHS AND NAMES !!

# Asset path for your district geometries
DISTRICT_ASSET_PATH = 'projects/gen-lang-client-0111974159/assets/districts'

# The property name in your district asset that holds the district name
# (This MUST match the 'district' column in your CSV)
DISTRICT_ID_PROPERTY = 'shapeName'

# Asset path for your CSV with season data
SEASON_TABLE_ASSET_PATH = 'projects/gen-lang-client-0111974159/assets/transformed_crop_yields_2'

# The property name in your CSV that holds the district name
SEASON_ID_PROPERTY = 'district'

# What to name the final exported file in your Google Drive
EXPORT_FILE_NAME = 'raw_ndvi_export'
# -------------------------

# Authenticate GEE
try:
    ee.Initialize(project="gen-lang-client-0111974159")
except Exception as e:
    print(e)
    sys.exit(1)

print("Loading assets...")
# 1. Load Assets
districts_fc = ee.FeatureCollection(DISTRICT_ASSET_PATH)
seasons_fc = ee.FeatureCollection(SEASON_TABLE_ASSET_PATH)

# 2. Join Geometries and Season Data
# We join the season data TO the district geometries.
join_filter = ee.Filter.equals(
    leftField=DISTRICT_ID_PROPERTY,
    rightField=SEASON_ID_PROPERTY
)
inner_join = ee.Join.inner()
joined_features = inner_join.apply(districts_fc, seasons_fc, join_filter)


# "Flatten" the join so properties are combined
def flatten_join(feature):
    # Extract the features properly
    primary = ee.Feature(feature.get('primary'))
    secondary = ee.Feature(feature.get('secondary'))

    # !! THIS IS THE MODIFIED PART !!
    # We take the secondary feature (from your CSV) and
    # just set its geometry to be the primary's geometry.
    # This results in a feature with ONLY the CSV's properties
    # and the district's geometry.
    return secondary.setGeometry(primary.geometry())


joined_fc = joined_features.map(flatten_join)
print("Properties of the first joined feature (should only be CSV props):")
print(joined_fc.first().getInfo()['properties'])


# 3. Define the Server-Side NDVI Calculation
# (This function is MODIFIED to fix the .unmask error)
def calculate_monthly_ndvi(feature):
    start_m = 1
    end_m = 12
    year = ee.Number(feature.get('year'))
    geom = feature.geometry()

    # Load and scale NDVI collection
    ndvi_collection = (
        ee.ImageCollection('MODIS/061/MOD13Q1')
        .select('NDVI')
        .map(lambda img: img.multiply(0.0001)
             .copyProperties(img, ['system:time_start']))
    )

    # List of months
    months = ee.List.sequence(start_m, end_m)

    # Function to compute mean NDVI for a month
    def get_monthly_mean(month):
        month = ee.Number(month)
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')

        monthly_composite = ndvi_collection.filterDate(start_date, end_date).mean()

        stats = monthly_composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=250,
            maxPixels=1e9
        )

        mean_ndvi = ee.Number(stats.get('NDVI', -9999))
        month_key = ee.String('ndvi_month_').cat(month.format('%d'))
        return ee.Dictionary.fromLists([month_key], [mean_ndvi])

    monthly_stats_list = months.map(get_monthly_mean)

    # Safe dictionary merge for iterate()
    def merge_dictionaries(current, prev):
        # Handle None on first iteration safely
        prev = ee.Algorithms.If(
            ee.Algorithms.IsEqual(prev, None),
            ee.Dictionary({}),
            ee.Dictionary(prev)
        )
        current = ee.Dictionary(current)
        return ee.Dictionary(prev).combine(current, overwrite=True)

    merged_stats = ee.Dictionary(monthly_stats_list.iterate(merge_dictionaries, ee.Dictionary({})))

    # Add the merged stats back to the feature
    return feature.set(merged_stats)


# 4. Map the main calculation over all joined features
print("Mapping NDVI calculations over all features...")
final_fc = joined_fc.map(calculate_monthly_ndvi)

# 5. PREPARE FOR EXPORT (NEW STEP)
# To remove the bulky '.geo' column from the final CSV,
# we explicitly set the geometry of each feature to None.
print("Removing geometries for table-only export...")
export_fc = final_fc.map(lambda f: f.setGeometry(None))

# 6. Export the final FeatureCollection to Google Drive
print("Submitting batch export task...")
task = ee.batch.Export.table.toDrive(
    collection=export_fc,  # <-- MODIFIED: Use the geometry-free collection
    description='Batch_NDVI_Export_for_Crops',
    fileNamePrefix=EXPORT_FILE_NAME,
    fileFormat='CSV'
)

task.start()

print(f"Task '{task.id}' started. Check the GEE 'Tasks' tab.")
print("Monitoring task status (will check every 60 seconds)...")

# 7. Monitor the task (Original step 6)
while task.active():
    status = task.status()['state']
    print(f"  Status: {status}")
    if status == 'FAILED':
        print(f"  Error: {task.status()['error_message']}")
        break
    time.sleep(60)

print(f"Task finished with state: {task.status()['state']}")
if task.status()['state'] == 'COMPLETED':
    print(f"Success! Check your Google Drive for '{EXPORT_FILE_NAME}.csv'.")