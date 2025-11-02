import pandas as pd
import google.generativeai as genai
import ee
import geopandas as gpd  # Make sure this is installed (pip install geopandas)
import os
import time
import json
import sys

# --- 1. CONFIGURATION & AUTHENTICATION ---

# !! PASTE YOUR GEMINI KEY HERE !!
GEMINI_API_KEY = 'AIzaSyAOwjSro3qEp20vYBdM4P7Gw58CktWzX8g'

# !! UPDATE YOUR SHP FOLDER PATH HERE !!
# This is the path to the folder on your computer
# where all your .shp (and .shx, .dbf) files are stored.
SHP_FOLDER_PATH = '/yield/raw_dataset/districts'  # Example for Windows
# SHP_FOLDER_PATH = '/home/YourName/shp_files' # Example for Linux/Mac

# Authenticate GEE
try:
    ee.Initialize(project="gen-lang-client-0111974159")
except Exception as e:
    print(e)
    sys.exit(1)

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-pro')
except Exception as e:
    print(f"Gemini configuration failed. Check your API key. Error: {e}")
    sys.exit(1)


# --- 2. HELPER FUNCTION: Get Crop Season (Gemini API) ---
# (This function is unchanged)

def get_crop_season(crop_name, district_name):
    return 6,10
    """
    Uses the Gemini API to find the growing season for a crop in a district.
    """
    prompt = f"""
    What are the primary growing season months for the crop "{crop_name}"
    in the "{district_name}" district of India?

    Please respond ONLY with a JSON object containing two keys:
    "start_month": (integer)
    "end_month": (integer)

    Example: {{"start_month": 6, "end_month": 10}}
    """
    try:
        response = gemini_model.generate_content(prompt)    
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        data = json.loads(json_text)
        
        if 'start_month' in data and 'end_month' in data:
            return data['start_month'], data['end_month']
        else:
            print(f"  [Gemini Error] Unexpected JSON format: {data}")
            return None, None
            
    except Exception as e:
        print(f"  [Gemini Error] Failed to get season for {crop_name} in {district_name}: {e}")
        return None, None

# --- 3. HELPER FUNCTION: Get Monthly NDVI (GEE API) ---
# (This function is unchanged from the previous "local file" version)

# --- 3. HELPER FUNCTION: Get Monthly NDVI (GEE API) ---
# (This function is MODIFIED to fix the .ifNull error)

# --- 3. HELPER FUNCTION: Get Monthly NDVI (GEE API) ---
# (This function is REWRITTEN to use a client-side loop)

def get_monthly_ndvi_stats(district_geom_ee, district_name, year, start_month, end_month):
    """
    Uses the GEE API to calculate mean NDVI for each month in the season.
    This version uses a client-side Python loop to avoid server-side mapping errors.
    """
    try:
        results = {}

        # Use a client-side Python loop for each month
        for month_num in range(start_month, end_month + 1):
            # 1. Define date range for this single month
            start_date = ee.Date.fromYMD(year, month_num, 1)
            end_date = start_date.advance(1, 'month')

            # 2. Load NDVI collection and filter for this month
            ndvi_collection = ee.ImageCollection('MODIS/061/MOD13Q1') \
                .filterDate(start_date, end_date) \
                .select('NDVI') \
                .map(lambda img: img.multiply(0.0001).set('system:time_start', img.get('system:time_start')))

            # 3. Create a single mean composite for the month
            monthly_composite = ndvi_collection.mean()

            # 4. Calculate the mean NDVI over the provided geometry
            stats = monthly_composite.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=district_geom_ee,  # Using the client-side geometry
                scale=250,
                maxPixels=1e9
            )

            # 5. Get the result for this month
            # .unmask(-9999) handles nulls if no data
            # .getInfo() pulls the single number from GEE
            mean_ndvi = ee.Number(stats.get('NDVI')).getInfo()

            # 6. Store the result
            results[f'ndvi_month_{month_num}'] = -1.0 if mean_ndvi == -9999 else mean_ndvi

        # Return the dictionary of results for all months
        return results

    except Exception as e:
        # This will now catch any errors from the reduceRegion or getInfo calls
        print(f"  [GEE Error] Failed to get NDVI for {district_name}, month {month_num}: {e}")
        return {}

# --- 4. MAIN PROCESSING LOOP (MODIFIED for .shp files) ---

def main():
    # !! UPDATE YOUR FILENAME HERE !!
    try:
        df = pd.read_csv('processed_dataset/transformed_crop_yields.csv')
    except FileNotFoundError:
        print(f"Error: 'transformed_crop_yields.csv' not found. Please check the filename.")
        return
        
    if not os.path.exists(SHP_FOLDER_PATH):
        print(f"Error: SHP folder not found at '{SHP_FOLDER_PATH}'. Please check the path.")
        return

    processed_data = []

    for index, row in df.iterrows():
        print(f"\nProcessing ({index+1}/{len(df)}): {row['crop_name']} in {row['district']} ({row['year']})...")

        # --- NEW: Load SHP file locally ---
        district_name = row['district']
        # Assumes file is named 'DistrictName.shp'
        shp_file_path = os.path.join(SHP_FOLDER_PATH, f"{district_name}.shp")
        
        if not os.path.exists(shp_file_path):
            print(f"  [File Error] SHP file not found: {shp_file_path}")
            print("  -> Skipping row.")
            continue
            
        try:
            # 1. Read SHP using geopandas
            gdf = gpd.read_file(shp_file_path)
            
            # 2. *** IMPORTANT ***
            # Re-project to EPSG:4326 (WGS84) if it's not already. GEE requires this.
            if gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            
            # 3. Convert geometry to GeoJSON
            district_geojson = gdf.geometry.iloc[0].__geo_interface__
            
            # 4. Convert GeoJSON to an ee.Geometry object
            district_geom_ee = ee.Geometry(district_geojson)
            
        except Exception as e:
            print(f"  [SHP Error] Failed to read or parse {shp_file_path}: {e}")
            print("  -> Skipping row.")
            continue
        # --- End of new SHP loading ---

        # 1. Get Season from Gemini
        start_m, end_m = get_crop_season(row['crop_name'], district_name)

        if start_m is None or end_m is None:
            print("  -> Skipping row (could not determine season).")
            continue
        
        print(f"  -> Season: {start_m} to {end_m}")

        # 2. Get NDVI data from GEE (pass the GEE geometry object)
        ndvi_data = get_monthly_ndvi_stats(
            district_geom_ee=district_geom_ee, # <-- Pass the geometry
            district_name=district_name,
            year=row['year'],
            start_month=start_m,
            end_month=end_m
        )
        
        if not ndvi_data:
            print("  -> Skipping row (could not retrieve NDVI data).")
            break

        print(f"  -> NDVI Data: {ndvi_data}")

        # 3. Combine original data with new data
        new_row = row.to_dict()
        new_row['season_start_month'] = start_m
        new_row['season_end_month'] = end_m
        new_row.update(ndvi_data) 
        
        processed_data.append(new_row)
        
        # Delay to avoid hitting API rate limits
        time.sleep(1) 

    # 4. Create and save the final DataFrame
    final_df = pd.DataFrame(processed_data)
    
    # Reorder columns (optional)
    if not final_df.empty:
        base_cols = ['district', 'crop_name', 'year', 'crop_yield', 'season_start_month', 'season_end_month']
        ndvi_cols = sorted([col for col in final_df.columns if col.startswith('ndvi_')])
        final_df = final_df[base_cols + ndvi_cols]

    output_filename = 'dataset/processed_dataset/final_crop_yield_with_ndvi.csv'
    final_df.to_csv(output_filename, index=False)

    print(f"\n--- Processing Complete! ---")
    print(f"Data saved to {output_filename}")
    print(final_df.head())


if __name__ == "__main__":
    main()