import pandas as pd
import numpy as np

output_transformed_ndvi_file = '../dataset/processed_dataset/transformed_crop_nvdi_yields_2.csv'
raw_ndvi_file = '../dataset/processed_dataset/raw_ndvi_export.csv'
ndvi_based_metrics_file = '../dataset/processed_dataset/training_ndvi_data_2.csv'
processed_ndvi_file = "../dataset/processed_dataset/transformed_crop_nvdi_yields_2.csv"
crops_season_file = "../dataset/raw_dataset/crops_season_data.csv"

def ndvi_csv_transform(file_path):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        print(df.columns.tolist())

        # 1. Define the pattern to find the yield columns
        yield_pattern = "ndvi_month_"

        # 2. Identify the yield columns
        yield_cols = [col for col in df.columns if yield_pattern in col]
        sorted(yield_cols)
        # 3. Identify the ID columns (all columns that are *not* yield columns)
        id_cols = [col for col in df.columns if col not in yield_cols]



        print(f"\nIdentified ID columns: {id_cols}")
        print(f"Identified Yield columns to transform: {yield_cols}")


        # We reorder the columns to be clean and drop the temporary 'original_col_name'
        final_cols = ['district','year','crop_name', 'crop_yield'] + yield_cols
        df_final = df[final_cols]
        print("\n--- Transformed Data (first 5 rows) ---")
        print(df_final.head())

        # 7. Save the transformed data to a new CSV file
        df_final.to_csv(output_transformed_ndvi_file, index=False)

        print(f"\nSuccessfully transformed data and saved to {output_transformed_ndvi_file}")

        return df_final

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def crop_season_based_month_mapping(ndvi_data_csv, season_data_csv):

    df_ndvi = pd.read_csv(ndvi_data_csv)
    df_timing = pd.read_csv(season_data_csv)

    # --- Main Logic ---

    # 1. Merge the dataframes to get start/end months for each row
    df_merged = pd.merge(df_ndvi, df_timing, on='crop_name', how='left')
    # 2. Define the function to filter NDVI values for a single row
    def filter_ndvi_row_with_stats(row):
        start = row['Sowing_Start_Month']
        end = row['Harvest_End_Month']

        # Get all column names that start with 'ndvi_'
        ndvi_cols = [col for col in row.index if col.startswith('ndvi_month_')]

        # If start/end info is missing, nullify all and set new stats to NaN
        if pd.isna(start) or pd.isna(end):
            for col in ndvi_cols:
                row[col] = np.nan

            # Add new stats as NaN
            row['ndvi_max'] = np.nan
            row['ndvi_mean'] = np.nan
            row['ndvi_min'] = np.nan
            row['ndvi_std_dev'] = np.nan
            row['season_length'] = np.nan
            return row

        # Convert to int (they might be read as floats, e.g., 6.0)
        start = int(start)
        end = int(end)

        # --- NEW: Calculate Season Length ---
        if start <= end:
            # Case 1: Normal season (e.g., Start: 6, End: 10 -> 10-6+1 = 5)
            row['season_length'] = (end - start) + 1
        else:
            # Case 2: Cross-year season (e.g., Start: 10, End: 3)
            # Months in first year (10, 11, 12) + months in second year (1, 2, 3)
            row['season_length'] = (12 - start + 1) + end

        # --- NEW: List to store valid NDVI values *within* the season ---
        season_ndvi_values = []

        # Iterate through months 1 to 12
        for m in range(1, 13):
            col_name = f'ndvi_month_{m}'

            # Check if the column exists in the row
            if col_name in row:
                is_in_range = False

                # Case 1: Normal season (e.g., Start: 6, End: 10)
                if start <= end:
                    if start <= m <= end:
                        is_in_range = True

                # Case 2: Cross-year season (e.g., Start: 10, End: 3)
                else:
                    if m >= start or m <= end:
                        is_in_range = True

                # --- MODIFIED: Check if in range ---
                if is_in_range:
                    # Month is IN the season.
                    # If the value is valid (not NaN), add it to our list.
                    if not pd.isna(row[col_name]) and row[col_name] != -9999:
                        season_ndvi_values.append(row[col_name])
                else:
                    # Month is *not* in the valid range, set its NDVI to NaN (Original behavior)
                    row[col_name] = np.nan

        # --- NEW: Calculate and store new stats ---
        if len(season_ndvi_values) > 0:
            # We have valid data, so we can calculate stats
            row['ndvi_max'] = np.max(season_ndvi_values)
            row['ndvi_mean'] = np.mean(season_ndvi_values)
            row['ndvi_min'] = np.min(season_ndvi_values)
            row['ndvi_std_dev'] = np.std(season_ndvi_values)
        else:
            # No valid NDVI data was found *within* the growing season
            # Set all stats to NaN
            row['ndvi_max'] = np.nan
            row['ndvi_mean'] = np.nan
            row['ndvi_min'] = np.nan
            row['ndvi_std_dev'] = np.nan

        return row

    # 3. Apply the function to every row
    df_filtered = df_merged.apply(filter_ndvi_row_with_stats, axis=1)
    listtodrop  = df_timing.head()
    listtodrop= listtodrop.drop("crop_name",axis=1)
    df_filtered = df_filtered.drop(listtodrop,axis=1)
    df_filtered = df_filtered.drop("year",axis=1)
    listtodrop = df_ndvi.head()
    torem = ndvi_cols = [col for col in listtodrop if col.startswith('ndvi_month_')]
    # 4. Save the result to a new CSV
    df_filtered = df_filtered.drop(torem,axis=1)
    df_filtered.to_csv(ndvi_based_metrics_file, index=False)

    print(f"\n--- Filtered NDVI Data (Saved to '{ndvi_based_metrics_file}') ---")



transformed_data = ndvi_csv_transform(raw_ndvi_file)
crop_season_based_month_mapping(ndvi_data_csv=processed_ndvi_file,season_data_csv=crops_season_file)
