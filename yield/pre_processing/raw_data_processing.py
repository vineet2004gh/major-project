import pandas as pd
import  ndvi_data_processing as ndvi

output_filename = '../dataset/processed_dataset/transformed_raw_data.csv'

def transform_yield_data(file_path):
    """
    Transforms a DataFrame from a wide format (with yield data in multiple columns)
    to a long format with 'crop_name' and 'yield' columns.

    Args:
        file_path (str): The file path to the CSV file.

    Returns:
        pandas.DataFrame: The transformed DataFrame, or None if an error occurs.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        print("--- Original Data (first 5 rows) ---")
        print(df.head())
        print("\n--- Original Columns ---")
        print(df.columns.tolist())

        # 1. Define the pattern to find the yield columns
        yield_pattern = "YIELD (Kg per ha)"
        
        # 2. Identify the yield columns
        yield_cols = [col for col in df.columns if yield_pattern in col]
        
        # 3. Identify the ID columns (all columns that are *not* yield columns)
        id_cols = [col for col in df.columns if col not in yield_cols]

        if not yield_cols:
            print(f"\nError: No columns found containing the text '{yield_pattern}'.")
            print("Please check your column names for typos.")
            return None
        
        print(f"\nIdentified ID columns: {id_cols}")
        print(f"Identified Yield columns to transform: {yield_cols}")

        # 4. "Melt" the DataFrame from wide to long format
        df_long = df.melt(
            id_vars=id_cols,
            value_vars=yield_cols,
            var_name='original_col_name',  # This new column holds the original column names
            value_name='yield'           # This new column holds the values
        )

        # 5. Create the new 'crop_name' column
        # This removes the " + YIELD (kg per ha)" part from the original name
        suffix_to_remove = 'YIELD (Kg per ha)'
        df_long['crop_name'] = df_long['original_col_name'].str.replace(suffix_to_remove, '', regex=False).str.strip()
        df_long['crop_yield'] = pd.to_numeric(df_long['yield'], errors='coerce')
        # 6. Create the final DataFrame
        # We reorder the columns to be clean and drop the temporary 'original_col_name'
        final_cols = id_cols + ['crop_name', 'yield']
        df_final = df_long[final_cols]
        df_final = df_final.dropna(subset=['yield'])
        df_final = df_final[df_final['yield'] != 0].copy()
        df_final = df_final.drop(columns=["Dist Code", "State Code","State Name"])
        df_final = df_final.rename(columns={"Dist Name":"district","Year":"year"})
        print("\n--- Transformed Data (first 5 rows) ---")
        print(df_final.head())
        
        # 7. Save the transformed data to a new CSV file
        df_final.to_csv(output_filename, index=False)
        
        print(f"\nSuccessfully transformed data and saved to {output_filename}")
        
        return df_final

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- How to use the code ---

# 1. Replace 'your_file.csv' with the actual path to your file
input_file = '../dataset/raw_dataset/raw_data_2.csv'

# 2. Run the function
transformed_data = transform_yield_data(input_file)

# 3. The function will print the results and save the new file.
if transformed_data is not None:
    print("\nTransformation complete.")