import pandas as pd
import  csv
def get_distinct_season(inputFile):
    df = pd.read_csv(inputFile)
    crops = df["crop_name"]
    crops = crops.drop_duplicates()
    crops.to_csv("crop_names.csv",index=0)



inputFile = "transformed_crop_nvdi_yields.csv"
get_distinct_season(inputFile)
