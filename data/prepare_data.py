import pandas as pd
import numpy as np
import os
import urllib.request

RAW_PATH     = os.path.join(os.path.dirname(__file__), "raw.csv")
CLEANED_PATH = os.path.join(os.path.dirname(__file__), "cleaned.csv")

LIMIT = 50000
API_URL = (
    "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
    "/soil-sensor-readings-historical-data/exports/csv"
    f"?limit={LIMIT}&timezone=UTC&use_labels=false&delimiter=%2C"
)

print(f"Downloading up to {LIMIT} records...")
urllib.request.urlretrieve(API_URL, RAW_PATH)
print(f"Raw data saved to {RAW_PATH}")

df = pd.read_csv(RAW_PATH)
print(f"\nRaw shape: {df.shape}")
print("\nMissing values per column:")
print(df.isnull().sum())

df["local_time"] = pd.to_datetime(df["local_time"], utc=True, errors="coerce")

df.dropna(subset=["local_time", "site_name", "probe_measure"], inplace=True)

df.drop_duplicates(inplace=True)

df["soil_value"] = df.groupby(["site_name", "probe_measure"])["soil_value"].transform(
    lambda x: x.fillna(x.median())
)

df["soil_value"].fillna(df["soil_value"].median(), inplace=True)

moisture_mask = df["unit"] == "%VWC"
df = df[~(moisture_mask & ((df["soil_value"] < 0) | (df["soil_value"] > 100)))]

df.drop(columns=["site_id", "id", "probe_id", "json_featuretype"], errors="ignore", inplace=True)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df.sort_values(["site_name", "local_time"], inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(CLEANED_PATH, index=False)

print(f"\nCleaned shape: {df.shape}")
print(f"Cleaned data saved to {CLEANED_PATH}")
print("\nSample:")
print(df.head())
