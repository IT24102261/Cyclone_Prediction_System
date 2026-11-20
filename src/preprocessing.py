import pandas as pd
import os

# Project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Dataset paths
input_file = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "ibtracs.NI.list.v04r01.csv"
)

output_file = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "cleaned_cyclones.csv"
)

print("Loading cyclone dataset...")

df = pd.read_csv(
    input_file,
    skiprows=[1],
    low_memory=False
)

# Select required columns
columns = [
    "SID",
    "SEASON",
    "NAME",
    "ISO_TIME",
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
    "USA_WIND",
    "USA_PRES",
    "STORM_SPEED",
    "STORM_DIR"
]

df = df[columns].copy()

# Convert numerical columns
numeric_columns = [
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
    "USA_WIND",
    "USA_PRES",
    "STORM_SPEED",
    "STORM_DIR"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Convert date
df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce"
)

# Remove invalid coordinates and dates
df = df.dropna(
    subset=["SID", "ISO_TIME", "LAT", "LON"]
)

df = df[
    df["LAT"].between(-90, 90) &
    df["LON"].between(-180, 180)
]

# Sort cyclone observations
df = df.sort_values(
    ["SID", "ISO_TIME"]
)

# Save cleaned dataset
df.to_csv(
    output_file,
    index=False
)

print("\nPreprocessing completed!")

print("Total observations:", len(df))

print("Unique cyclones:", df["SID"].nunique())

print("\nMissing values:")
print(df.isna().sum())

print("\nCleaned dataset saved to:")
print(output_file)