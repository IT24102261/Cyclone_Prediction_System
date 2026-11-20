import pandas as pd
import numpy as np

df = pd.read_csv(
    "dataset/processed/cyclone_features.csv"
)

print("INITIAL DATASET")
print("Total rows:", len(df))

# Convert columns
columns = [
    "LAT", "LON", "TIME_DIFF",
    "MOVEMENT_SPEED", "USA_WIND",
    "USA_PRES", "WMO_WIND", "WMO_PRES"
]

for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Check time intervals
print("\nTIME DIFFERENCE STATISTICS")
print(df["TIME_DIFF"].describe())

print("\nTIME INTERVAL DISTRIBUTION")
print(df["TIME_DIFF"].value_counts().head(20))

# Check time filter
time_filter = df["TIME_DIFF"].between(5, 7)

print("\nOBSERVATIONS WITH 5-7 HOUR INTERVALS:")
print(time_filter.sum())

# Check movement speed
speed_filter = df["MOVEMENT_SPEED"].between(0, 100)

print("\nOBSERVATIONS WITH VALID MOVEMENT SPEED:")
print(speed_filter.sum())

# Check wind and pressure
print("\nWIND AND PRESSURE MISSING VALUES")

print("USA Wind:", df["USA_WIND"].isna().sum())
print("WMO Wind:", df["WMO_WIND"].isna().sum())

print("USA Pressure:", df["USA_PRES"].isna().sum())
print("WMO Pressure:", df["WMO_PRES"].isna().sum())

# Check combined weather availability
wind = df["USA_WIND"].fillna(df["WMO_WIND"])
pressure = df["USA_PRES"].fillna(df["WMO_PRES"])

print("\nAVAILABLE WEATHER OBSERVATIONS")
print("Wind available:", wind.notna().sum())
print("Pressure available:", pressure.notna().sum())

print("\nDATA DIAGNOSTICS COMPLETED")