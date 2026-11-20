import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    "dataset/raw/ibtracs.NI.list.v04r01.csv",
    skiprows=[1],
    low_memory=False
)

# Convert important columns
df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
df["USA_WIND"] = pd.to_numeric(df["USA_WIND"], errors="coerce")
df["USA_PRES"] = pd.to_numeric(df["USA_PRES"], errors="coerce")

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce"
)

# 1. Total cyclone observations
print("Total observations:", len(df))

# 2. Unique cyclones
print("Unique cyclones:", df["SID"].nunique())

# 3. Available years
print("First year:", df["SEASON"].min())
print("Last year:", df["SEASON"].max())

# 4. Missing values
print("\nMissing values:")
print(df[["LAT", "LON", "USA_WIND", "USA_PRES"]].isna().sum())

# 5. Wind speed statistics
print("\nWind speed statistics:")
print(df["USA_WIND"].describe())

# 6. Geographical distribution
plt.figure(figsize=(10, 6))

plt.scatter(
    df["LON"],
    df["LAT"],
    s=2,
    alpha=0.3
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Historical Cyclone Locations - North Indian Ocean")

plt.grid()
plt.savefig("results/cyclone_locations.png")
plt.show()

print("\nEDA completed successfully!")