import pandas as pd
import os

# Get project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
file_path = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "ibtracs.NI.list.v04r01.csv"
)

# Check whether dataset exists
if not os.path.exists(file_path):
    print("Dataset not found!")
else:

    print("Loading cyclone dataset...")

    # Load dataset (skip the units row)
    data = pd.read_csv(
        file_path,
        skiprows=[1],
        low_memory=False
    )

    # Display first five records
    print("\nFirst five cyclone records:")
    print(data.head())

    # Display dataset information
    print("\nDataset Information:")
    print(data.info())

    # Display dataset dimensions
    print("\nDataset Shape:")
    print(data.shape)

    # Display column names
    print("\nAvailable Columns:")
    print(data.columns.tolist())

    print("\nDataset loaded successfully!")