import requests
import pandas as pd

from bs4 import BeautifulSoup
from io import StringIO
from pathlib import Path


# =====================================================
# CONFIGURATION
# =====================================================

URL = (
    "https://tc.cira.colostate.edu/tc_realtime/"
    "storm.asp?storm_identifier=io052026"
)

OUTPUT_DIR = Path("dataset/processed")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# DOWNLOAD CIRA WEBPAGE
# =====================================================

def download_cira_page():

    print("\nConnecting to CIRA...")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        URL,
        headers=headers,
        timeout=60
    )

    response.raise_for_status()

    print("CIRA connection successful.")

    return response.text


# =====================================================
# EXTRACT TABLE
# =====================================================

def extract_table(table, expected_columns):

    table = table.copy()

    # First row contains column names
    table.columns = expected_columns

    # Remove header row from data
    table = table.iloc[1:].copy()

    return table.reset_index(drop=True)


# =====================================================
# PROCESS HISTORICAL TRACK
# =====================================================

def process_historical_track(table):

    historical = extract_table(
        table,
        [
            "ISO_TIME",
            "LAT",
            "LON",
            "INTENSITY"
        ]
    )

    historical["ISO_TIME"] = pd.to_datetime(
        historical["ISO_TIME"],
        errors="coerce",
        utc=True
    )

    for column in [
        "LAT",
        "LON",
        "INTENSITY"
    ]:

        historical[column] = pd.to_numeric(
            historical[column],
            errors="coerce"
        )

    historical = historical.dropna(
        subset=[
            "ISO_TIME",
            "LAT",
            "LON"
        ]
    )

    historical = historical.sort_values(
        "ISO_TIME"
    ).reset_index(drop=True)

    historical["TIME_DIFF"] = (
        historical["ISO_TIME"]
        .diff()
        .dt.total_seconds() / 3600
    )

    return historical


# =====================================================
# PROCESS CIRA FORECAST
# =====================================================

def process_forecast(table):

    forecast = extract_table(
        table,
        [
            "FORECAST_HOUR",
            "LAT",
            "LON",
            "INTENSITY"
        ]
    )

    for column in [
        "FORECAST_HOUR",
        "LAT",
        "LON",
        "INTENSITY"
    ]:

        forecast[column] = pd.to_numeric(
            forecast[column],
            errors="coerce"
        )

    forecast = forecast.dropna(
        subset=[
            "FORECAST_HOUR",
            "LAT",
            "LON"
        ]
    )

    # Remove invalid zero-coordinate forecasts
    forecast = forecast[
        (forecast["LAT"] != 0)
        &
        (forecast["LON"] != 0)
    ]

    return forecast.reset_index(drop=True)


# =====================================================
# MAIN FUNCTION
# =====================================================

def get_cira_cyclone_data():

    html = download_cira_page()

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    tables = pd.read_html(
        StringIO(str(soup)),
        header=None
    )

    if len(tables) < 2:

        raise ValueError(
            "Expected two CIRA tables, but they were not found."
        )

    # Table 1: Forecast positions
    forecast = process_forecast(
        tables[0]
    )

    # Table 2: Historical observations
    historical = process_historical_track(
        tables[1]
    )

    # Save historical track
    historical_path = (
        OUTPUT_DIR / "cira_historical_track.csv"
    )

    historical.to_csv(
        historical_path,
        index=False
    )

    # Save forecast
    forecast_path = (
        OUTPUT_DIR / "cira_forecast_track.csv"
    )

    forecast.to_csv(
        forecast_path,
        index=False
    )

    return historical, forecast


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    print("=" * 50)
    print("CIRA CYCLONE DATA EXTRACTION")
    print("=" * 50)

    try:

        historical, forecast = (
            get_cira_cyclone_data()
        )

        print("\nHISTORICAL TRACK")
        print("-" * 50)

        print(
            historical.to_string(index=False)
        )

        print("\nFORECAST TRACK")
        print("-" * 50)

        print(
            forecast.to_string(index=False)
        )

        print("\nSUMMARY")
        print("-" * 50)

        print(
            "Historical observations:",
            len(historical)
        )

        print(
            "Forecast positions:",
            len(forecast)
        )

        print(
            "Latest observation:",
            historical["ISO_TIME"].max()
        )

        print("\nFiles saved successfully.")

        print(
            "dataset/processed/cira_historical_track.csv"
        )

        print(
            "dataset/processed/cira_forecast_track.csv"
        )

    except Exception as error:

        print("\nERROR:", error)