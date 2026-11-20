import pandas as pd
import folium

from pathlib import Path
from folium import plugins


# ==========================================
# FILE PATHS
# ==========================================

HISTORICAL_FILE = Path(
    "dataset/processed/cira_historical_track.csv"
)

LSTM_FILE = Path(
    "results/cira_lstm_prediction.csv"
)

CIRA_FORECAST_FILE = Path(
    "dataset/processed/cira_forecast_track.csv"
)

OUTPUT_FILE = Path(
    "results/cira_lstm_prediction_map.html"
)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    historical = pd.read_csv(HISTORICAL_FILE)

    lstm = pd.read_csv(LSTM_FILE)

    cira_forecast = pd.read_csv(
        CIRA_FORECAST_FILE
    )

    return historical, lstm, cira_forecast


# ==========================================
# CREATE MAP
# ==========================================

def create_map():

    print("\nCreating cyclone prediction map...")

    historical, lstm, cira_forecast = load_data()

    # Latest actual observation
    latest_lat = historical["LAT"].iloc[-1]

    latest_lon = historical["LON"].iloc[-1]

    # Initialize map
    cyclone_map = folium.Map(
        location=[latest_lat, latest_lon],
        zoom_start=5,
        tiles=None
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Street_Map/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Street Map",
        name="Street Map"
    ).add_to(cyclone_map)

    # ======================================
    # HISTORICAL TRACK
    # ======================================

    historical_points = list(
        zip(
            historical["LAT"],
            historical["LON"]
        )
    )

    folium.PolyLine(
        historical_points,
        color="blue",
        weight=4,
        opacity=0.8,
        tooltip="CIRA Historical Track"
    ).add_to(cyclone_map)

    for _, row in historical.iterrows():

        folium.CircleMarker(
            location=[
                row["LAT"],
                row["LON"]
            ],
            radius=4,
            color="blue",
            fill=True,
            popup=(
                f"Historical observation<br>"
                f"Time: {row['ISO_TIME']}<br>"
                f"Latitude: {row['LAT']}<br>"
                f"Longitude: {row['LON']}"
            )
        ).add_to(cyclone_map)

    # ======================================
    # LATEST POSITION
    # ======================================

    folium.Marker(
        location=[
            latest_lat,
            latest_lon
        ],
        popup=(
            f"Latest CIRA observation<br>"
            f"Latitude: {latest_lat}<br>"
            f"Longitude: {latest_lon}"
        ),
        tooltip="Latest CIRA Position",
        icon=folium.Icon(
            color="green",
            icon="flag"
        )
    ).add_to(cyclone_map)

    # ======================================
    # LSTM PREDICTION
    # ======================================

    lstm_points = [
        (latest_lat, latest_lon)
    ]

    lstm_points.extend(
        list(
            zip(
                lstm["LAT"],
                lstm["LON"]
            )
        )
    )

    folium.PolyLine(
        lstm_points,
        color="red",
        weight=4,
        opacity=0.9,
        dash_array="10, 8",
        tooltip="Experimental LSTM Prediction"
    ).add_to(cyclone_map)

    for _, row in lstm.iterrows():

        folium.CircleMarker(
            location=[
                row["LAT"],
                row["LON"]
            ],
            radius=5,
            color="red",
            fill=True,
            popup=(
                f"LSTM Prediction<br>"
                f"Hours Ahead: {row['HOURS_AHEAD']}<br>"
                f"Time: {row['ISO_TIME']}<br>"
                f"Latitude: {row['LAT']:.3f}<br>"
                f"Longitude: {row['LON']:.3f}"
            )
        ).add_to(cyclone_map)

    # ======================================
    # CIRA FORECAST
    # ======================================

    cira_points = list(
        zip(
            cira_forecast["LAT"],
            cira_forecast["LON"]
        )
    )

    folium.PolyLine(
        cira_points,
        color="purple",
        weight=4,
        opacity=0.8,
        tooltip="CIRA Forecast Track"
    ).add_to(cyclone_map)

    for _, row in cira_forecast.iterrows():

        folium.CircleMarker(
            location=[
                row["LAT"],
                row["LON"]
            ],
            radius=6,
            color="purple",
            fill=True,
            popup=(
                f"CIRA Forecast<br>"
                f"Forecast hour: {row['FORECAST_HOUR']}<br>"
                f"Latitude: {row['LAT']}<br>"
                f"Longitude: {row['LON']}"
            )
        ).add_to(cyclone_map)

    # ======================================
    # MAP LAYERS
    # ======================================

    folium.LayerControl().add_to(cyclone_map)

    plugins.Fullscreen().add_to(cyclone_map)

    # Fit map to all tracks
    all_points = (
        historical_points
        + lstm_points
        + cira_points
    )

    cyclone_map.fit_bounds(all_points)

    # Save map
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cyclone_map.save(OUTPUT_FILE)

    print("\nMAP GENERATED SUCCESSFULLY!")

    print("Historical points:", len(historical))

    print("LSTM predicted points:", len(lstm))

    print("CIRA forecast points:", len(cira_forecast))

    print("\nSaved to:", OUTPUT_FILE)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    create_map()