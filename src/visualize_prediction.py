import os
import pandas as pd
import folium

# ==========================================
# FILE PATHS
# ==========================================

HISTORICAL_FILE = "results/historical_cyclone_path.csv"
PREDICTED_FILE = "results/predicted_cyclone_path.csv"
OUTPUT_FILE = "results/cyclone_prediction_map.html"


# ==========================================
# LOAD DATA
# ==========================================

print("Loading cyclone prediction data...")

historical = pd.read_csv(HISTORICAL_FILE)
predicted = pd.read_csv(PREDICTED_FILE)


# ==========================================
# FIND LATITUDE AND LONGITUDE COLUMNS
# ==========================================

def find_column(data, possible_names):

    for name in possible_names:
        for column in data.columns:

            if column.upper() == name.upper():
                return column

    raise ValueError(
        f"Could not find required columns: {possible_names}\n"
        f"Available columns: {list(data.columns)}"
    )


hist_lat = find_column(historical, ["LAT", "LATITUDE"])
hist_lon = find_column(historical, ["LON", "LONGITUDE"])

pred_lat = find_column(predicted, ["LAT", "LATITUDE"])
pred_lon = find_column(predicted, ["LON", "LONGITUDE"])


# ==========================================
# CLEAN DATA
# ==========================================

historical[hist_lat] = pd.to_numeric(
    historical[hist_lat], errors="coerce"
)

historical[hist_lon] = pd.to_numeric(
    historical[hist_lon], errors="coerce"
)

predicted[pred_lat] = pd.to_numeric(
    predicted[pred_lat], errors="coerce"
)

predicted[pred_lon] = pd.to_numeric(
    predicted[pred_lon], errors="coerce"
)

# Remove missing coordinates
# No inplace=True, avoiding the Pandas deprecation warning

historical = historical.dropna(
    subset=[hist_lat, hist_lon]
)

predicted = predicted.dropna(
    subset=[pred_lat, pred_lon]
)

if historical.empty:
    raise ValueError("Historical cyclone data is empty.")

if predicted.empty:
    raise ValueError("Predicted cyclone data is empty.")


# ==========================================
# SORT HISTORICAL OBSERVATIONS
# ==========================================

time_column = next(
    (
        col for col in historical.columns
        if col.upper() in [
            "ISO_TIME",
            "DATETIME",
            "TIME",
            "DATE"
        ]
    ),
    None
)

if time_column:

    historical[time_column] = pd.to_datetime(
        historical[time_column],
        errors="coerce"
    )

    historical = historical.sort_values(
        time_column,
        na_position="last"
    )


# ==========================================
# EXTRACT COORDINATES
# ==========================================

historical_points = [
    [float(row[hist_lat]), float(row[hist_lon])]
    for _, row in historical.iterrows()
]

predicted_points = [
    [float(row[pred_lat]), float(row[pred_lon])]
    for _, row in predicted.iterrows()
]

# Last historical cyclone position

last_historical_point = historical_points[-1]

# Connect predicted path to historical path

full_predicted_route = (
    [last_historical_point] + predicted_points
)


# ==========================================
# CREATE MAP
# ==========================================

center_lat, center_lon = last_historical_point

cyclone_map = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles=None
)


# ==========================================
# ESRI WORLD STREET MAP
# ==========================================

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Tiles © Esri",
    name="Esri World Street Map",
    overlay=False,
    control=True
).add_to(cyclone_map)


# ==========================================
# HISTORICAL CYCLONE PATH
# ==========================================

folium.PolyLine(
    locations=historical_points,
    color="blue",
    weight=4,
    opacity=0.9,
    tooltip="Historical Cyclone Path"
).add_to(cyclone_map)


# ==========================================
# PREDICTED CYCLONE PATH
# ==========================================

folium.PolyLine(
    locations=full_predicted_route,
    color="red",
    weight=4,
    opacity=0.9,
    dash_array="10, 8",
    tooltip="Predicted Cyclone Path"
).add_to(cyclone_map)


# ==========================================
# LAST OBSERVED POSITION
# ==========================================

folium.Marker(
    location=last_historical_point,
    popup=folium.Popup(
        "<b>Last Observed Cyclone Position</b>",
        max_width=250
    ),
    tooltip="Last Observed Position",
    icon=folium.Icon(
        color="green",
        icon="info-sign"
    )
).add_to(cyclone_map)


# ==========================================
# PREDICTED POSITIONS
# ==========================================

forecast_column = next(
    (
        col for col in predicted.columns
        if col.upper() in [
            "FORECAST_HOUR",
            "HOUR",
            "TIME_STEP",
            "STEP"
        ]
    ),
    None
)

speed_column = next(
    (
        col for col in predicted.columns
        if col.upper() in [
            "SPEED",
            "STORM_SPEED",
            "MOVEMENT_SPEED"
        ]
    ),
    None
)


for position, (_, row) in enumerate(
    predicted.iterrows()
):

    latitude = float(row[pred_lat])
    longitude = float(row[pred_lon])

    # Forecast time

    if forecast_column:

        forecast_hour = row[forecast_column]

    else:

        forecast_hour = (position + 1) * 3

    # Popup information

    popup_text = (
        f"<b>Predicted Cyclone Position</b><br>"
        f"Forecast: {forecast_hour} hours<br>"
        f"Latitude: {latitude:.4f}<br>"
        f"Longitude: {longitude:.4f}"
    )

    # Movement speed

    if speed_column:

        speed = pd.to_numeric(
            row[speed_column],
            errors="coerce"
        )

        if pd.notna(speed):

            popup_text += (
                f"<br>Movement Speed: {speed:.2f} km/h"
            )

    # Add predicted position marker

    folium.CircleMarker(
        location=[latitude, longitude],
        radius=5,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.9,
        popup=folium.Popup(
            popup_text,
            max_width=250
        ),
        tooltip=f"Predicted position: {forecast_hour} hours"
    ).add_to(cyclone_map)


# ==========================================
# SRI LANKA MARKER
# ==========================================

folium.Marker(
    location=[7.8731, 80.7718],
    popup="Sri Lanka",
    tooltip="Sri Lanka",
    icon=folium.Icon(
        color="orange",
        icon="flag"
    )
).add_to(cyclone_map)


# ==========================================
# MAP LEGEND
# ==========================================

legend_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    width: 230px;
    background-color: white;
    border: 2px solid grey;
    z-index: 9999;
    padding: 12px;
    font-size: 14px;
    border-radius: 6px;
">

    <b>Cyclone Path Legend</b>
    <br><br>

    <span style="color:blue;">━━━━</span>
    Historical Path
    <br>

    <span style="color:red;">━━━━</span>
    Predicted Path
    <br>

    <span style="color:green;">●</span>
    Last Observed Position
    <br>

    <span style="color:orange;">●</span>
    Sri Lanka

</div>
"""

cyclone_map.get_root().html.add_child(
    folium.Element(legend_html)
)


# ==========================================
# LAYER CONTROL
# ==========================================

folium.LayerControl().add_to(cyclone_map)


# ==========================================
# FIT MAP TO CYCLONE PATH
# ==========================================

all_points = (
    historical_points + predicted_points
)

cyclone_map.fit_bounds(all_points)


# ==========================================
# SAVE MAP
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

cyclone_map.save(OUTPUT_FILE)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n================================")
print("CYCLONE MAP GENERATED SUCCESSFULLY")
print("================================")

print(
    f"Historical points: {len(historical_points)}"
)

print(
    f"Predicted points: {len(predicted_points)}"
)

print("\nMap saved to:")

print(os.path.abspath(OUTPUT_FILE))

print("\nOpen the HTML file in your browser.")