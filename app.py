import sys
import subprocess
import html
from pathlib import Path

import streamlit as st
import pandas as pd
import folium
import streamlit.components.v1 as components


# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "dataset" / "processed"
RESULTS_DIR = BASE_DIR / "results"

sys.path.insert(0, str(SRC_DIR))

from prediction_engine import (
    load_cyclone_data,
    get_eligible_cyclones,
    load_prediction_model,
    predict_cyclone,
)

ACTIVE_SUMMARY = DATA_DIR / "active_cyclone_summary.csv"
PREPARATION_SUMMARY = (
    DATA_DIR / "active_lstm_inputs" / "preparation_summary.csv"
)


# ==========================================================
# 2. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="CycloneWatch | Global Cyclone Monitoring",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# 3. CUSTOM CSS AND PUBLIC-FACING HEADER
# ==========================================================

st.markdown(
    """
    <style>
    /* Overall application palette */
    :root {
        --cw-bg: #0B1220;
        --cw-panel: #172338;
        --cw-panel-soft: #1D2D45;
        --cw-blue: #2196F3;
        --cw-teal: #00D4AA;
        --cw-text: #EAF2FF;
        --cw-muted: #A9B8CC;
    }
    .stApp { background: var(--cw-bg); color: var(--cw-text); }
    [data-testid="stHeader"] { background: rgba(11,18,32,0.92); }
    [data-testid="stSidebar"] {
        background: #101B2D; border-right: 1px solid #263750;
    }
    [data-testid="stSidebar"] * { color: var(--cw-text); }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1600px; }
    h1, h2, h3 { color: #F4F8FF !important; letter-spacing: -0.02em; }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--cw-muted); }

    /* Hero / brand header */
    .cw-hero {
        padding: 1.5rem 1.7rem; border-radius: 20px; margin-bottom: 1.2rem;
        background: linear-gradient(115deg, #142943 0%, #102039 55%, #123B50 100%);
        border: 1px solid #294766; box-shadow: 0 12px 34px rgba(0,0,0,.22);
    }
    .cw-brand-row { display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
    .cw-icon {
        width:54px; height:54px; display:flex; align-items:center; justify-content:center;
        border-radius:16px; background:rgba(33,150,243,.17); font-size:30px;
        border:1px solid rgba(91,180,255,.35);
    }
    .cw-title { color:#F5FAFF; font-size:clamp(27px, 3vw, 39px);
        font-weight:800; line-height:1.12; margin:0; }
    .cw-subtitle { color:#B9CBE0; font-size:15px; margin:8px 0 0 0; }
    .cw-badge {
        display:inline-block; margin-top:15px; padding:6px 11px; border-radius:999px;
        color:#8EF0D8; background:rgba(0,212,170,.10);
        border:1px solid rgba(0,212,170,.32); font-size:12px; font-weight:700;
        letter-spacing:.04em; text-transform:uppercase;
    }

    /* Panels, metrics, controls */
    [data-testid="stMetric"] {
        background:linear-gradient(145deg, #192A42, #142238);
        padding:16px 18px; border:1px solid #2B405C; border-radius:15px;
    }
    [data-testid="stMetricLabel"] { color:#AFC2D9 !important; }
    [data-testid="stMetricValue"] { color:#F4F8FF !important; }
    [data-testid="stMetricDelta"] { color:var(--cw-teal) !important; }
    [data-testid="stTabs"] button { color:#B9CBE0; font-weight:650; }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color:#62B6FF; border-bottom-color:#2196F3;
    }
    .stButton > button {
        border-radius:10px; border:1px solid #278FE0; font-weight:650;
        background:#176DB2; color:white;
    }
    .stButton > button:hover { background:#2188D5; border-color:#64B9FF; color:white; }
    div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
        background:#142238; border-color:#344B68; color:#F4F8FF;
    }
    [data-testid="stAlert"] { border-radius:12px; }
    hr { border-color:#2A3D56; }

    /* Tablet layout */
    @media (max-width: 992px) {
        .block-container {
            padding-left: 1.1rem;
            padding-right: 1.1rem;
            max-width: 100%;
        }
        [data-testid="stIFrame"] iframe {
            width: 100% !important;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            max-width: 100%;
            overflow-x: auto;
        }
    }

    /* Phone layout */
    @media (max-width: 700px) {
        .block-container {
            padding: .75rem .75rem 1.5rem .75rem;
        }
        .cw-hero {
            padding: 1rem;
            border-radius: 14px;
            margin-bottom: .85rem;
        }
        .cw-brand-row {
            align-items: flex-start;
            gap: 10px;
        }
        .cw-icon {
            width: 42px;
            height: 42px;
            min-width: 42px;
            font-size: 23px;
            border-radius: 12px;
        }
        .cw-title {
            font-size: 27px;
            line-height: 1.1;
        }
        .cw-subtitle {
            font-size: 13px;
            line-height: 1.45;
        }
        .cw-badge {
            font-size: 10px;
            line-height: 1.4;
            padding: 5px 8px;
        }
        h1 { font-size: 1.65rem !important; }
        h2 { font-size: 1.35rem !important; }
        h3 { font-size: 1.12rem !important; }

        /* Make Streamlit columns wrap instead of squeezing cards */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: .65rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: min(100%, 260px) !important;
            flex: 1 1 100% !important;
            width: 100% !important;
        }

        /* Touch-friendly controls */
        .stButton > button,
        [data-testid="stDownloadButton"] button {
            width: 100%;
            min-height: 44px;
            border-radius: 11px;
        }
        div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input {
            min-height: 42px;
        }
        [data-testid="stMultiSelect"] {
            width: 100%;
        }

        /* Tabs can scroll horizontally on narrow screens */
        [data-testid="stTabs"] [role="tablist"] {
            gap: .25rem;
            overflow-x: auto;
            flex-wrap: nowrap;
            scrollbar-width: thin;
        }
        [data-testid="stTabs"] button[role="tab"] {
            flex: 0 0 auto;
            white-space: nowrap;
            font-size: 13px;
            padding: .55rem .7rem;
        }

        /* Keep map and tables within the viewport */
        [data-testid="stIFrame"] {
            width: 100% !important;
            max-width: 100vw !important;
        }
        [data-testid="stIFrame"] iframe {
            width: 100% !important;
            height: 58vh !important;
            min-height: 360px !important;
            max-height: 560px !important;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            width: 100%;
            max-width: 100%;
            overflow-x: auto;
        }
        [data-testid="stMetric"] {
            padding: 12px 13px;
            border-radius: 12px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.45rem;
        }
        [data-testid="stAlert"] {
            padding: .75rem .85rem;
        }
        [data-testid="stSidebar"] {
            min-width: 0;
        }
        footer { visibility: hidden; }
    }

    /* Very narrow phones */
    @media (max-width: 380px) {
        .block-container {
            padding-left: .55rem;
            padding-right: .55rem;
        }
        .cw-title { font-size: 24px; }
        .cw-subtitle { font-size: 12px; }
        [data-testid="stMetricValue"] { font-size: 1.25rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="cw-hero">
      <div class="cw-brand-row">
        <div class="cw-icon">🌀</div>
        <div>
          <h1 class="cw-title">CycloneWatch</h1>
          <p class="cw-subtitle">Global Cyclone Monitoring &amp; Experimental Movement Prediction</p>
        </div>
      </div>
      <span class="cw-badge">Global monitoring • LSTM research forecasts</span>
    </section>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Forecasts shown here are experimental research outputs, not official cyclone warnings. "
    "For safety decisions, follow alerts from your national meteorological authority.",
    icon="⚠️",
)


# ==========================================================
# 4. LOAD HISTORICAL DATA AND MODEL
# ==========================================================

@st.cache_data(show_spinner=False)
def load_dataset():
    return load_cyclone_data()


@st.cache_resource(show_spinner=False)
def load_trained_model():
    return load_prediction_model()


try:
    with st.spinner("Loading historical cyclone dataset..."):
        cyclone_data = load_dataset()

    with st.spinner("Finding eligible historical cyclones..."):
        eligible_cyclones = get_eligible_cyclones(cyclone_data)

    with st.spinner("Loading trained LSTM model..."):
        model, scaler = load_trained_model()

except Exception as error:
    st.error("Unable to initialize the historical prediction system.")
    st.exception(error)
    st.stop()


# ==========================================================
# 5. GENERAL HELPER FUNCTIONS
# ==========================================================

def run_script(script_name, *arguments, timeout=240):
    """Run a project Python script and return its console output."""
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = [sys.executable, str(script_path), *map(str, arguments)]

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    return result.stdout


def load_csv_if_exists(path):
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.error(f"Unable to read {path.name}: {error}")
        return None


def first_existing_column(dataframe, candidates):
    """Return the first matching column name."""
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def clean_track(dataframe):
    """Normalize and validate a cyclone track dataframe."""
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    data = dataframe.copy()

    required = ["ISO_TIME", "LAT", "LON"]
    if not all(column in data.columns for column in required):
        return pd.DataFrame()

    data["ISO_TIME"] = pd.to_datetime(
        data["ISO_TIME"], utc=True, errors="coerce"
    )
    data["LAT"] = pd.to_numeric(data["LAT"], errors="coerce")
    data["LON"] = pd.to_numeric(data["LON"], errors="coerce")

    data = data.dropna(subset=required).sort_values("ISO_TIME")

    # Reject impossible coordinate values.
    data = data[
        data["LAT"].between(-90, 90)
        & data["LON"].between(-180, 180)
    ]

    return data.reset_index(drop=True)


def get_forecast_hour(row, index):
    if pd.notna(row.get("HOURS_AHEAD", None)):
        return float(row["HOURS_AHEAD"])
    if pd.notna(row.get("FORECAST_HOURS", None)):
        return float(row["FORECAST_HOURS"])
    if pd.notna(row.get("STEP", None)):
        return float(row["STEP"]) * 3
    return (index + 1) * 3


def create_track_map(historical, lstm_prediction=None):
    """Create a map with historical observations and LSTM forecast."""
    historical = clean_track(historical)

    if historical.empty:
        return None

    latest = historical.iloc[-1]
    latest_lat = float(latest["LAT"])
    latest_lon = float(latest["LON"])

    map_object = folium.Map(
        location=[latest_lat, latest_lon],
        zoom_start=5,
        tiles=None,
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Street Map",
        name="Street Map",
    ).add_to(map_object)

    historical_points = historical[["LAT", "LON"]].values.tolist()

    folium.PolyLine(
        historical_points,
        color="blue",
        weight=4,
        opacity=0.9,
        tooltip="Historical cyclone track",
    ).add_to(map_object)

    folium.Marker(
        [latest_lat, latest_lon],
        tooltip="Latest observed position",
        popup=(
            f"Latest observed position<br>"
            f"Latitude: {latest_lat:.3f}<br>"
            f"Longitude: {latest_lon:.3f}<br>"
            f"Time (UTC): {latest['ISO_TIME']}"
        ),
        icon=folium.Icon(color="green", icon="info-sign"),
    ).add_to(map_object)

    all_points = list(historical_points)

    if lstm_prediction is not None and not lstm_prediction.empty:
        prediction = lstm_prediction.copy()
        prediction["LAT"] = pd.to_numeric(
            prediction["LAT"], errors="coerce"
        )
        prediction["LON"] = pd.to_numeric(
            prediction["LON"], errors="coerce"
        )
        prediction = prediction.dropna(subset=["LAT", "LON"])

        if not prediction.empty:
            predicted_points = prediction[["LAT", "LON"]].values.tolist()
            full_path = [[latest_lat, latest_lon]] + predicted_points

            folium.PolyLine(
                full_path,
                color="red",
                weight=4,
                opacity=0.9,
                dash_array="10, 8",
                tooltip="Experimental LSTM prediction",
            ).add_to(map_object)

            for index, (_, row) in enumerate(prediction.iterrows()):
                forecast_hour = get_forecast_hour(row, index)
                forecast_time = row.get("ISO_TIME", "Not available")

                popup = (
                    "<b>Experimental LSTM prediction</b><br>"
                    f"Forecast: +{forecast_hour:g} hours<br>"
                    f"Time: {forecast_time}<br>"
                    f"Latitude: {row['LAT']:.3f}<br>"
                    f"Longitude: {row['LON']:.3f}"
                )

                folium.CircleMarker(
                    [float(row["LAT"]), float(row["LON"])],
                    radius=5,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.8,
                    popup=folium.Popup(popup, max_width=300),
                ).add_to(map_object)

            all_points.extend(predicted_points)

    folium.LayerControl().add_to(map_object)
    if all_points:
        map_object.fit_bounds(all_points)

    return map_object


def display_map(map_object, height=650):
    if map_object is None:
        st.warning("A map cannot be created because valid track coordinates are unavailable.")
        return

    components.html(
        map_object.get_root().render(),
        height=height,
        scrolling=False,
    )


def create_global_active_map(summary, sid_col, name_col=None, basin_col=None, highlighted_sid=None):
    """
    Display every discovered cyclone's historical track and, when available,
    its saved LSTM forecast on one global map.

    Visual convention:
      Blue solid line  = historical track
      Red dashed line  = experimental LSTM forecast
      Green marker     = latest observed position
    """
    if summary is None or summary.empty or sid_col not in summary.columns:
        return None, 0, 0

    global_map = folium.Map(
        location=[5, 0],
        zoom_start=2,
        min_zoom=2,
        tiles=None,
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Street Map",
        name="Esri World Street Map",
    ).add_to(global_map)

    historical_layer = folium.FeatureGroup(
        name="Historical cyclone tracks",
        show=True,
    )
    prediction_layer = folium.FeatureGroup(
        name="Experimental LSTM predicted tracks",
        show=True,
    )
    latest_layer = folium.FeatureGroup(
        name="Latest observed positions",
        show=True,
    )

    all_coordinates = []
    plotted_storms = 0
    predicted_storms = 0

    for _, storm in summary.iterrows():
        sid = str(storm[sid_col]).strip()
        if not sid or sid.lower() == "nan":
            continue

        storm_name = (
            str(storm[name_col]).strip()
            if name_col and pd.notna(storm.get(name_col))
            else "Unnamed / Invest"
        )
        basin_name = (
            str(storm[basin_col]).strip()
            if basin_col and pd.notna(storm.get(basin_col))
            else "Basin unavailable"
        )

        track_file = DATA_DIR / f"{sid}_track.csv"
        track = clean_track(load_csv_if_exists(track_file))
        if track.empty:
            continue

        latest = track.iloc[-1]
        latest_lat = float(latest["LAT"])
        latest_lon = float(latest["LON"])
        historical_points = track[["LAT", "LON"]].values.tolist()

        popup_details = (
            f"<b>{storm_name}</b><br>"
            f"Storm ID: {sid}<br>"
            f"Basin: {basin_name}<br>"
            f"Historical observations: {len(track)}<br>"
            f"Latest observed position: {latest_lat:.2f}°, {latest_lon:.2f}°<br>"
            f"Latest observation (UTC): {latest['ISO_TIME']}"
        )

        is_highlighted = highlighted_sid is not None and sid == str(highlighted_sid)
        folium.PolyLine(
            historical_points,
            color="#00D4AA" if is_highlighted else "#2684FF",
            weight=6 if is_highlighted else 3,
            opacity=1.0 if is_highlighted else 0.82,
            tooltip=f"{storm_name} ({sid}) — historical track",
            popup=folium.Popup(popup_details, max_width=360),
        ).add_to(historical_layer)

        folium.Marker(
            [latest_lat, latest_lon],
            tooltip=f"{storm_name} ({sid}) — latest observed position",
            popup=folium.Popup(popup_details, max_width=360),
            icon=folium.Icon(
                color="cadetblue" if is_highlighted else "green",
                icon="star" if is_highlighted else "map-marker",
                prefix="fa",
            ),
        ).add_to(latest_layer)

        all_coordinates.extend(historical_points)
        plotted_storms += 1

        # A prediction is included only when its output exists and is not
        # older than the prepared input used to generate it.
        input_file = (
            DATA_DIR
            / "active_lstm_inputs"
            / f"{sid}_3hour_input.csv"
        )
        prediction_file = RESULTS_DIR / f"{sid}_lstm_prediction.csv"

        prediction_is_fresh = (
            input_file.exists()
            and prediction_file.exists()
            and prediction_file.stat().st_mtime >= input_file.stat().st_mtime
        )

        if prediction_is_fresh:
            prediction = load_csv_if_exists(prediction_file)

            if prediction is not None and not prediction.empty:
                if {"LAT", "LON"}.issubset(prediction.columns):
                    prediction["LAT"] = pd.to_numeric(
                        prediction["LAT"], errors="coerce"
                    )
                    prediction["LON"] = pd.to_numeric(
                        prediction["LON"], errors="coerce"
                    )
                    prediction = prediction.dropna(subset=["LAT", "LON"])

                    if not prediction.empty:
                        predicted_points = (
                            prediction[["LAT", "LON"]].values.tolist()
                        )
                        forecast_path = (
                            [[latest_lat, latest_lon]] + predicted_points
                        )

                        folium.PolyLine(
                            forecast_path,
                            color="#FF4B5C",
                            weight=6 if is_highlighted else 4,
                            opacity=1.0 if is_highlighted else 0.92,
                            dash_array="10, 7",
                            tooltip=(
                                f"{storm_name} ({sid}) — experimental "
                                "LSTM predicted path"
                            ),
                            popup=folium.Popup(
                                f"<b>{storm_name} — Experimental LSTM forecast</b><br>"
                                f"Storm ID: {sid}<br>"
                                f"Forecast positions: {len(prediction)}<br>"
                                f"Forecast horizon: up to 24 hours<br>"
                                f"Forecast starts from latest observed position: "
                                f"{latest['ISO_TIME']}<br>"
                                "This is an experimental model output, "
                                "not an official warning.",
                                max_width=360,
                            ),
                        ).add_to(prediction_layer)

                        for point_index, (_, forecast_row) in enumerate(
                            prediction.iterrows()
                        ):
                            forecast_hour = get_forecast_hour(
                                forecast_row, point_index
                            )
                            forecast_time = forecast_row.get(
                                "ISO_TIME", "Not available"
                            )

                            folium.CircleMarker(
                                [
                                    float(forecast_row["LAT"]),
                                    float(forecast_row["LON"]),
                                ],
                                radius=4,
                                color="red",
                                fill=True,
                                fill_color="red",
                                fill_opacity=0.9,
                                tooltip=(
                                    f"{storm_name} +{forecast_hour:g}h "
                                    "experimental prediction"
                                ),
                                popup=folium.Popup(
                                    f"<b>{storm_name}</b><br>"
                                    f"Forecast: +{forecast_hour:g} hours<br>"
                                    f"Time (UTC): {forecast_time}<br>"
                                    f"Latitude: {forecast_row['LAT']:.3f}<br>"
                                    f"Longitude: {forecast_row['LON']:.3f}",
                                    max_width=320,
                                ),
                            ).add_to(prediction_layer)

                        all_coordinates.extend(predicted_points)
                        predicted_storms += 1

    historical_layer.add_to(global_map)
    prediction_layer.add_to(global_map)
    latest_layer.add_to(global_map)

    folium.LayerControl(collapsed=False).add_to(global_map)

    if all_coordinates:
        global_map.fit_bounds(all_coordinates)

    # Add a compact visual legend.
    legend_html = """
    <div style="
        position: fixed; bottom: 25px; left: 25px; z-index: 9999;
        background: white; padding: 10px 13px; border: 1px solid #888;
        border-radius: 5px; font-size: 13px; line-height: 1.7;">
        <b>Track Legend</b><br>
        <span style="color:#2684FF;">━━</span> Historical track<br>
        <span style="color:#FF4B5C;">┄┄</span> Experimental LSTM prediction<br>
        <span style="color:green;">●</span> Latest observed position<br>
        <span style="color:#00A98F;">★</span> Selected cyclone
    </div>
    """
    global_map.get_root().html.add_child(folium.Element(legend_html))

    return global_map, plotted_storms, predicted_storms


def refresh_all_active_cyclones():
    """Refresh CIRA tracks, prepare inputs, and forecast every eligible storm."""
    logs = []

    logs.append("=== Discovering active cyclones and downloading tracks ===")
    logs.append(run_script("inspect_active_cyclone_tracks.py", timeout=300))

    logs.append("=== Preparing 3-hour LSTM inputs ===")
    logs.append(run_script("prepare_active_cyclone_inputs.py", timeout=300))

    prep = load_csv_if_exists(PREPARATION_SUMMARY)
    if prep is None or prep.empty:
        logs.append("No preparation summary was produced.")
        return logs, 0, 0

    sid_column = first_existing_column(prep, ["SID", "STORM_ID", "storm_id"])
    eligible_column = first_existing_column(
        prep, ["LSTM_ELIGIBLE", "ELIGIBLE", "IS_ELIGIBLE"]
    )

    if sid_column is None or eligible_column is None:
        logs.append(
            "Preparation summary is missing SID or LSTM_ELIGIBLE. "
            "Check preparation_summary.csv."
        )
        return logs, 0, 0

    eligible_sids = []
    for _, row in prep.iterrows():
        value = str(row.get(eligible_column, "")).strip().lower()
        if value in ["true", "1", "yes", "y"]:
            sid = str(row.get(sid_column, "")).strip()
            if sid and sid.lower() != "nan":
                eligible_sids.append(sid)

    eligible_sids = list(dict.fromkeys(eligible_sids))
    logs.append(
        f"=== Generating forecasts for {len(eligible_sids)} eligible cyclone(s) ==="
    )

    successful = 0
    failed = 0

    for sid in eligible_sids:
        try:
            output = run_script(
                "predict_active_cyclone.py",
                sid,
                timeout=300,
            )
            logs.append(f"--- {sid}: prediction completed ---\n{output}")
            successful += 1
        except Exception as error:
            logs.append(f"--- {sid}: prediction failed ---\n{error}")
            failed += 1

    return logs, successful, failed


@st.fragment(run_every="15m")
def automatic_refresh_controls():
    """Refresh on first load and every 15 minutes while the app is open."""
    now = pd.Timestamp.now(tz="UTC")
    previous = st.session_state.get("last_active_refresh_utc")

    due = previous is None
    if previous is not None:
        previous_time = pd.to_datetime(previous, utc=True, errors="coerce")
        due = (
            pd.isna(previous_time)
            or (now - previous_time) >= pd.Timedelta(minutes=15)
        )

    manual_refresh = st.button(
        "🔄 Refresh Active Cyclones Now",
        type="primary",
        key="refresh_active",
    )

    if due or manual_refresh:
        reason = "scheduled 15-minute update" if due else "manual update"
        try:
            with st.spinner(
                f"Running {reason}: discovering cyclones, updating tracks "
                "and generating eligible forecasts..."
            ):
                logs, successful, failed = refresh_all_active_cyclones()

            st.session_state["last_active_refresh_utc"] = (
                pd.Timestamp.now(tz="UTC").isoformat()
            )
            st.session_state["active_refresh_logs"] = logs
            st.session_state["active_refresh_successful"] = successful
            st.session_state["active_refresh_failed"] = failed

            st.rerun(scope="app")

        except Exception as error:
            st.session_state["last_active_refresh_utc"] = (
                pd.Timestamp.now(tz="UTC").isoformat()
            )
            st.error("Automatic active-cyclone refresh failed.")
            st.exception(error)

    last = st.session_state.get("last_active_refresh_utc")
    if last:
        last_time = pd.to_datetime(last, utc=True, errors="coerce")
        if pd.notna(last_time):
            next_time = last_time + pd.Timedelta(minutes=15)
            st.caption(
                f"Last update: {last_time.strftime('%Y-%m-%d %H:%M:%S UTC')} "
                f"| Next check: approximately {next_time.strftime('%H:%M:%S UTC')}"
            )

    if "active_refresh_successful" in st.session_state:
        st.caption(
            "Latest refresh predictions: "
            f"{st.session_state['active_refresh_successful']} successful, "
            f"{st.session_state.get('active_refresh_failed', 0)} failed."
        )

    with st.expander("View latest automatic refresh log"):
        for item in st.session_state.get("active_refresh_logs", []):
            st.text(item)


# ==========================================================
# 6. DASHBOARD TABS
# ==========================================================

historical_tab, active_tab = st.tabs(
    ["📊 Historical Cyclone Prediction", "🌪️ Active Cyclone Monitoring"]
)


# ==========================================================
# TAB 1 - HISTORICAL CYCLONE PREDICTION
# ==========================================================

with historical_tab:
    st.header("Historical Cyclone Prediction")
    st.write(
        "Select a historical cyclone and generate an experimental "
        "24-hour LSTM prediction."
    )

    if eligible_cyclones.empty:
        st.error("No eligible historical cyclones found.")
    else:
        include_unnamed = st.checkbox(
            "Include unnamed cyclones",
            value=False,
            key="include_unnamed",
        )

        if include_unnamed:
            available_cyclones = eligible_cyclones.copy()
        else:
            available_cyclones = eligible_cyclones[
                eligible_cyclones["NAME"].notna()
                & (
                    eligible_cyclones["NAME"]
                    .astype(str)
                    .str.upper()
                    != "UNNAMED"
                )
            ].copy()

        available_cyclones = available_cyclones.sort_values(
            "LAST_OBSERVATION", ascending=False
        )

        if available_cyclones.empty:
            st.warning("No cyclones available.")
        else:
            cyclone_options = {}

            for _, row in available_cyclones.iterrows():
                name = str(row["NAME"]).strip()
                if not name or name.lower() == "nan":
                    name = "UNNAMED"

                date = pd.to_datetime(
                    row["LAST_OBSERVATION"], errors="coerce"
                )
                date_label = (
                    date.strftime("%Y-%m-%d")
                    if pd.notna(date)
                    else "Date unavailable"
                )

                label = f"{name} | {row['SEASON']} | {date_label}"
                cyclone_options[label] = row["SID"]

            selected_label = st.selectbox(
                "Select historical cyclone",
                list(cyclone_options.keys()),
            )
            selected_sid = cyclone_options[selected_label]

            try:
                with st.spinner("Generating historical LSTM prediction..."):
                    historical, predicted, metadata = predict_cyclone(
                        selected_sid,
                        cyclone_data,
                        model,
                        scaler,
                    )
            except Exception as error:
                st.error("Historical prediction failed.")
                st.exception(error)
                historical, predicted, metadata = None, None, {}

            if (
                historical is None
                or historical.empty
                or predicted is None
                or predicted.empty
            ):
                st.error("Unable to generate a historical cyclone prediction.")
            else:
                historical = clean_track(historical)
                predicted = predicted.copy()

                if "ISO_TIME" in predicted.columns:
                    predicted["ISO_TIME"] = pd.to_datetime(
                        predicted["ISO_TIME"], utc=True, errors="coerce"
                    )

                predicted = predicted.dropna(
                    subset=["LAT", "LON"]
                ).sort_values(
                    "STEP" if "STEP" in predicted.columns else predicted.index.name
                    if predicted.index.name in predicted.columns
                    else predicted.columns[0]
                )

                name = metadata.get(
                    "name", selected_label.split("|")[0].strip()
                )
                season = metadata.get("season", "")
                latest_time = historical["ISO_TIME"].max()

                st.subheader(f"🌪️ {name} ({season})")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Historical Observations", len(historical))
                col2.metric("Predicted Positions", len(predicted))
                col3.metric("Forecast Duration", "24 Hours")
                col4.metric(
                    "Last Observation",
                    latest_time.strftime("%Y-%m-%d")
                    if pd.notna(latest_time)
                    else "N/A",
                )

                st.warning(
                    "Experimental LSTM prediction only. This is not an "
                    "official cyclone warning or operational weather forecast."
                )

                st.subheader("🗺️ Historical Cyclone Map")
                display_map(create_track_map(historical, predicted))

                st.subheader("📊 Forecast Details")
                st.dataframe(
                    predicted,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("📈 Movement Analysis")
                c1, c2 = st.columns(2)

                if "MOVEMENT_SPEED" in predicted.columns:
                    c1.metric(
                        "Average Predicted Speed",
                        f"{pd.to_numeric(predicted['MOVEMENT_SPEED'], errors='coerce').mean():.2f} km/h",
                    )

                if "MOVEMENT_DIRECTION" in predicted.columns:
                    c2.metric(
                        "Final Movement Direction",
                        f"{pd.to_numeric(predicted['MOVEMENT_DIRECTION'], errors='coerce').iloc[-1]:.2f}°",
                    )

                st.download_button(
                    "Download Historical Cyclone Forecast",
                    predicted.to_csv(index=False),
                    file_name="historical_cyclone_forecast.csv",
                    mime="text/csv",
                )

                with st.expander("View Historical Observations"):
                    st.dataframe(
                        historical,
                        use_container_width=True,
                        hide_index=True,
                    )


# ==========================================================
# TAB 2 - AUTOMATIC MULTI-CYCLONE MONITORING
# ==========================================================

with active_tab:
    st.header("🌪️ Active Cyclone Monitoring")
    st.write(
        "Discover active cyclones from CIRA, inspect their available "
        "historical tracks, and generate an individual experimental "
        "24-hour LSTM forecast."
    )

    st.info(
        "CIRA observations may be delayed. Forecasts are experimental "
        "research outputs and must not be used for safety decisions."
    )

    # Automatic refresh: on dashboard start and every 15 minutes
    # while this Streamlit session remains open.
    automatic_refresh_controls()

    st.caption(
        "The dashboard checks CIRA every 15 minutes while it remains open. "
        "Newly discovered tracks are prepared, and forecasts are generated "
        "for every cyclone marked LSTM eligible."
    )
    st.divider()

    summary = load_csv_if_exists(ACTIVE_SUMMARY)
    prep_summary = load_csv_if_exists(PREPARATION_SUMMARY)

    if summary is None or summary.empty:
        st.warning(
            "Active cyclone summary is not available. Click the refresh "
            "button to discover and download active cyclone tracks."
        )
    else:
        summary = summary.copy()

        sid_col = first_existing_column(summary, ["SID", "STORM_ID", "storm_id"])
        name_col = first_existing_column(summary, ["NAME", "name", "STORM_NAME"])
        basin_col = first_existing_column(summary, ["BASIN", "basin"])
        count_col = first_existing_column(
            summary, ["OBSERVATIONS", "OBSERVATION_COUNT", "N_OBSERVATIONS", "COUNT"]
        )
        latest_col = first_existing_column(
            summary, ["LATEST_OBSERVATION", "LAST_OBSERVATION", "LATEST_TIME"]
        )

        if sid_col is None:
            st.error(
                "The active cyclone summary does not contain a recognizable "
                "storm ID column. Expected SID."
            )
        else:
            summary[sid_col] = summary[sid_col].astype(str)

            # --------------------------------------------------
            # GLOBAL ACTIVE CYCLONE OVERVIEW
            # --------------------------------------------------

            st.subheader("🌍 Global Active Cyclone Overview")
            st.write(
                "All discovered active cyclones with available historical "
                "track data are displayed together. Click a marker or track "
                "to inspect cyclone information."
            )

            # --------------------------------------------------
            # GLOBAL MAP FILTERS AND CYCLONE FOCUS
            # --------------------------------------------------
            map_filter_col1, map_filter_col2, map_filter_col3 = st.columns([1, 1, 1.3])

            available_basins = []
            if basin_col and basin_col in summary.columns:
                available_basins = sorted(
                    summary[basin_col].dropna().astype(str).unique().tolist()
                )
            with map_filter_col1:
                selected_basins = st.multiselect(
                    "Filter by ocean basin",
                    options=available_basins,
                    default=available_basins,
                    key="global_map_basins",
                    help="Choose which CIRA basin tracks to display.",
                ) if available_basins else []
            with map_filter_col2:
                prediction_filter = st.selectbox(
                    "Prediction availability",
                    ["All cyclones", "With LSTM prediction only", "Without LSTM prediction"],
                    key="global_prediction_filter",
                )
            with map_filter_col3:
                storm_options = {}
                for _, storm_row in summary.iterrows():
                    storm_sid = str(storm_row[sid_col]).strip()
                    storm_label = (
                        str(storm_row[name_col]).strip()
                        if name_col and pd.notna(storm_row.get(name_col))
                        else "Unnamed / Invest"
                    )
                    if storm_sid and storm_sid.lower() != "nan":
                        storm_options[f"{storm_label} ({storm_sid})"] = storm_sid
                selected_label = st.selectbox(
                    "Highlight a cyclone",
                    ["None"] + list(storm_options.keys()),
                    key="global_highlight_cyclone",
                )
                highlighted_sid = storm_options.get(selected_label)

            filtered_summary = summary.copy()
            if basin_col and available_basins:
                filtered_summary = filtered_summary[
                    filtered_summary[basin_col].astype(str).isin(selected_basins)
                ]

            def _has_fresh_prediction(storm_sid):
                input_path = DATA_DIR / "active_lstm_inputs" / f"{storm_sid}_3hour_input.csv"
                output_path = RESULTS_DIR / f"{storm_sid}_lstm_prediction.csv"
                return (
                    input_path.exists() and output_path.exists()
                    and output_path.stat().st_mtime >= input_path.stat().st_mtime
                )

            filtered_sids = filtered_summary[sid_col].astype(str).str.strip()
            if prediction_filter != "All cyclones":
                has_prediction = filtered_sids.map(_has_fresh_prediction)
                if prediction_filter == "With LSTM prediction only":
                    filtered_summary = filtered_summary[has_prediction]
                else:
                    filtered_summary = filtered_summary[~has_prediction]

            global_map, plotted_count, predicted_count = create_global_active_map(
                filtered_summary,
                sid_col,
                name_col,
                basin_col,
                highlighted_sid,
            )

            map_col1, map_col2 = st.columns(2)
            map_col1.metric(
                "Cyclones shown on map",
                plotted_count,
            )
            map_col2.metric(
                "Cyclones with LSTM predictions",
                predicted_count,
            )

            if global_map is not None and plotted_count > 0:
                display_map(global_map, height=700)
            else:
                st.warning(
                    "No valid active cyclone tracks are available for the "
                    "global map. Refresh the active cyclone data."
                )

            st.caption(
                "Blue solid lines show historical tracks; red dashed lines "
                "show experimental LSTM forecasts; green markers show the "
                "latest observed positions. A selected cyclone is highlighted "
                "in teal. Use the filters above to explore the map."
            )

            st.caption(
                "This map shows cyclones discovered by the configured CIRA "
                "source that have valid track files. It does not guarantee "
                "that every active cyclone worldwide is represented."
            )

            st.divider()

            # Attach preparation eligibility and synthetic-input information.
            if prep_summary is not None and not prep_summary.empty:
                prep_sid_col = first_existing_column(
                    prep_summary, ["SID", "STORM_ID", "storm_id"]
                )
                if prep_sid_col:
                    prep_summary[prep_sid_col] = prep_summary[
                        prep_sid_col
                    ].astype(str)
                    summary = summary.merge(
                        prep_summary,
                        left_on=sid_col,
                        right_on=prep_sid_col,
                        how="left",
                        suffixes=("", "_PREP"),
                    )

            prep_eligible_col = first_existing_column(
                summary, ["LSTM_ELIGIBLE", "ELIGIBLE", "IS_ELIGIBLE"]
            )
            synthetic_col = first_existing_column(
                summary,
                [
                    "SYNTHETIC_OBSERVATIONS",
                    "SYNTHETIC_COUNT",
                    "N_SYNTHETIC",
                ],
            )

            # Create selector labels and retain SID as the key.
            options = {}
            for _, row in summary.iterrows():
                sid = str(row[sid_col])
                name = (
                    str(row[name_col]).strip()
                    if name_col and pd.notna(row.get(name_col))
                    else "Unnamed / Invest"
                )
                basin = (
                    str(row[basin_col]).strip()
                    if basin_col and pd.notna(row.get(basin_col))
                    else "Basin unavailable"
                )

                eligible = True
                if prep_eligible_col and pd.notna(row.get(prep_eligible_col)):
                    value = str(row[prep_eligible_col]).strip().lower()
                    eligible = value in ["true", "1", "yes", "y"]

                status = "LSTM eligible" if eligible else "Insufficient data"
                label = f"{name} | {sid} | {basin} | {status}"
                options[label] = sid

            if not options:
                st.warning("No active cyclone entries were found.")
            else:
                selected_label = st.selectbox(
                    "Select an active cyclone",
                    list(options.keys()),
                    key="active_cyclone_selector",
                )
                selected_sid = options[selected_label]

                selected_rows = summary[
                    summary[sid_col] == selected_sid
                ]
                selected_info = selected_rows.iloc[0]

                name = (
                    str(selected_info[name_col])
                    if name_col and pd.notna(selected_info.get(name_col))
                    else "Unnamed / Invest"
                )
                basin = (
                    str(selected_info[basin_col])
                    if basin_col and pd.notna(selected_info.get(basin_col))
                    else "Not available"
                )

                track_path = DATA_DIR / f"{selected_sid}_track.csv"
                input_path = (
                    DATA_DIR
                    / "active_lstm_inputs"
                    / f"{selected_sid}_3hour_input.csv"
                )
                prediction_path = RESULTS_DIR / f"{selected_sid}_lstm_prediction.csv"

                active_history = clean_track(load_csv_if_exists(track_path))
                prepared_input = load_csv_if_exists(input_path)

                st.subheader(f"🌀 {name} — {selected_sid}")
                st.write(f"**Basin:** {basin}")

                if active_history.empty:
                    st.error(
                        "A valid historical track is not available for this "
                        "cyclone. It cannot currently be mapped or forecast."
                    )
                else:
                    latest = active_history.iloc[-1]
                    latest_time = latest["ISO_TIME"]
                    current_time = pd.Timestamp.now(tz="UTC")
                    age_hours = (
                        current_time - latest_time
                    ).total_seconds() / 3600

                    # Use only values actually supplied by the source track.
                    def profile_number(candidates, suffix="", decimals=0):
                        column = first_existing_column(active_history, candidates)
                        if column is None:
                            return "N/A"
                        value = pd.to_numeric(latest.get(column), errors="coerce")
                        if pd.isna(value):
                            return "N/A"
                        return f"{value:.{decimals}f}{suffix}"

                    wind_value = profile_number(
                        ["USA_WIND", "WMO_WIND", "INTENSITY", "WIND"], " kt"
                    )
                    pressure_value = profile_number(
                        ["USA_PRES", "WMO_PRES", "PRESSURE"], " hPa"
                    )
                    speed_value = profile_number(
                        ["STORM_SPEED", "MOVEMENT_SPEED", "SPEED"], " kt", 1
                    )
                    direction_value = profile_number(
                        ["STORM_DIR", "MOVEMENT_DIRECTION", "DIRECTION"], "°"
                    )
                    first_time = active_history.iloc[0]["ISO_TIME"]
                    track_span_days = max(
                        0.0, (latest_time - first_time).total_seconds() / 86400
                    )
                    freshness_label = (
                        "Recent source observation" if age_hours <= 6
                        else "Delayed source observation" if age_hours <= 24
                        else "Stale source observation"
                    )
                    freshness_color = (
                        "#00D4AA" if age_hours <= 6
                        else "#F5B942" if age_hours <= 24
                        else "#FF7272"
                    )
                    safe_name = html.escape(name)
                    safe_sid = html.escape(selected_sid)
                    safe_basin = html.escape(basin)
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(120deg,#172B43,#1A344B);
                                    border:1px solid #34516E;border-radius:16px;
                                    padding:18px 20px;margin:8px 0 16px 0;">
                          <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center;">
                            <div>
                              <div style="font-size:12px;color:#9FB6CE;text-transform:uppercase;letter-spacing:.08em;">Cyclone profile</div>
                              <div style="font-size:25px;font-weight:800;color:#F4F8FF;margin-top:3px;">{safe_name}</div>
                              <div style="font-size:13px;color:#B9CBE0;margin-top:4px;">Storm ID: {safe_sid} &nbsp;•&nbsp; Basin: {safe_basin}</div>
                            </div>
                            <div style="padding:7px 11px;border-radius:999px;border:1px solid {freshness_color};color:{freshness_color};font-size:12px;font-weight:700;">
                              ● {freshness_label}
                            </div>
                          </div>
                          <div style="display:flex;gap:24px;flex-wrap:wrap;margin-top:17px;">
                            <div><div style="font-size:11px;color:#9FB6CE;">LATEST OBSERVATION (UTC)</div>
                              <div style="font-size:15px;color:#F4F8FF;font-weight:650;">{latest_time.strftime('%d %b %Y, %H:%M')}</div></div>
                            <div><div style="font-size:11px;color:#9FB6CE;">TRACK COVERAGE</div>
                              <div style="font-size:15px;color:#F4F8FF;font-weight:650;">{track_span_days:.1f} days</div></div>
                            <div><div style="font-size:11px;color:#9FB6CE;">OBSERVATIONS</div>
                              <div style="font-size:15px;color:#F4F8FF;font-weight:650;">{len(active_history)}</div></div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    metric1, metric2, metric3, metric4 = st.columns(4)
                    metric1.metric("Latitude", f"{latest['LAT']:.2f}°")
                    metric2.metric("Longitude", f"{latest['LON']:.2f}°")
                    metric3.metric("Available wind", wind_value)
                    metric4.metric("Available pressure", pressure_value)

                    movement_col1, movement_col2, movement_col3 = st.columns(3)
                    movement_col1.metric("Reported movement speed", speed_value)
                    movement_col2.metric("Reported movement direction", direction_value)
                    movement_col3.metric("Observation age", f"{age_hours:.1f} hours")

                    st.caption(
                        "Wind, pressure and movement values appear only when supplied by the source track. "
                        "Units and field meanings depend on the source dataset."
                    )

                    if age_hours < 0:
                        st.warning(
                            "The observation timestamp is in the future "
                            "relative to this computer's clock. Verify source time."
                        )
                    elif age_hours <= 6:
                        st.success(
                            f"Observation age: {age_hours:.1f} hours. "
                            "Recently updated according to the source timestamp."
                        )
                    elif age_hours <= 24:
                        st.warning(
                            f"Observation age: {age_hours:.1f} hours. "
                            "The source data may be delayed."
                        )
                    else:
                        st.error(
                            f"Observation age: {age_hours:.1f} hours. "
                            "Data is stale; do not treat it as a current position."
                        )

                    st.divider()

                    # Show input preparation diagnostics.
                    st.subheader("🧪 LSTM Input Readiness")

                    synthetic_count = 0
                    if (
                        prepared_input is not None
                        and "IS_SYNTHETIC" in prepared_input.columns
                    ):
                        synthetic_count = int(
                            prepared_input["IS_SYNTHETIC"]
                            .astype(str)
                            .str.lower()
                            .isin(["true", "1", "yes"])
                            .sum()
                        )

                    eligible = True
                    if prep_eligible_col and pd.notna(
                        selected_info.get(prep_eligible_col)
                    ):
                        eligible = (
                            str(selected_info[prep_eligible_col])
                            .strip()
                            .lower()
                            in ["true", "1", "yes", "y"]
                        )

                    p1, p2, p3 = st.columns(3)
                    p1.metric(
                        "Prepared Input Rows",
                        len(prepared_input)
                        if prepared_input is not None
                        else 0,
                    )
                    p2.metric("Synthetic Input Rows", synthetic_count)
                    p3.metric(
                        "Prediction Readiness",
                        "Eligible" if eligible else "Not eligible",
                    )

                    if synthetic_count:
                        st.warning(
                            f"The prepared track contains {synthetic_count} "
                            "interpolated observations. These are synthetic "
                            "positions, not direct observations."
                        )

                    if prepared_input is None or prepared_input.empty:
                        st.warning(
                            "Prepared LSTM input is unavailable. Refresh the "
                            "active cyclone data first."
                        )
                    elif not eligible:
                        st.warning(
                            "This cyclone is currently marked ineligible by "
                            "the preparation step. A forecast will not be generated."
                        )
                    else:
                        if st.button(
                            f"🧠 Generate / Refresh 24-hour Forecast for {name}",
                            type="primary",
                            key=f"predict_{selected_sid}",
                        ):
                            try:
                                with st.spinner(
                                    f"Generating LSTM forecast for {name}..."
                                ):
                                    output = run_script(
                                        "predict_active_cyclone.py",
                                        selected_sid,
                                        timeout=300,
                                    )
                                st.text(output)

                                if not prediction_path.exists():
                                    raise FileNotFoundError(
                                        "Prediction script completed but the "
                                        f"expected output was not found: {prediction_path}"
                                    )

                                st.session_state[
                                    "active_prediction_sid"
                                ] = selected_sid
                                st.success(
                                    f"Experimental forecast generated for {name}."
                                )
                                st.rerun()

                            except Exception as error:
                                st.error("Active cyclone prediction failed.")
                                st.exception(error)

                    # Display a forecast only when it is eligible and its
                    # output file is at least as new as the prepared input.
                    active_prediction = None

                    prediction_is_fresh = (
                        eligible
                        and prediction_path.exists()
                        and input_path.exists()
                        and prediction_path.stat().st_mtime
                        >= input_path.stat().st_mtime
                    )

                    if prediction_is_fresh:
                        active_prediction = load_csv_if_exists(
                            prediction_path
                        )

                    st.subheader("🗺️ Active Cyclone Track and Prediction")

                    if active_prediction is not None:
                        active_map = create_track_map(
                            active_history,
                            active_prediction,
                        )
                    else:
                        active_map = create_track_map(active_history)

                    display_map(active_map)

                    if active_prediction is not None and not active_prediction.empty:
                        st.subheader("📊 Experimental LSTM Forecast")
                        st.dataframe(
                            active_prediction,
                            use_container_width=True,
                            hide_index=True,
                        )

                        if "MOVEMENT_SPEED" in active_prediction.columns:
                            average_speed = pd.to_numeric(
                                active_prediction["MOVEMENT_SPEED"],
                                errors="coerce",
                            ).mean()
                            st.metric(
                                "Average Predicted Movement Speed",
                                f"{average_speed:.2f} km/h",
                            )

                        st.download_button(
                            "Download Selected Cyclone Forecast",
                            active_prediction.to_csv(index=False),
                            file_name=f"{selected_sid}_lstm_prediction.csv",
                            mime="text/csv",
                            key=f"download_{selected_sid}",
                        )
                    else:
                        st.info(
                            "The historical track is shown above. Generate a "
                            "forecast to display the experimental LSTM path "
                            "and forecast table."
                        )

                    with st.expander("View Historical Track Data"):
                        st.dataframe(
                            active_history,
                            use_container_width=True,
                            hide_index=True,
                        )

                    with st.expander("View Prepared LSTM Input"):
                        if prepared_input is not None:
                            st.dataframe(
                                prepared_input,
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("Prepared input is not available.")

                    with st.expander("View Active Cyclone Summary"):
                        st.dataframe(
                            summary,
                            use_container_width=True,
                            hide_index=True,
                        )


# ==========================================================
# 7. FOOTER
# ==========================================================

st.divider()
st.markdown(
    """
    <div style="text-align:center; color:grey;">
        <b>Cyclone Movement Prediction System</b><br>
        LSTM-Based Experimental Research Project<br>
        Historical Data Analysis | CIRA Active Cyclone Integration |
        Experimental 24-Hour Track Prediction<br><br>
        Created by SanthoshM
    </div>
    """,
    unsafe_allow_html=True,
)
