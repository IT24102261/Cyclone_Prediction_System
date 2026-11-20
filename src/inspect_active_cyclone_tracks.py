import requests
import pandas as pd

from io import StringIO
from pathlib import Path

from discover_active_cyclones import discover_active_cyclones


# ==========================================
# CONFIGURATION
# ==========================================

OUTPUT_DIR = Path("dataset/processed")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# DOWNLOAD INDIVIDUAL STORM TRACK
# ==========================================

def retrieve_storm_track(cyclone):

    name = cyclone["NAME"]
    storm_id = cyclone["SID"]
    url = cyclone["URL"]

    print("\n" + "=" * 60)
    print(f"CYCLONE: {name}")
    print(f"STORM ID: {storm_id}")
    print("=" * 60)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=60
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(response.text),
        header=None
    )

    print("Tables found:", len(tables))

    if len(tables) < 2:

        print("Track history table not found.")

        return None

    # The second table contains historical observations
    table = tables[1].copy()

    print("\nTable columns:")
    print(table.columns.tolist())

    print("\nFirst five rows:")
    print(table.head().to_string(index=False))

    # Expected CIRA table structure
    table.columns = [
        "ISO_TIME",
        "LAT",
        "LON",
        "INTENSITY"
    ]

    # Remove repeated header row if present
    table = table[
        table["ISO_TIME"].astype(str).str.lower()
        != "synoptic time"
    ].copy()

    table["ISO_TIME"] = pd.to_datetime(
        table["ISO_TIME"],
        errors="coerce",
        utc=True
    )

    for column in ["LAT", "LON", "INTENSITY"]:

        table[column] = pd.to_numeric(
            table[column],
            errors="coerce"
        )

    table = table.dropna(
        subset=["ISO_TIME", "LAT", "LON"]
    )

    table = table[
        (table["LAT"] != 0)
        &
        (table["LON"] != 0)
    ]

    table = table.sort_values(
        "ISO_TIME"
    ).reset_index(drop=True)

    print("\nValid historical observations:", len(table))

    if not table.empty:

        print("\nLatest observation:")
        print(table.iloc[-1].to_string())

    # Save individual track
    output_file = OUTPUT_DIR / f"{storm_id}_track.csv"

    table.to_csv(
        output_file,
        index=False
    )

    print("\nSaved:", output_file)

    return table


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("MULTI-CYCLONE TRACK INSPECTION")
    print("=" * 60)

    cyclones = discover_active_cyclones()

    print("\nDetected cyclones:", len(cyclones))

    results = []

    for cyclone in cyclones:

        try:

            track = retrieve_storm_track(cyclone)

            if track is not None:

                results.append({

                    "NAME": cyclone["NAME"],

                    "SID": cyclone["SID"],

                    "BASIN": cyclone["BASIN"],

                    "OBSERVATIONS": len(track),

                    "LATEST_OBSERVATION": (
                        track["ISO_TIME"].max()
                        if not track.empty
                        else None
                    )

                })

        except Exception as error:

            print(
                f"\nFailed to retrieve {cyclone['SID']}: "
                f"{error}"
            )

    print("\n" + "=" * 60)
    print("MULTI-CYCLONE TRACK SUMMARY")
    print("=" * 60)

    if results:

        summary = pd.DataFrame(results)

        print(
            summary.to_string(index=False)
        )

        summary.to_csv(
            OUTPUT_DIR / "active_cyclone_summary.csv",
            index=False
        )

    else:

        print("No cyclone tracks retrieved.")

    print("\nTRACK INSPECTION COMPLETED")


if __name__ == "__main__":

    main()