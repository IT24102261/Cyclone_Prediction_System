import requests
import re

from bs4 import BeautifulSoup
from urllib.parse import urljoin


# ==========================================
# CIRA CONFIGURATION
# ==========================================

BASE_URL = (
    "https://rammb-data.cira.colostate.edu/"
    "tc_realtime/"
)

ACTIVE_URL = BASE_URL


# ==========================================
# DISCOVER ACTIVE CYCLONES
# ==========================================

def discover_active_cyclones():

    print("\nConnecting to CIRA Active Cyclone Directory...")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        ACTIVE_URL,
        headers=headers,
        timeout=60
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    cyclones = []

    for link in soup.find_all("a", href=True):

        href = link["href"]

        match = re.search(
            r"storm_identifier=([a-zA-Z0-9]+)",
            href,
            re.IGNORECASE
        )

        if not match:
            continue

        storm_id = match.group(1)

        description = link.get_text(
            " ",
            strip=True
        )

        # Extract storm name
        name_match = re.search(
            r"\b([A-Z][A-Z0-9-]+)\s*$",
            description
        )

        name = (
            name_match.group(1)
            if name_match
            else "UNKNOWN"
        )

        # Extract basin from storm ID
        if storm_id.upper().startswith("IO"):
            basin = "North Indian Ocean"

        elif storm_id.upper().startswith("AL"):
            basin = "North Atlantic"

        elif storm_id.upper().startswith("EP"):
            basin = "Eastern Pacific"

        elif storm_id.upper().startswith("WP"):
            basin = "Western Pacific"

        elif storm_id.upper().startswith("SH"):
            basin = "Southern Hemisphere"

        elif storm_id.upper().startswith("CP"):
            basin = "Central Pacific"

        else:
            basin = "Other"

        storm_url = urljoin(
            BASE_URL,
            href
        )

        cyclones.append({

            "SID": storm_id,

            "NAME": name,

            "BASIN": basin,

            "URL": storm_url

        })

    # Remove duplicate IDs
    unique = {}

    for cyclone in cyclones:

        unique[cyclone["SID"]] = cyclone

    cyclones = list(
        unique.values()
    )

    return cyclones


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("AUTOMATIC ACTIVE CYCLONE DETECTION")
    print("=" * 55)

    try:

        cyclones = discover_active_cyclones()

        print("\nActive cyclones discovered:", len(cyclones))

        print("\nAVAILABLE CYCLONES")

        print("-" * 55)

        if not cyclones:

            print("No active cyclones found.")

        for cyclone in cyclones:

            print(
                f"\nName: {cyclone['NAME']}"
                f"\nStorm ID: {cyclone['SID']}"
                f"\nBasin: {cyclone['BASIN']}"
                f"\nURL: {cyclone['URL']}"
            )

        print("\n" + "=" * 55)

        print("DETECTION COMPLETED")

    except Exception as error:

        print("\nERROR:", error)