from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "production_real_sensor_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "production"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "gnss_outage_demo_8500_60s.csv"
)

OUTPUT_PLOT = (
    OUTPUT_DIR
    / "gnss_outage_demo_8500_60s.png"
)


# ============================================================
# OUTAGE SETTINGS
# ============================================================

OUTAGE_START_S = 8500.0
OUTAGE_DURATION_S = 60.0

OUTAGE_END_S = (
    OUTAGE_START_S +
    OUTAGE_DURATION_S
)

EARTH_RADIUS_M = 6371000.0


# ============================================================
# FUNCTIONS
# ============================================================

def latlon_to_local(
    latitude,
    longitude,
    origin_latitude,
    origin_longitude
):
    """
    Convert latitude/longitude to local
    North/East coordinates in metres.
    """

    latitude = np.asarray(
        latitude,
        dtype=float
    )

    longitude = np.asarray(
        longitude,
        dtype=float
    )

    lat0_rad = np.radians(
        origin_latitude
    )

    north = (
        np.radians(
            latitude - origin_latitude
        )
        * EARTH_RADIUS_M
    )

    east = (
        np.radians(
            longitude - origin_longitude
        )
        * EARTH_RADIUS_M
        * np.cos(lat0_rad)
    )

    return north, east


def local_to_latlon(
    north,
    east,
    origin_latitude,
    origin_longitude
):
    """
    Convert local North/East coordinates
    back to latitude/longitude.
    """

    lat0_rad = np.radians(
        origin_latitude
    )

    latitude = (
        origin_latitude
        + np.degrees(
            north / EARTH_RADIUS_M
        )
    )

    longitude = (
        origin_longitude
        + np.degrees(
            east
            / (
                EARTH_RADIUS_M
                * np.cos(lat0_rad)
            )
        )
    )

    return latitude, longitude


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PRODUCTION GNSS OUTAGE DEMONSTRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading production predictions...")

    df = pd.read_csv(
        INPUT_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "time_s"
    ).reset_index(
        drop=True
    )

    print(
        f"Loaded {len(df):,} rows."
    )

    # --------------------------------------------------------
    # Select required columns
    # --------------------------------------------------------

    required = [
        "timestamp",
        "time_s",
        "latitude",
        "longitude",
        "vn_ai_mps",
        "ve_ai_mps",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Remove rows without AI prediction
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "vn_ai_mps",
            "ve_ai_mps",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Locate outage
    # --------------------------------------------------------

    outage_mask = (
        (df["time_s"] >= OUTAGE_START_S)
        &
        (df["time_s"] <= OUTAGE_END_S)
    )

    outage_rows = df[
        outage_mask
    ].copy()

    if len(outage_rows) == 0:

        raise ValueError(
            "No rows found inside the requested "
            "GNSS outage interval."
        )

    print("\nGNSS outage:")

    print(
        f"Start: {OUTAGE_START_S:.1f} s"
    )

    print(
        f"End:   {OUTAGE_END_S:.1f} s"
    )

    print(
        f"Duration: {OUTAGE_DURATION_S:.1f} s"
    )

    print(
        f"Rows: {len(outage_rows):,}"
    )

    # --------------------------------------------------------
    # Reference origin
    # --------------------------------------------------------

    origin_lat = float(
        outage_rows.iloc[0]["latitude"]
    )

    origin_lon = float(
        outage_rows.iloc[0]["longitude"]
    )

    # --------------------------------------------------------
    # Convert GNSS reference to local coordinates
    # --------------------------------------------------------

    ref_north, ref_east = latlon_to_local(
        outage_rows["latitude"].values,
        outage_rows["longitude"].values,
        origin_lat,
        origin_lon
    )

    # --------------------------------------------------------
    # AI dead reckoning
    # --------------------------------------------------------

    ai_north = np.zeros(
        len(outage_rows),
        dtype=float
    )

    ai_east = np.zeros(
        len(outage_rows),
        dtype=float
    )

    timestamps = (
        outage_rows["timestamp"]
        .values
    )

    times = (
        outage_rows["time_s"]
        .values
    )

    vn = (
        outage_rows["vn_ai_mps"]
        .values
    )

    ve = (
        outage_rows["ve_ai_mps"]
        .values
    )

    # --------------------------------------------------------
    # Integrate AI velocity
    # --------------------------------------------------------

    for i in range(1, len(outage_rows)):

        dt = (
            times[i] -
            times[i - 1]
        )

        # Protect against abnormal timestamp gaps
        if dt <= 0 or dt > 1.0:

            dt = 0.1

        ai_north[i] = (
            ai_north[i - 1]
            + vn[i] * dt
        )

        ai_east[i] = (
            ai_east[i - 1]
            + ve[i] * dt
        )

    # --------------------------------------------------------
    # AI latitude/longitude
    # --------------------------------------------------------

    ai_latitude, ai_longitude = (
        local_to_latlon(
            ai_north,
            ai_east,
            origin_lat,
            origin_lon
        )
    )

    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    north_error = (
        ai_north -
        ref_north
    )

    east_error = (
        ai_east -
        ref_east
    )

    position_error = np.sqrt(
        north_error ** 2
        +
        east_error ** 2
    )

    # --------------------------------------------------------
    # Create result
    # --------------------------------------------------------

    result = pd.DataFrame({

        "timestamp": timestamps,

        "time_s": times,

        "reference_latitude":
            outage_rows["latitude"].values,

        "reference_longitude":
            outage_rows["longitude"].values,

        "reference_north_m":
            ref_north,

        "reference_east_m":
            ref_east,

        "ai_latitude":
            ai_latitude,

        "ai_longitude":
            ai_longitude,

        "ai_north_m":
            ai_north,

        "ai_east_m":
            ai_east,

        "vn_ai_mps":
            vn,

        "ve_ai_mps":
            ve,

        "position_error_m":
            position_error,
    })

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    final_error = float(
        position_error[-1]
    )

    mean_error = float(
        position_error.mean()
    )

    max_error = float(
        position_error.max()
    )

    reference_distance = float(
        np.sqrt(
            ref_north[-1] ** 2
            +
            ref_east[-1] ** 2
        )
    )

    ai_distance = float(
        np.sqrt(
            ai_north[-1] ** 2
            +
            ai_east[-1] ** 2
        )
    )

    drift_percent = (
        final_error /
        max(reference_distance, 1e-6)
        * 100
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    plt.plot(
        ref_east,
        ref_north,
        label="GNSS Reference",
        linewidth=2
    )

    plt.plot(
        ai_east,
        ai_north,
        label="AI Dead Reckoning",
        linewidth=2
    )

    plt.scatter(
        [0],
        [0],
        s=70,
        label="GNSS Loss Start"
    )

    plt.scatter(
        [ref_east[-1]],
        [ref_north[-1]],
        s=70,
        label="GNSS Final Position"
    )

    plt.scatter(
        [ai_east[-1]],
        [ai_north[-1]],
        s=70,
        label="AI Final Position"
    )

    plt.xlabel(
        "East displacement (m)"
    )

    plt.ylabel(
        "North displacement (m)"
    )

    plt.title(
        "AI Dead Reckoning During 60s GNSS Outage"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.axis(
        "equal"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GNSS OUTAGE RESULTS")
    print("=" * 70)

    print(
        f"\nReference displacement : "
        f"{reference_distance:.3f} m"
    )

    print(
        f"AI displacement        : "
        f"{ai_distance:.3f} m"
    )

    print(
        f"Mean position error    : "
        f"{mean_error:.3f} m"
    )

    print(
        f"Final position error   : "
        f"{final_error:.3f} m"
    )

    print(
        f"Maximum position error : "
        f"{max_error:.3f} m"
    )

    print(
        f"Final drift            : "
        f"{drift_percent:.3f}%"
    )

    print("\nSaved CSV:")

    print(
        OUTPUT_CSV
    )

    print("\nSaved trajectory plot:")

    print(
        OUTPUT_PLOT
    )

    print("=" * 70)


if __name__ == "__main__":
    main()