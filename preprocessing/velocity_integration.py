# ================================================================
# STEP 6
# IMU VELOCITY INTEGRATION
# ================================================================
#
# Project:
# AI-ML Based Intelligent Dead Reckoning System
#
# STEP 6 PIPELINE:
#
# NED acceleration
#        ↓
# Timestamp validation
#        ↓
# dt calculation
#        ↓
# Trapezoidal integration
#        ↓
# Raw IMU velocity
#        ↓
# Stationary detection
#        ↓
# Zero Velocity Update (ZUPT)
#        ↓
# Corrected velocity
#        ↓
# GPS speed comparison
#
# Position integration is NOT performed here.
#
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ================================================================
# FILE PATHS
# ================================================================

NED_PATH = Path(
    "data/processed/ned_acceleration.csv"
)

RAW_PATH = Path(
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

OUTPUT_PATH = Path(
    "data/processed/imu_velocity.csv"
)

PLOT_PATH = Path(
    "outputs/velocity_integration_validation.png"
)


# ================================================================
# PARAMETERS
# ================================================================

# Maximum acceptable integration interval.
#
# Your dataset normally has approximately:
#
#     dt = 0.1 s
#
# There are two larger gaps:
#
#     1.158 s
#     1.365 s
#
# These must NOT be directly integrated.
#
MAX_INTEGRATION_DT = 0.25


# GPS speed below this value is treated as stationary.
STATIONARY_GPS_SPEED_KMH = 0.5


# NED acceleration magnitude threshold for stationary detection.
#
# This is intentionally generous because the phone has
# residual sensor noise.
STATIONARY_ACCEL_THRESHOLD = 1.5


# ================================================================
# COLUMN FINDER
# ================================================================

def find_column(df, candidates):

    """
    Find a column using:

    1. Exact match
    2. Case-insensitive match
    """

    # ------------------------------------------------------------
    # Exact match
    # ------------------------------------------------------------

    for candidate in candidates:

        if candidate in df.columns:

            return candidate

    # ------------------------------------------------------------
    # Case-insensitive match
    # ------------------------------------------------------------

    lower_map = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = str(candidate).strip().lower()

        if key in lower_map:

            return lower_map[key]

    return None


# ================================================================
# LOAD NED DATA
# ================================================================

def load_ned_data():

    print("\nLoading NED acceleration...")

    if not NED_PATH.exists():

        raise FileNotFoundError(
            f"\nNED acceleration file not found:\n"
            f"{NED_PATH}\n\n"
            f"Run ned_acceleration.py first."
        )

    df = pd.read_csv(
        NED_PATH
    )

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    return df


# ================================================================
# LOAD RAW DATA
# ================================================================

def load_raw_data():

    print("\nLoading raw sensor data...")

    if not RAW_PATH.exists():

        raise FileNotFoundError(
            f"\nRaw dataset not found:\n"
            f"{RAW_PATH}"
        )

    df = pd.read_csv(
        RAW_PATH,
        encoding="cp1252"
    )

    print(
        f"Raw rows : {len(df)}"
    )

    return df


# ================================================================
# TIMESTAMP PREPARATION
# ================================================================

def prepare_timestamp(
    df,
    dataframe_name
):

    df = df.copy()

    print(
        f"\nPreparing timestamp for "
        f"{dataframe_name}..."
    )

    # ============================================================
    # CASE 1
    # Already has timestamp
    # ============================================================

    timestamp_column = find_column(
        df,
        [
            "timestamp",
            "Timestamp",
            "TIMESTAMP"
        ]
    )

    if timestamp_column is not None:

        print(
            f"Using existing timestamp column: "
            f"{timestamp_column}"
        )

        df["timestamp"] = pd.to_datetime(
            df[timestamp_column],
            errors="coerce"
        )

        invalid = (
            df["timestamp"]
            .isna()
            .sum()
        )

        print(
            f"Invalid timestamps : {invalid}"
        )

        if invalid > 0:

            raise ValueError(
                f"{invalid} invalid timestamps "
                f"found in {dataframe_name}."
            )

        return df

    # ============================================================
    # CASE 2
    # Original IO-VNBD date column
    # ============================================================

    date_column = None

    exact_candidates = [

        "DATE (YYYY-MO-DD HH-MI-SS_SSS)",

        "DATE_YYYY-MO-DD_HH-MI-SS_SSS",

        "DATE_YYYY-MO-DD_HH-MI-SS_SSS",

    ]

    for candidate in exact_candidates:

        if candidate in df.columns:

            date_column = candidate

            break

    # ============================================================
    # CASE 3
    # Flexible DATE search
    # ============================================================

    if date_column is None:

        for column in df.columns:

            text = str(column).upper()

            if (
                "DATE" in text
                and "YYYY" in text
            ):

                date_column = column

                break

    # ============================================================
    # CASE 4
    # Last-resort DATE search
    # ============================================================

    if date_column is None:

        for column in df.columns:

            text = str(column).upper()

            if "DATE" in text:

                date_column = column

                break

    # ============================================================
    # FAILURE
    # ============================================================

    if date_column is None:

        print(
            "\nERROR: Could not detect timestamp."
        )

        print(
            "\nAvailable columns:"
        )

        for index, column in enumerate(
            df.columns
        ):

            print(
                f"{index:02d}: {column}"
            )

        raise ValueError(
            f"Could not find timestamp/date "
            f"column in {dataframe_name}."
        )

    print(
        f"Using date column: "
        f"{date_column}"
    )

    # ============================================================
    # IO-VNBD FORMAT
    #
    # Example:
    #
    # 2019-09-07 09:13:29:506
    #
    # ============================================================

    df["timestamp"] = pd.to_datetime(
        df[date_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    invalid = (
        df["timestamp"]
        .isna()
        .sum()
    )

    print(
        f"Invalid timestamps : {invalid}"
    )

    # ============================================================
    # FALLBACK PARSER
    # ============================================================

    if invalid > 0:

        print(
            "\nTrying fallback timestamp parser..."
        )

        fallback = pd.to_datetime(
            df[date_column],
            errors="coerce"
        )

        fallback_invalid = (
            fallback.isna()
            .sum()
        )

        if fallback_invalid < invalid:

            df["timestamp"] = fallback

            invalid = fallback_invalid

            print(
                f"Invalid timestamps after "
                f"fallback : {invalid}"
            )

    # ============================================================
    # FINAL VALIDATION
    # ============================================================

    if invalid > 0:

        raise ValueError(
            f"{invalid} invalid timestamps "
            f"remain in {dataframe_name}."
        )

    valid = df[
        "timestamp"
    ].dropna()

    if len(valid) > 0:

        print(
            f"Start : {valid.iloc[0]}"
        )

        print(
            f"End   : {valid.iloc[-1]}"
        )

    return df


# ================================================================
# TIMESTAMP VALIDATION
# ================================================================

def validate_timestamp_range(df):

    timestamps = df[
        "timestamp"
    ]

    valid = timestamps.dropna()

    if len(valid) == 0:

        raise ValueError(
            "No valid timestamps."
        )

    start = valid.iloc[0]

    end = valid.iloc[-1]

    duration = (
        end - start
    ).total_seconds()

    print(
        f"Start    : {start}"
    )

    print(
        f"End      : {end}"
    )

    print(
        f"Duration : {duration:.3f} s"
    )

    return duration


# ================================================================
# DT CALCULATION
# ================================================================

def calculate_dt(df):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TIMESTAMP SAMPLING"
    )

    print(
        "=" * 70
    )

    timestamps = pd.to_datetime(
        df["timestamp"]
    )

    # ============================================================
    # IMPORTANT
    #
    # .copy() is required because newer NumPy/pandas combinations
    # can return a read-only array from to_numpy().
    # ============================================================

    dt = (
        timestamps.diff()
        .dt.total_seconds()
        .to_numpy()
        .copy()
    )

    # First sample has no previous sample

    dt[0] = np.nan

    # ============================================================
    # Find valid intervals
    # ============================================================

    valid_mask = (
        np.isfinite(dt)
        &
        (dt > 0)
        &
        (dt <= MAX_INTEGRATION_DT)
    )

    valid_dt = dt[
        valid_mask
    ]

    if len(valid_dt) == 0:

        raise ValueError(
            "No valid timestamp intervals."
        )

    median_dt = np.median(
        valid_dt
    )

    print(
        f"Median dt : "
        f"{np.median(valid_dt):.4f} s"
    )

    print(
        f"Mean dt   : "
        f"{np.mean(valid_dt):.4f} s"
    )

    print(
        f"Min dt    : "
        f"{np.min(valid_dt):.4f} s"
    )

    print(
        f"Max dt    : "
        f"{np.max(valid_dt):.4f} s"
    )

    # ============================================================
    # Detect invalid/large intervals
    # ============================================================

    invalid_mask = (
        ~np.isfinite(dt)
        |
        (dt <= 0)
        |
        (dt > MAX_INTEGRATION_DT)
    )

    invalid_count = np.sum(
        invalid_mask
    )

    print(
        f"Invalid/large intervals : "
        f"{invalid_count}"
    )

    # ============================================================
    # Print large gaps
    # ============================================================

    large_gap_indices = np.where(
        dt > MAX_INTEGRATION_DT
    )[0]

    if len(large_gap_indices) > 0:

        print(
            "\nLarge timestamp gaps:"
        )

        for index in (
            large_gap_indices[:20]
        ):

            print(
                f"Row {index}: "
                f"{dt[index]:.3f} s"
            )

    # ============================================================
    # Replace bad intervals
    # ============================================================

    dt[invalid_mask] = median_dt

    print(
        f"\nReplacement dt : "
        f"{median_dt:.4f} s"
    )

    return dt


# ================================================================
# GET NED ACCELERATION
# ================================================================

def get_ned_acceleration(df):

    print(
        "\nDetecting NED acceleration columns..."
    )

    # ------------------------------------------------------------
    # Candidate names
    # ------------------------------------------------------------

    north_candidates = [

        "accel_north",

        "ned_acceleration_n",

        "ned_accel_n",

        "north_acceleration",

        "North acceleration",

        "NED_North",

    ]

    east_candidates = [

        "accel_east",

        "ned_acceleration_e",

        "ned_accel_e",

        "east_acceleration",

        "East acceleration",

        "NED_East",

    ]

    down_candidates = [

        "accel_down",

        "ned_acceleration_d",

        "ned_accel_d",

        "down_acceleration",

        "Down acceleration",

        "NED_Down",

    ]

    north = find_column(
        df,
        north_candidates
    )

    east = find_column(
        df,
        east_candidates
    )

    down = find_column(
        df,
        down_candidates
    )

    # ============================================================
    # Flexible search
    # ============================================================

    if north is None:

        for column in df.columns:

            text = str(column).lower()

            if (
                "north" in text
                and "accel" in text
            ):

                north = column

                break

    if east is None:

        for column in df.columns:

            text = str(column).lower()

            if (
                "east" in text
                and "accel" in text
            ):

                east = column

                break

    if down is None:

        for column in df.columns:

            text = str(column).lower()

            if (
                "down" in text
                and "accel" in text
            ):

                down = column

                break

    # ============================================================
    # Print detected columns
    # ============================================================

    print(
        f"North : {north}"
    )

    print(
        f"East  : {east}"
    )

    print(
        f"Down  : {down}"
    )

    # ============================================================
    # Verify
    # ============================================================

    if (
        north is None
        or east is None
        or down is None
    ):

        print(
            "\nAvailable NED file columns:"
        )

        for column in df.columns:

            print(
                f" - {column}"
            )

        raise ValueError(
            "Could not detect NED acceleration columns."
        )

    acceleration = df[
        [
            north,
            east,
            down
        ]
    ].to_numpy(
        dtype=float
    )

    return acceleration


# ================================================================
# GET GPS SPEED
# ================================================================

def get_gps_speed(df):

    print(
        "\nDetecting GPS speed column..."
    )

    candidates = [

        "GPS_SPEED_Kmh",

        "GPS_SPEED_kmh",

        "GPS_SPEED",

        "gps_speed",

    ]

    gps_column = find_column(
        df,
        candidates
    )

    # ------------------------------------------------------------
    # Flexible search
    # ------------------------------------------------------------

    if gps_column is None:

        for column in df.columns:

            text = str(column).upper()

            if (
                "GPS" in text
                and "SPEED" in text
            ):

                gps_column = column

                break

    if gps_column is None:

        print(
            "\nAvailable raw columns:"
        )

        for column in df.columns:

            print(
                f" - {column}"
            )

        raise ValueError(
            "GPS speed column not found."
        )

    print(
        f"GPS speed : {gps_column}"
    )

    gps_speed = pd.to_numeric(
        df[gps_column],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    return gps_speed


# ================================================================
# STATIONARY DETECTION
# ================================================================

def detect_stationary(
    acceleration,
    gps_speed
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATIONARY DETECTION"
    )

    print(
        "=" * 70
    )

    # ============================================================
    # Acceleration magnitude
    # ============================================================

    acceleration_magnitude = (
        np.linalg.norm(
            acceleration,
            axis=1
        )
    )

    # ============================================================
    # GPS condition
    # ============================================================

    gps_stationary = (
        np.isfinite(gps_speed)
        &
        (
            gps_speed
            <= STATIONARY_GPS_SPEED_KMH
        )
    )

    # ============================================================
    # Acceleration condition
    # ============================================================

    acceleration_stationary = (
        np.isfinite(
            acceleration_magnitude
        )
        &
        (
            acceleration_magnitude
            <= STATIONARY_ACCEL_THRESHOLD
        )
    )

    # ============================================================
    # Combined
    # ============================================================

    stationary = (
        gps_stationary
        &
        acceleration_stationary
    )

    print(
        f"GPS stationary samples : "
        f"{np.sum(gps_stationary)}"
    )

    print(
        f"Low acceleration samples : "
        f"{np.sum(acceleration_stationary)}"
    )

    print(
        f"Combined stationary samples : "
        f"{np.sum(stationary)}"
    )

    print(
        f"Stationary percentage : "
        f"{100 * np.mean(stationary):.2f}%"
    )

    return stationary


# ================================================================
# VELOCITY INTEGRATION
# ================================================================

def integrate_velocity(
    acceleration,
    dt
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAW VELOCITY INTEGRATION"
    )

    print(
        "=" * 70
    )

    n = len(
        acceleration
    )

    velocity = np.zeros(
        (n, 3),
        dtype=float
    )

    print(
        "Using trapezoidal integration..."
    )

    # ============================================================
    # v[k] = v[k-1]
    #      + 0.5 * (a[k-1] + a[k]) * dt
    # ============================================================

    for i in range(
        1,
        n
    ):

        velocity[i] = (
            velocity[i - 1]
            +
            0.5
            *
            (
                acceleration[i - 1]
                +
                acceleration[i]
            )
            *
            dt[i]
        )

    print(
        "Velocity integration completed."
    )

    return velocity


# ================================================================
# ZERO VELOCITY UPDATE
# ================================================================

def apply_zupt(
    velocity,
    stationary
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ZERO VELOCITY UPDATE"
    )

    print(
        "=" * 70
    )

    corrected = velocity.copy()

    count = 0

    for i in range(
        len(corrected)
    ):

        if stationary[i]:

            corrected[i] = 0.0

            count += 1

    print(
        f"ZUPT corrections : {count}"
    )

    return corrected


# ================================================================
# VELOCITY MAGNITUDE
# ================================================================

def velocity_magnitude(
    velocity
):

    return np.linalg.norm(
        velocity,
        axis=1
    )


# ================================================================
# GPS SPEED CONVERSION
# ================================================================

def kmh_to_ms(
    speed_kmh
):

    return (
        speed_kmh
        /
        3.6
    )


# ================================================================
# VALIDATE VELOCITY
# ================================================================

def validate_velocity(
    raw_velocity,
    corrected_velocity,
    gps_speed,
    stationary
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VELOCITY VALIDATION"
    )

    print(
        "=" * 70
    )

    raw_speed = velocity_magnitude(
        raw_velocity
    )

    corrected_speed = velocity_magnitude(
        corrected_velocity
    )

    gps_speed_ms = kmh_to_ms(
        gps_speed
    )

    # ============================================================
    # Raw velocity
    # ============================================================

    print(
        "\nRAW IMU VELOCITY"
    )

    print(
        f"Final North : "
        f"{raw_velocity[-1, 0]:.4f} m/s"
    )

    print(
        f"Final East  : "
        f"{raw_velocity[-1, 1]:.4f} m/s"
    )

    print(
        f"Final Down  : "
        f"{raw_velocity[-1, 2]:.4f} m/s"
    )

    print(
        f"Maximum speed : "
        f"{np.max(raw_speed):.4f} m/s"
    )

    print(
        f"Maximum speed : "
        f"{np.max(raw_speed) * 3.6:.4f} km/h"
    )

    # ============================================================
    # Corrected velocity
    # ============================================================

    print(
        "\nZUPT-CORRECTED VELOCITY"
    )

    print(
        f"Final North : "
        f"{corrected_velocity[-1, 0]:.4f} m/s"
    )

    print(
        f"Final East  : "
        f"{corrected_velocity[-1, 1]:.4f} m/s"
    )

    print(
        f"Final Down  : "
        f"{corrected_velocity[-1, 2]:.4f} m/s"
    )

    print(
        f"Maximum speed : "
        f"{np.max(corrected_speed):.4f} m/s"
    )

    print(
        f"Maximum speed : "
        f"{np.max(corrected_speed) * 3.6:.4f} km/h"
    )

    # ============================================================
    # GPS comparison
    # ============================================================

    valid = (
        np.isfinite(
            corrected_speed
        )
        &
        np.isfinite(
            gps_speed_ms
        )
    )

    if np.sum(valid) > 0:

        error = (
            corrected_speed[valid]
            -
            gps_speed_ms[valid]
        )

        mae = np.mean(
            np.abs(error)
        )

        rmse = np.sqrt(
            np.mean(
                error ** 2
            )
        )

        print(
            "\nGPS SPEED COMPARISON"
        )

        print(
            f"GPS mean speed : "
            f"{np.mean(gps_speed_ms[valid]):.4f} m/s"
        )

        print(
            f"IMU mean speed : "
            f"{np.mean(corrected_speed[valid]):.4f} m/s"
        )

        print(
            f"Velocity MAE : "
            f"{mae:.4f} m/s"
        )

        print(
            f"Velocity RMSE : "
            f"{rmse:.4f} m/s"
        )

        print(
            f"Velocity MAE : "
            f"{mae * 3.6:.4f} km/h"
        )

        print(
            f"Velocity RMSE : "
            f"{rmse * 3.6:.4f} km/h"
        )

    # ============================================================
    # Stationary validation
    # ============================================================

    stationary_indices = (
        stationary
        &
        np.isfinite(
            corrected_speed
        )
    )

    if np.sum(
        stationary_indices
    ) > 0:

        stationary_speed = (
            corrected_speed[
                stationary_indices
            ]
        )

        print(
            "\nSTATIONARY VELOCITY"
        )

        print(
            f"Samples : "
            f"{len(stationary_speed)}"
        )

        print(
            f"Mean speed : "
            f"{np.mean(stationary_speed):.6f} m/s"
        )

        print(
            f"Maximum speed : "
            f"{np.max(stationary_speed):.6f} m/s"
        )

    return (
        raw_speed,
        corrected_speed,
        gps_speed_ms
    )


# ================================================================
# SAVE RESULTS
# ================================================================

def save_results(
    ned_df,
    dt,
    raw_velocity,
    corrected_velocity,
    stationary,
    gps_speed
):

    print(
        "\nSaving velocity results..."
    )

    output = pd.DataFrame()

    # ============================================================
    # Timestamp
    # ============================================================

    output[
        "timestamp"
    ] = ned_df[
        "timestamp"
    ]

    # ============================================================
    # dt
    # ============================================================

    output[
        "dt_seconds"
    ] = dt

    # ============================================================
    # Acceleration
    # ============================================================

    output[
        "accel_north_m_s2"
    ] = ned_df[
        get_ned_column(
            ned_df,
            "north"
        )
    ]

    output[
        "accel_east_m_s2"
    ] = ned_df[
        get_ned_column(
            ned_df,
            "east"
        )
    ]

    output[
        "accel_down_m_s2"
    ] = ned_df[
        get_ned_column(
            ned_df,
            "down"
        )
    ]

    # ============================================================
    # Raw velocity
    # ============================================================

    output[
        "velocity_north_raw_m_s"
    ] = raw_velocity[:, 0]

    output[
        "velocity_east_raw_m_s"
    ] = raw_velocity[:, 1]

    output[
        "velocity_down_raw_m_s"
    ] = raw_velocity[:, 2]

    output[
        "speed_raw_m_s"
    ] = velocity_magnitude(
        raw_velocity
    )

    # ============================================================
    # ZUPT velocity
    # ============================================================

    output[
        "velocity_north_m_s"
    ] = corrected_velocity[:, 0]

    output[
        "velocity_east_m_s"
    ] = corrected_velocity[:, 1]

    output[
        "velocity_down_m_s"
    ] = corrected_velocity[:, 2]

    output[
        "speed_m_s"
    ] = velocity_magnitude(
        corrected_velocity
    )

    output[
        "speed_kmh"
    ] = (
        velocity_magnitude(
            corrected_velocity
        )
        * 3.6
    )

    # ============================================================
    # GPS
    # ============================================================

    output[
        "gps_speed_kmh"
    ] = gps_speed

    output[
        "gps_speed_m_s"
    ] = kmh_to_ms(
        gps_speed
    )

    # ============================================================
    # Error
    # ============================================================

    output[
        "velocity_error_m_s"
    ] = (
        output[
            "speed_m_s"
        ]
        -
        output[
            "gps_speed_m_s"
        ]
    )

    # ============================================================
    # Stationary
    # ============================================================

    output[
        "stationary"
    ] = stationary.astype(
        int
    )

    # ============================================================
    # Save
    # ============================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print(
        f"Rows saved : {len(output)}"
    )

    return output


# ================================================================
# GET NED COLUMN FOR SAVING
# ================================================================

def get_ned_column(
    df,
    direction
):

    if direction == "north":

        candidates = [
            "accel_north",
            "ned_acceleration_n",
            "north_acceleration",
            "North acceleration"
        ]

    elif direction == "east":

        candidates = [
            "accel_east",
            "ned_acceleration_e",
            "east_acceleration",
            "East acceleration"
        ]

    else:

        candidates = [
            "accel_down",
            "ned_acceleration_d",
            "down_acceleration",
            "Down acceleration"
        ]

    column = find_column(
        df,
        candidates
    )

    if column is None:

        for candidate in df.columns:

            text = str(
                candidate
            ).lower()

            if (
                direction in text
                and "accel" in text
            ):

                return candidate

        raise ValueError(
            f"Could not find {direction} "
            f"acceleration column."
        )

    return column


# ================================================================
# GENERATE PLOT
# ================================================================

def generate_plot(
    output,
    stationary
):

    print(
        "\nGenerating velocity validation plot..."
    )

    timestamps = pd.to_datetime(
        output[
            "timestamp"
        ]
    )

    time_seconds = (
        timestamps
        -
        timestamps.iloc[0]
    ).dt.total_seconds().to_numpy()

    # ============================================================
    # Extract
    # ============================================================

    north = output[
        "velocity_north_m_s"
    ].to_numpy()

    east = output[
        "velocity_east_m_s"
    ].to_numpy()

    down = output[
        "velocity_down_m_s"
    ].to_numpy()

    imu_speed = output[
        "speed_kmh"
    ].to_numpy()

    gps_speed = output[
        "gps_speed_kmh"
    ].to_numpy()

    # ============================================================
    # Create plot
    # ============================================================

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 13),
        sharex=True
    )

    # ============================================================
    # NORTH
    # ============================================================

    axes[0].plot(
        time_seconds,
        north,
        label="IMU North velocity"
    )

    axes[0].axhline(
        0,
        linestyle="--"
    )

    axes[0].set_ylabel(
        "m/s"
    )

    axes[0].set_title(
        "NED North Velocity"
    )

    axes[0].grid(
        alpha=0.3
    )

    axes[0].legend()

    # ============================================================
    # EAST
    # ============================================================

    axes[1].plot(
        time_seconds,
        east,
        label="IMU East velocity"
    )

    axes[1].axhline(
        0,
        linestyle="--"
    )

    axes[1].set_ylabel(
        "m/s"
    )

    axes[1].set_title(
        "NED East Velocity"
    )

    axes[1].grid(
        alpha=0.3
    )

    axes[1].legend()

    # ============================================================
    # DOWN
    # ============================================================

    axes[2].plot(
        time_seconds,
        down,
        label="IMU Down velocity"
    )

    axes[2].axhline(
        0,
        linestyle="--"
    )

    axes[2].set_ylabel(
        "m/s"
    )

    axes[2].set_title(
        "NED Down Velocity"
    )

    axes[2].grid(
        alpha=0.3
    )

    axes[2].legend()

    # ============================================================
    # SPEED COMPARISON
    # ============================================================

    axes[3].plot(
        time_seconds,
        imu_speed,
        label="IMU speed"
    )

    axes[3].plot(
        time_seconds,
        gps_speed,
        linestyle="--",
        label="GPS speed"
    )

    # ------------------------------------------------------------
    # Stationary markers
    # ------------------------------------------------------------

    indices = np.where(
        stationary
    )[0]

    if len(indices) > 0:

        axes[3].scatter(
            time_seconds[
                indices
            ],
            imu_speed[
                indices
            ],
            s=4,
            label="ZUPT"
        )

    axes[3].set_ylabel(
        "km/h"
    )

    axes[3].set_xlabel(
        "Time (seconds)"
    )

    axes[3].set_title(
        "IMU Speed vs GPS Speed"
    )

    axes[3].grid(
        alpha=0.3
    )

    axes[3].legend()

    # ============================================================
    # Save
    # ============================================================

    PLOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=150
    )

    plt.close()

    print(
        f"Plot saved to: {PLOT_PATH}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "STEP 6"
    )

    print(
        "IMU VELOCITY INTEGRATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nPipeline:"
    )

    print(
        "NED acceleration"
    )

    print(
        "      ↓"
    )

    print(
        "timestamp-based dt"
    )

    print(
        "      ↓"
    )

    print(
        "trapezoidal integration"
    )

    print(
        "      ↓"
    )

    print(
        "raw velocity"
    )

    print(
        "      ↓"
    )

    print(
        "stationary detection"
    )

    print(
        "      ↓"
    )

    print(
        "ZUPT"
    )

    print(
        "      ↓"
    )

    print(
        "corrected velocity"
    )

    print(
        "      ↓"
    )

    print(
        "GPS comparison"
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    ned_df = load_ned_data()

    raw_df = load_raw_data()

    # ============================================================
    # PREPARE TIMESTAMP
    # ============================================================

    ned_df = prepare_timestamp(
        ned_df,
        "NED dataset"
    )

    raw_df = prepare_timestamp(
        raw_df,
        "raw dataset"
    )

    # ============================================================
    # ALIGNMENT
    # ============================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATA ALIGNMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"NED rows : {len(ned_df)}"
    )

    print(
        f"Raw rows : {len(raw_df)}"
    )

    if len(ned_df) != len(raw_df):

        raise ValueError(
            "NED and raw dataset row counts "
            "do not match."
        )

    print(
        "Row alignment : OK"
    )

    # ============================================================
    # TIMESTAMP VALIDATION
    # ============================================================

    print(
        "\nNED timestamp:"
    )

    validate_timestamp_range(
        ned_df
    )

    print(
        "\nRaw timestamp:"
    )

    validate_timestamp_range(
        raw_df
    )

    # ============================================================
    # DT
    # ============================================================

    dt = calculate_dt(
        ned_df
    )

    # ============================================================
    # NED ACCELERATION
    # ============================================================

    acceleration = (
        get_ned_acceleration(
            ned_df
        )
    )

    # ============================================================
    # GPS SPEED
    # ============================================================

    gps_speed = (
        get_gps_speed(
            raw_df
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SENSOR SHAPES"
    )

    print(
        "=" * 70
    )

    print(
        f"NED acceleration : "
        f"{acceleration.shape}"
    )

    print(
        f"GPS speed        : "
        f"{gps_speed.shape}"
    )

    # ============================================================
    # NaN CHECK
    # ============================================================

    invalid_acceleration = (
        ~np.isfinite(
            acceleration
        ).all(
            axis=1
        )
    )

    invalid_gps = (
        ~np.isfinite(
            gps_speed
        )
    )

    print(
        "\nInvalid acceleration rows : "
        f"{np.sum(invalid_acceleration)}"
    )

    print(
        "Invalid GPS speed rows : "
        f"{np.sum(invalid_gps)}"
    )

    if np.any(
        invalid_acceleration
    ):

        print(
            "Replacing invalid acceleration "
            "with zero."
        )

        acceleration[
            invalid_acceleration
        ] = 0.0

    # ============================================================
    # STATIONARY
    # ============================================================

    stationary = (
        detect_stationary(
            acceleration,
            gps_speed
        )
    )

    # ============================================================
    # RAW VELOCITY
    # ============================================================

    raw_velocity = (
        integrate_velocity(
            acceleration,
            dt
        )
    )

    # ============================================================
    # ZUPT
    # ============================================================

    corrected_velocity = (
        apply_zupt(
            raw_velocity,
            stationary
        )
    )

    # ============================================================
    # VALIDATION
    # ============================================================

    (
        raw_speed,
        corrected_speed,
        gps_speed_ms
    ) = validate_velocity(
        raw_velocity,
        corrected_velocity,
        gps_speed,
        stationary
    )

    # ============================================================
    # SAVE
    # ============================================================

    output = save_results(
        ned_df,
        dt,
        raw_velocity,
        corrected_velocity,
        stationary,
        gps_speed
    )

    # ============================================================
    # PLOT
    # ============================================================

    generate_plot(
        output,
        stationary
    )

    # ============================================================
    # COMPLETE
    # ============================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 6 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nIMU velocity integration completed."
    )

    print(
        "Velocity integration : PERFORMED"
    )

    print(
        "ZUPT correction      : PERFORMED"
    )

    print(
        "Position integration : NOT performed"
    )

    print(
        f"\nOutput:"
    )

    print(
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nPlot:"
    )

    print(
        f"{PLOT_PATH}"
    )

    print(
        "\nNext:"
    )

    print(
        "Validate velocity drift before "
        "starting position integration."
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()