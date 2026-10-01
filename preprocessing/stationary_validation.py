# ============================================================
# stationary_validation.py
#
# STEP 5.6
# STATIONARY / LOW-MOTION VALIDATION
#
# Purpose:
#   Find stationary and low-motion portions of IO-VNBD
#   before attempting velocity or position integration.
#
# IMPORTANT:
#   This script DOES NOT integrate velocity.
#   This script DOES NOT integrate position.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

# ------------------------------------------------------------
# Stationary thresholds
# ------------------------------------------------------------

# Gyroscope magnitude
GYRO_STATIONARY_THRESHOLD = 0.08       # rad/s

# Linear acceleration magnitude
LINEAR_ACCEL_THRESHOLD = 0.30          # m/s²

# GPS speed
GPS_SPEED_THRESHOLD = 0.50             # km/h

# Gravity magnitude tolerance
GRAVITY_MIN = 9.5
GRAVITY_MAX = 10.1

# Minimum continuous stationary duration
MIN_STATIONARY_DURATION = 2.0          # seconds

# Maximum allowed timestamp gap
MAX_DT = 0.20


# ============================================================
# COLUMN CLEANING
# ============================================================

def clean_column_names(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(
            " ",
            "_"
        )
        .str.replace(
            "(",
            "",
            regex=False
        )
        .str.replace(
            ")",
            "",
            regex=False
        )
        .str.replace(
            "/",
            "_",
            regex=False
        )
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATIONARY / LOW-MOTION VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nLoading dataset..."
    )

    df = pd.read_csv(
        DATA_PATH,
        encoding="cp1252"
    )

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = clean_column_names(df)

    date_column = (
        "DATE_YYYY-MO-DD_HH-MI-SS_SSS"
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df[date_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamps detected."
        )

    df["time_seconds"] = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    # --------------------------------------------------------
    # Sampling interval
    # --------------------------------------------------------

    dt = np.diff(
        df["time_seconds"].to_numpy()
    )

    positive_dt = dt[
        dt > 0
    ]

    print(
        "\nTimestamp validation:"
    )

    print(
        f"Start : {df['timestamp'].iloc[0]}"
    )

    print(
        f"End   : {df['timestamp'].iloc[-1]}"
    )

    print(
        f"Duration : "
        f"{df['time_seconds'].iloc[-1]:.3f} s"
    )

    print()

    print(
        f"Median dt : "
        f"{np.median(positive_dt):.4f} s"
    )

    print(
        f"Mean dt   : "
        f"{np.mean(positive_dt):.4f} s"
    )

    print(
        f"Max dt    : "
        f"{np.max(positive_dt):.4f} s"
    )

    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    prefix
):

    matches = [
        column
        for column in df.columns
        if column.startswith(prefix)
    ]

    if not matches:

        raise KeyError(
            f"Column not found: {prefix}"
        )

    return matches[0]


# ============================================================
# EXTRACT SENSORS
# ============================================================

def extract_sensors(df):

    accel_columns = [
        find_column(
            df,
            "ACCELEROMETER_X"
        ),
        find_column(
            df,
            "ACCELEROMETER_Y"
        ),
        find_column(
            df,
            "ACCELEROMETER_Z"
        )
    ]

    gravity_columns = [
        find_column(
            df,
            "GRAVITY_X"
        ),
        find_column(
            df,
            "GRAVITY_Y"
        ),
        find_column(
            df,
            "GRAVITY_Z"
        )
    ]

    gyro_columns = [
        find_column(
            df,
            "GYROSCOPE_Roll"
        ),
        find_column(
            df,
            "GYROSCOPE_Pitch"
        ),
        find_column(
            df,
            "GYROSCOPE_Yaw"
        )
    ]

    # --------------------------------------------------------
    # GPS speed
    # --------------------------------------------------------

    gps_speed_column = find_column(
        df,
        "GPS_SPEED"
    )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    acceleration = (
        df[
            accel_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    gravity = (
        df[
            gravity_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    gyro = (
        df[
            gyro_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    gps_speed = (
        pd.to_numeric(
            df[gps_speed_column],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    time = (
        df["time_seconds"]
        .to_numpy(
            dtype=float
        )
    )

    return (
        time,
        acceleration,
        gravity,
        gyro,
        gps_speed
    )


# ============================================================
# CALCULATE SENSOR MAGNITUDES
# ============================================================

def calculate_magnitudes(
    acceleration,
    gravity,
    gyro
):

    # --------------------------------------------------------
    # Gravity magnitude
    # --------------------------------------------------------

    gravity_magnitude = (
        np.linalg.norm(
            gravity,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Raw acceleration magnitude
    # --------------------------------------------------------

    acceleration_magnitude = (
        np.linalg.norm(
            acceleration,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Linear acceleration
    #
    # Accelerometer ≈ specific force + gravity
    #
    # Dataset provides gravity separately.
    #
    # For stationary validation:
    #
    # linear = acceleration - gravity
    # --------------------------------------------------------

    linear_acceleration = (
        acceleration
        - gravity
    )

    linear_acceleration_magnitude = (
        np.linalg.norm(
            linear_acceleration,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Gyroscope magnitude
    # --------------------------------------------------------

    gyro_magnitude = (
        np.linalg.norm(
            gyro,
            axis=1
        )
    )

    return (
        gravity_magnitude,
        acceleration_magnitude,
        linear_acceleration,
        linear_acceleration_magnitude,
        gyro_magnitude
    )


# ============================================================
# CREATE STATIONARY MASK
# ============================================================

def create_stationary_mask(
    gyro_magnitude,
    linear_acceleration_magnitude,
    gravity_magnitude,
    gps_speed
):

    gyro_ok = (
        gyro_magnitude
        < GYRO_STATIONARY_THRESHOLD
    )

    linear_accel_ok = (
        linear_acceleration_magnitude
        < LINEAR_ACCEL_THRESHOLD
    )

    gravity_ok = (
        (gravity_magnitude >= GRAVITY_MIN)
        &
        (gravity_magnitude <= GRAVITY_MAX)
    )

    gps_ok = (
        gps_speed
        < GPS_SPEED_THRESHOLD
    )

    stationary_mask = (
        gyro_ok
        &
        linear_accel_ok
        &
        gravity_ok
        &
        gps_ok
    )

    return (
        stationary_mask,
        gyro_ok,
        linear_accel_ok,
        gravity_ok,
        gps_ok
    )


# ============================================================
# FIND CONTINUOUS STATIONARY WINDOWS
# ============================================================

def find_stationary_windows(
    time,
    stationary_mask
):

    windows = []

    start_index = None

    for i in range(
        len(stationary_mask)
    ):

        if stationary_mask[i]:

            if start_index is None:

                start_index = i

        else:

            if start_index is not None:

                end_index = i - 1

                duration = (
                    time[end_index]
                    - time[start_index]
                )

                if (
                    duration
                    >= MIN_STATIONARY_DURATION
                ):

                    windows.append(
                        (
                            start_index,
                            end_index,
                            duration
                        )
                    )

                start_index = None

    # --------------------------------------------------------
    # Handle final window
    # --------------------------------------------------------

    if start_index is not None:

        end_index = (
            len(stationary_mask)
            - 1
        )

        duration = (
            time[end_index]
            - time[start_index]
        )

        if (
            duration
            >= MIN_STATIONARY_DURATION
        ):

            windows.append(
                (
                    start_index,
                    end_index,
                    duration
                )
            )

    return windows


# ============================================================
# REPORT STATIONARY WINDOWS
# ============================================================

def report_windows(
    windows,
    time,
    gyro_magnitude,
    linear_acceleration_magnitude,
    gravity_magnitude,
    gps_speed
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATIONARY WINDOWS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Number of stationary windows : "
        f"{len(windows)}"
    )

    if len(windows) == 0:

        print(
            "\nNo stationary windows found."
        )

        print(
            "We may need to relax the thresholds."
        )

        return

    # --------------------------------------------------------
    # Sort by duration
    # --------------------------------------------------------

    windows_sorted = sorted(
        windows,
        key=lambda x: x[2],
        reverse=True
    )

    print()

    print(
        "Longest stationary windows:"
    )

    print()

    print(
        f"{'Start(s)':>12}"
        f"{'End(s)':>12}"
        f"{'Duration':>12}"
        f"{'Gyro':>12}"
        f"{'LinAccel':>12}"
        f"{'Gravity':>12}"
        f"{'GPS km/h':>12}"
    )

    print(
        "-" * 84
    )

    for (
        start,
        end,
        duration
    ) in windows_sorted[:20]:

        print(
            f"{time[start]:12.2f}"
            f"{time[end]:12.2f}"
            f"{duration:12.2f}"
            f"{np.mean(gyro_magnitude[start:end+1]):12.4f}"
            f"{np.mean(linear_acceleration_magnitude[start:end+1]):12.4f}"
            f"{np.mean(gravity_magnitude[start:end+1]):12.4f}"
            f"{np.mean(gps_speed[start:end+1]):12.4f}"
        )


# ============================================================
# VALIDATE BEST STATIONARY WINDOW
# ============================================================

def validate_best_window(
    windows,
    time,
    acceleration,
    gravity,
    gyro,
    gps_speed,
    gravity_magnitude,
    linear_acceleration_magnitude,
    gyro_magnitude
):

    if len(windows) == 0:

        return

    # --------------------------------------------------------
    # Longest window
    # --------------------------------------------------------

    start, end, duration = max(
        windows,
        key=lambda x: x[2]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST STATIONARY WINDOW"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Start    : {time[start]:.3f} s"
    )

    print(
        f"End      : {time[end]:.3f} s"
    )

    print(
        f"Duration : {duration:.3f} s"
    )

    # --------------------------------------------------------
    # Slice
    # --------------------------------------------------------

    g = gravity[
        start:end + 1
    ]

    a = acceleration[
        start:end + 1
    ]

    gyro_segment = gyro[
        start:end + 1
    ]

    gps_segment = gps_speed[
        start:end + 1
    ]

    # --------------------------------------------------------
    # Means
    # --------------------------------------------------------

    gravity_mean = np.mean(
        g,
        axis=0
    )

    acceleration_mean = np.mean(
        a,
        axis=0
    )

    gyro_mean = np.mean(
        gyro_segment,
        axis=0
    )

    # --------------------------------------------------------
    # Standard deviations
    # --------------------------------------------------------

    gravity_std = np.std(
        g,
        axis=0
    )

    acceleration_std = np.std(
        a,
        axis=0
    )

    gyro_std = np.std(
        gyro_segment,
        axis=0
    )

    print(
        "\nGravity mean:"
    )

    print(
        gravity_mean
    )

    print(
        "\nGravity std:"
    )

    print(
        gravity_std
    )

    print(
        "\nGravity magnitude:"
    )

    print(
        f"Mean   : "
        f"{np.mean(gravity_magnitude[start:end+1]):.6f}"
    )

    print(
        f"Std    : "
        f"{np.std(gravity_magnitude[start:end+1]):.6f}"
    )

    print(
        "\nLinear acceleration:"
    )

    print(
        f"Mean magnitude : "
        f"{np.mean(linear_acceleration_magnitude[start:end+1]):.6f}"
    )

    print(
        f"Std magnitude  : "
        f"{np.std(linear_acceleration_magnitude[start:end+1]):.6f}"
    )

    print(
        "\nGyroscope mean:"
    )

    print(
        gyro_mean
    )

    print(
        "\nGyroscope std:"
    )

    print(
        gyro_std
    )

    print(
        "\nGPS speed:"
    )

    print(
        f"Mean : "
        f"{np.mean(gps_segment):.4f} km/h"
    )

    print(
        f"Max  : "
        f"{np.max(gps_segment):.4f} km/h"
    )

    # --------------------------------------------------------
    # Stationary quality score
    # --------------------------------------------------------

    gravity_error = abs(
        np.mean(
            gravity_magnitude[
                start:end + 1
            ]
        )
        - 9.80665
    )

    gyro_score = np.mean(
        gyro_magnitude[
            start:end + 1
        ]
    )

    linear_score = np.mean(
        linear_acceleration_magnitude[
            start:end + 1
        ]
    )

    print(
        "\nStationary quality:"
    )

    print(
        f"Gravity error : "
        f"{gravity_error:.6f} m/s²"
    )

    print(
        f"Gyro motion   : "
        f"{gyro_score:.6f} rad/s"
    )

    print(
        f"Linear motion : "
        f"{linear_score:.6f} m/s²"
    )


# ============================================================
# PLOT STATIONARY DETECTION
# ============================================================

def plot_stationary_detection(
    time,
    gyro_magnitude,
    linear_acceleration_magnitude,
    gravity_magnitude,
    gps_speed,
    stationary_mask
):

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 12),
        sharex=True
    )

    # --------------------------------------------------------
    # Gyroscope
    # --------------------------------------------------------

    axes[0].plot(
        time,
        gyro_magnitude,
        linewidth=0.5
    )

    axes[0].axhline(
        GYRO_STATIONARY_THRESHOLD,
        linestyle="--"
    )

    axes[0].set_ylabel(
        "Gyro (rad/s)"
    )

    axes[0].set_title(
        "Gyroscope Magnitude"
    )

    axes[0].grid(True)

    # --------------------------------------------------------
    # Linear acceleration
    # --------------------------------------------------------

    axes[1].plot(
        time,
        linear_acceleration_magnitude,
        linewidth=0.5
    )

    axes[1].axhline(
        LINEAR_ACCEL_THRESHOLD,
        linestyle="--"
    )

    axes[1].set_ylabel(
        "Linear Accel (m/s²)"
    )

    axes[1].set_title(
        "Linear Acceleration Magnitude"
    )

    axes[1].grid(True)

    # --------------------------------------------------------
    # Gravity
    # --------------------------------------------------------

    axes[2].plot(
        time,
        gravity_magnitude,
        linewidth=0.5
    )

    axes[2].axhline(
        9.80665,
        linestyle="--"
    )

    axes[2].set_ylabel(
        "Gravity (m/s²)"
    )

    axes[2].set_title(
        "Gravity Magnitude"
    )

    axes[2].grid(True)

    # --------------------------------------------------------
    # GPS speed + stationary
    # --------------------------------------------------------

    axes[3].plot(
        time,
        gps_speed,
        linewidth=0.5,
        label="GPS speed"
    )

    axes[3].axhline(
        GPS_SPEED_THRESHOLD,
        linestyle="--",
        label="Stationary threshold"
    )

    stationary_time = time[
        stationary_mask
    ]

    stationary_speed = gps_speed[
        stationary_mask
    ]

    axes[3].scatter(
        stationary_time,
        stationary_speed,
        s=2,
        label="Detected stationary"
    )

    axes[3].set_ylabel(
        "GPS speed (km/h)"
    )

    axes[3].set_xlabel(
        "Time (seconds)"
    )

    axes[3].set_title(
        "GPS Speed / Stationary Detection"
    )

    axes[3].legend()

    axes[3].grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    df = prepare_data(
        df
    )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    (
        time,
        acceleration,
        gravity,
        gyro,
        gps_speed
    ) = extract_sensors(
        df
    )

    # --------------------------------------------------------
    # SENSOR MAGNITUDES
    # --------------------------------------------------------

    (
        gravity_magnitude,
        acceleration_magnitude,
        linear_acceleration,
        linear_acceleration_magnitude,
        gyro_magnitude
    ) = calculate_magnitudes(
        acceleration,
        gravity,
        gyro
    )

    # --------------------------------------------------------
    # RAW SENSOR SUMMARY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAW SENSOR SUMMARY"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Gravity magnitude median : "
        f"{np.median(gravity_magnitude):.4f}"
    )

    print(
        f"Gravity magnitude std    : "
        f"{np.std(gravity_magnitude):.4f}"
    )

    print(
        f"Linear acceleration median : "
        f"{np.median(linear_acceleration_magnitude):.4f}"
    )

    print(
        f"Gyroscope median : "
        f"{np.median(gyro_magnitude):.4f}"
    )

    print(
        f"GPS speed median : "
        f"{np.median(gps_speed):.4f} km/h"
    )

    # --------------------------------------------------------
    # STATIONARY MASK
    # --------------------------------------------------------

    (
        stationary_mask,
        gyro_ok,
        linear_accel_ok,
        gravity_ok,
        gps_ok
    ) = create_stationary_mask(
        gyro_magnitude,
        linear_acceleration_magnitude,
        gravity_magnitude,
        gps_speed
    )

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

    print()

    print(
        f"Gyro condition      : "
        f"{np.sum(gyro_ok)} samples"
    )

    print(
        f"Linear accel        : "
        f"{np.sum(linear_accel_ok)} samples"
    )

    print(
        f"Gravity condition   : "
        f"{np.sum(gravity_ok)} samples"
    )

    print(
        f"GPS speed condition : "
        f"{np.sum(gps_ok)} samples"
    )

    print(
        f"All conditions      : "
        f"{np.sum(stationary_mask)} samples"
    )

    percentage = (
        100.0
        * np.mean(
            stationary_mask
        )
    )

    print(
        f"Stationary percentage : "
        f"{percentage:.2f}%"
    )

    # --------------------------------------------------------
    # FIND WINDOWS
    # --------------------------------------------------------

    windows = find_stationary_windows(
        time,
        stationary_mask
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report_windows(
        windows,
        time,
        gyro_magnitude,
        linear_acceleration_magnitude,
        gravity_magnitude,
        gps_speed
    )

    # --------------------------------------------------------
    # BEST WINDOW
    # --------------------------------------------------------

    validate_best_window(
        windows,
        time,
        acceleration,
        gravity,
        gyro,
        gps_speed,
        gravity_magnitude,
        linear_acceleration_magnitude,
        gyro_magnitude
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    print(
        "\nGenerating stationary validation plot..."
    )

    plot_stationary_detection(
        time,
        gyro_magnitude,
        linear_acceleration_magnitude,
        gravity_magnitude,
        gps_speed,
        stationary_mask
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 5.6 COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Stationary / low-motion validation completed."
    )

    print(
        "Velocity integration was NOT performed."
    )

    print(
        "Position integration was NOT performed."
    )