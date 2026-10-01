# ============================================================
# orientation_stationary_validation.py
#
# STEP 5.7
#
# Validate orientation during stationary / low-motion periods.
#
# IMPORTANT:
#   - No velocity integration
#   - No position integration
#   - No GNSS trajectory estimation
#
# We evaluate:
#   1. Roll stability
#   2. Pitch stability
#   3. Yaw stability
#   4. NED gravity error
#   5. Horizontal gravity error
#   6. Gravity direction error
#   7. Orientation jumps
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.transform import Rotation


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

# Primary stationary window discovered in Step 5.6
PRIMARY_START = 3133.60
PRIMARY_END = 3141.80

# Maximum acceptable orientation jump between samples
MAX_ORIENTATION_JUMP = 5.0


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("STEP 5.7")
    print("STATIONARY ORIENTATION VALIDATION")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        encoding="cp1252"
    )

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    return df


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
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
# PREPARE TIMESTAMP
# ============================================================

def prepare_timestamp(df):

    date_column = "DATE_YYYY-MO-DD_HH-MI-SS_SSS"

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

    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, prefix):

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

    acceleration_columns = [
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

    magnetometer_columns = [
        find_column(
            df,
            "MAGNETIC_FIELD_X"
        ),
        find_column(
            df,
            "MAGNETIC_FIELD_Y"
        ),
        find_column(
            df,
            "MAGNETIC_FIELD_Z"
        )
    ]

    acceleration = (
        df[
            acceleration_columns
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

    magnetometer = (
        df[
            magnetometer_columns
        ]
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
        magnetometer
    )


# ============================================================
# FIND STATIONARY WINDOWS
#
# Same criteria used in Step 5.6
# ============================================================

def calculate_stationary_windows(
    time,
    acceleration,
    gravity,
    gyro
):

    gravity_magnitude = np.linalg.norm(
        gravity,
        axis=1
    )

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

    gyro_magnitude = np.linalg.norm(
        gyro,
        axis=1
    )

    # GPS speed
    gps_speed_column = find_column(
        df_global,
        "GPS_SPEED"
    )

    gps_speed = (
        pd.to_numeric(
            df_global[
                gps_speed_column
            ],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    gyro_ok = (
        gyro_magnitude < 0.08
    )

    accel_ok = (
        linear_acceleration_magnitude
        < 0.30
    )

    gravity_ok = (
        (gravity_magnitude >= 9.5)
        &
        (gravity_magnitude <= 10.1)
    )

    gps_ok = (
        gps_speed < 0.50
    )

    stationary = (
        gyro_ok
        &
        accel_ok
        &
        gravity_ok
        &
        gps_ok
    )

    windows = []

    start = None

    for i in range(len(stationary)):

        if stationary[i]:

            if start is None:
                start = i

        else:

            if start is not None:

                end = i - 1

                duration = (
                    time[end]
                    - time[start]
                )

                if duration >= 2.0:

                    windows.append(
                        (
                            start,
                            end,
                            duration
                        )
                    )

                start = None

    if start is not None:

        end = len(stationary) - 1

        duration = (
            time[end]
            - time[start]
        )

        if duration >= 2.0:

            windows.append(
                (
                    start,
                    end,
                    duration
                )
            )

    return windows


# ============================================================
# INITIAL ORIENTATION
#
# Gravity determines roll/pitch.
# Magnetometer determines heading.
# ============================================================

def calculate_initial_orientation(
    gravity,
    magnetometer
):

    # Normalize gravity
    g = np.mean(
        gravity,
        axis=0
    )

    g_norm = np.linalg.norm(g)

    g = g / g_norm

    # Normalize magnetic field
    m = np.mean(
        magnetometer,
        axis=0
    )

    m_norm = np.linalg.norm(m)

    m = m / m_norm

    # --------------------------------------------------------
    # Roll / pitch from gravity
    # --------------------------------------------------------

    roll = np.arctan2(
        g[1],
        g[2]
    )

    pitch = np.arctan2(
        -g[0],
        np.sqrt(
            g[1] ** 2
            +
            g[2] ** 2
        )
    )

    # --------------------------------------------------------
    # Tilt compensated magnetometer
    # --------------------------------------------------------

    mx = (
        m[0] * np.cos(pitch)
        +
        m[2] * np.sin(pitch)
    )

    my = (
        m[0] * np.sin(roll)
        * np.sin(pitch)
        +
        m[1] * np.cos(roll)
        -
        m[2] * np.sin(roll)
        * np.cos(pitch)
    )

    yaw = np.arctan2(
        -my,
        mx
    )

    return np.array(
        [
            roll,
            pitch,
            yaw
        ]
    )


# ============================================================
# SIMPLE STATIONARY ORIENTATION
#
# For validation, use the actual measured gravity vector
# rather than integrating gyro drift.
#
# This gives us a reference for roll/pitch.
# ============================================================

def calculate_stationary_orientation(
    gravity,
    magnetometer
):

    n = len(gravity)

    orientation = np.zeros(
        (n, 3)
    )

    for i in range(n):

        g = gravity[i]

        g_norm = np.linalg.norm(g)

        if g_norm < 1e-8:
            continue

        g = g / g_norm

        m = magnetometer[i]

        m_norm = np.linalg.norm(m)

        if m_norm < 1e-8:
            continue

        m = m / m_norm

        # ----------------------------------------------------
        # Roll
        # ----------------------------------------------------

        roll = np.arctan2(
            g[1],
            g[2]
        )

        # ----------------------------------------------------
        # Pitch
        # ----------------------------------------------------

        pitch = np.arctan2(
            -g[0],
            np.sqrt(
                g[1] ** 2
                +
                g[2] ** 2
            )
        )

        # ----------------------------------------------------
        # Tilt compensation
        # ----------------------------------------------------

        mx = (
            m[0] * np.cos(pitch)
            +
            m[2] * np.sin(pitch)
        )

        my = (
            m[0]
            * np.sin(roll)
            * np.sin(pitch)
            +
            m[1]
            * np.cos(roll)
            -
            m[2]
            * np.sin(roll)
            * np.cos(pitch)
        )

        yaw = np.arctan2(
            -my,
            mx
        )

        orientation[i] = [
            np.degrees(roll),
            np.degrees(pitch),
            np.degrees(yaw)
        ]

    return orientation


# ============================================================
# NED GRAVITY FROM ORIENTATION
# ============================================================

def calculate_ned_gravity(
    gravity,
    orientation
):

    n = len(gravity)

    ned_gravity = np.zeros(
        (n, 3)
    )

    for i in range(n):

        roll = np.radians(
            orientation[i, 0]
        )

        pitch = np.radians(
            orientation[i, 1]
        )

        yaw = np.radians(
            orientation[i, 2]
        )

        # ----------------------------------------------------
        # Rotation:
        #
        # body -> NED
        # ----------------------------------------------------

        rotation = Rotation.from_euler(
            "ZYX",
            [
                yaw,
                pitch,
                roll
            ],
            degrees=False
        )

        ned_gravity[i] = (
            rotation.apply(
                gravity[i]
            )
        )

    return ned_gravity


# ============================================================
# ANGLE STATISTICS
# ============================================================

def angle_statistics(
    angles,
    name
):

    mean = np.mean(
        angles
    )

    median = np.median(
        angles
    )

    std = np.std(
        angles
    )

    minimum = np.min(
        angles
    )

    maximum = np.max(
        angles
    )

    print(
        f"\n{name}"
    )

    print(
        f"Mean   : {mean:.4f}°"
    )

    print(
        f"Median : {median:.4f}°"
    )

    print(
        f"Std    : {std:.4f}°"
    )

    print(
        f"Min    : {minimum:.4f}°"
    )

    print(
        f"Max    : {maximum:.4f}°"
    )


# ============================================================
# GRAVITY STATISTICS
# ============================================================

def gravity_statistics(
    ned_gravity
):

    north = ned_gravity[:, 0]
    east = ned_gravity[:, 1]
    down = ned_gravity[:, 2]

    horizontal = np.sqrt(
        north ** 2
        +
        east ** 2
    )

    expected = np.array(
        [
            0.0,
            0.0,
            9.80665
        ]
    )

    measured = ned_gravity

    dot = np.sum(
        measured * expected,
        axis=1
    )

    measured_norm = np.linalg.norm(
        measured,
        axis=1
    )

    expected_norm = np.linalg.norm(
        expected
    )

    cos_angle = (
        dot
        /
        (
            measured_norm
            *
            expected_norm
        )
    )

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )

    angle_error = np.degrees(
        np.arccos(
            cos_angle
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NED GRAVITY VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nExpected:"
    )

    print(
        "North ≈ 0"
    )

    print(
        "East  ≈ 0"
    )

    print(
        "Down  ≈ +9.80665"
    )

    print(
        "\nMeasured:"
    )

    print(
        f"North mean : "
        f"{np.mean(north):.6f}"
    )

    print(
        f"East mean  : "
        f"{np.mean(east):.6f}"
    )

    print(
        f"Down mean  : "
        f"{np.mean(down):.6f}"
    )

    print()

    print(
        f"North std : "
        f"{np.std(north):.6f}"
    )

    print(
        f"East std  : "
        f"{np.std(east):.6f}"
    )

    print(
        f"Down std  : "
        f"{np.std(down):.6f}"
    )

    print()

    print(
        f"Horizontal gravity mean : "
        f"{np.mean(horizontal):.6f}"
    )

    print(
        f"Horizontal gravity median : "
        f"{np.median(horizontal):.6f}"
    )

    print(
        f"Horizontal gravity max : "
        f"{np.max(horizontal):.6f}"
    )

    print()

    print(
        f"Gravity direction mean error : "
        f"{np.mean(angle_error):.6f}°"
    )

    print(
        f"Gravity direction median error : "
        f"{np.median(angle_error):.6f}°"
    )

    print(
        f"Gravity direction max error : "
        f"{np.max(angle_error):.6f}°"
    )

    return (
        north,
        east,
        down,
        horizontal,
        angle_error
    )


# ============================================================
# ORIENTATION JUMP ANALYSIS
# ============================================================

def orientation_jump_analysis(
    orientation,
    time
):

    # Unwrap each angle first
    unwrapped = np.zeros_like(
        orientation
    )

    for axis in range(3):

        unwrapped[:, axis] = np.degrees(
            np.unwrap(
                np.radians(
                    orientation[:, axis]
                )
            )
        )

    delta = np.diff(
        unwrapped,
        axis=0
    )

    jump_magnitude = np.linalg.norm(
        delta,
        axis=1
    )

    large_jumps = (
        jump_magnitude
        >
        MAX_ORIENTATION_JUMP
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ORIENTATION JUMP ANALYSIS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Maximum sample-to-sample "
        f"orientation change : "
        f"{np.max(jump_magnitude):.4f}°"
    )

    print(
        f"Median sample-to-sample "
        f"change : "
        f"{np.median(jump_magnitude):.4f}°"
    )

    print(
        f"Jumps > {MAX_ORIENTATION_JUMP}° : "
        f"{np.sum(large_jumps)}"
    )

    if np.any(large_jumps):

        indices = np.where(
            large_jumps
        )[0]

        print(
            "\nFirst large jumps:"
        )

        for index in indices[:10]:

            print(
                f"Time {time[index]:.3f}s"
                f" -> "
                f"{time[index + 1]:.3f}s"
                f" | "
                f"Jump = "
                f"{jump_magnitude[index]:.3f}°"
            )


# ============================================================
# PRIMARY WINDOW ANALYSIS
# ============================================================

def analyze_primary_window(
    time,
    gravity,
    magnetometer
):

    mask = (
        (time >= PRIMARY_START)
        &
        (time <= PRIMARY_END)
    )

    window_time = time[mask]

    window_gravity = gravity[mask]

    window_magnetometer = (
        magnetometer[mask]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PRIMARY STATIONARY WINDOW"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Start : {window_time[0]:.3f} s"
    )

    print(
        f"End   : {window_time[-1]:.3f} s"
    )

    print(
        f"Samples : {len(window_time)}"
    )

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    orientation = (
        calculate_stationary_orientation(
            window_gravity,
            window_magnetometer
        )
    )

    roll = orientation[:, 0]
    pitch = orientation[:, 1]
    yaw = orientation[:, 2]

    print(
        "\n"
        "Orientation statistics:"
    )

    angle_statistics(
        roll,
        "Roll"
    )

    angle_statistics(
        pitch,
        "Pitch"
    )

    # Yaw has wrap-around, so use circular statistics
    yaw_rad = np.radians(yaw)

    yaw_mean = np.degrees(
        np.arctan2(
            np.mean(
                np.sin(yaw_rad)
            ),
            np.mean(
                np.cos(yaw_rad)
            )
        )
    )

    yaw_error = np.degrees(
        np.arctan2(
            np.sin(
                yaw_rad
                - np.radians(
                    yaw_mean
                )
            ),
            np.cos(
                yaw_rad
                - np.radians(
                    yaw_mean
                )
            )
        )
    )

    print(
        "\nYaw"
    )

    print(
        f"Circular mean : "
        f"{yaw_mean:.4f}°"
    )

    print(
        f"Circular std  : "
        f"{np.std(yaw_error):.4f}°"
    )

    print(
        f"Minimum       : "
        f"{np.min(yaw):.4f}°"
    )

    print(
        f"Maximum       : "
        f"{np.max(yaw):.4f}°"
    )

    # --------------------------------------------------------
    # NED gravity
    # --------------------------------------------------------

    ned_gravity = (
        calculate_ned_gravity(
            window_gravity,
            orientation
        )
    )

    (
        north,
        east,
        down,
        horizontal,
        angle_error
    ) = gravity_statistics(
        ned_gravity
    )

    return (
        window_time,
        orientation,
        ned_gravity
    )


# ============================================================
# PLOT PRIMARY WINDOW
# ============================================================

def plot_primary_window(
    time,
    orientation,
    ned_gravity
):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True
    )

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    axes[0].plot(
        time,
        orientation[:, 0],
        label="Roll",
        linewidth=1
    )

    axes[0].plot(
        time,
        orientation[:, 1],
        label="Pitch",
        linewidth=1
    )

    axes[0].plot(
        time,
        orientation[:, 2],
        label="Yaw",
        linewidth=1
    )

    axes[0].set_ylabel(
        "Angle (degrees)"
    )

    axes[0].set_title(
        "Stationary Orientation Validation"
    )

    axes[0].legend()

    axes[0].grid(True)

    # --------------------------------------------------------
    # NED gravity
    # --------------------------------------------------------

    axes[1].plot(
        time,
        ned_gravity[:, 0],
        label="North gravity",
        linewidth=1
    )

    axes[1].plot(
        time,
        ned_gravity[:, 1],
        label="East gravity",
        linewidth=1
    )

    axes[1].plot(
        time,
        ned_gravity[:, 2],
        label="Down gravity",
        linewidth=1
    )

    axes[1].axhline(
        9.80665,
        linestyle="--"
    )

    axes[1].axhline(
        0.0,
        linestyle="--"
    )

    axes[1].set_xlabel(
        "Time (seconds)"
    )

    axes[1].set_ylabel(
        "Gravity (m/s²)"
    )

    axes[1].set_title(
        "Stationary NED Gravity"
    )

    axes[1].legend()

    axes[1].grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # Global reference used by stationary window function
    df_global = clean_column_names(
        df
    )

    df_global = prepare_timestamp(
        df_global
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    (
        time,
        acceleration,
        gravity,
        gyro,
        magnetometer
    ) = extract_sensors(
        df_global
    )

    # --------------------------------------------------------
    # Validate timestamps
    # --------------------------------------------------------

    print(
        "\nTimestamp validation:"
    )

    print(
        f"Start : "
        f"{df_global['timestamp'].iloc[0]}"
    )

    print(
        f"End   : "
        f"{df_global['timestamp'].iloc[-1]}"
    )

    dt = np.diff(time)

    positive_dt = dt[
        dt > 0
    ]

    print(
        f"Median dt : "
        f"{np.median(positive_dt):.4f} s"
    )

    print(
        f"Mean dt   : "
        f"{np.mean(positive_dt):.4f} s"
    )

    # --------------------------------------------------------
    # Stationary windows
    # --------------------------------------------------------

    windows = calculate_stationary_windows(
        time,
        acceleration,
        gravity,
        gyro
    )

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

    print(
        f"\nNumber of windows : "
        f"{len(windows)}"
    )

    for i, (
        start,
        end,
        duration
    ) in enumerate(
        windows[:20],
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{time[start]:.2f}s -> "
            f"{time[end]:.2f}s "
            f"({duration:.2f}s)"
        )

    # --------------------------------------------------------
    # Primary stationary window
    # --------------------------------------------------------

    (
        primary_time,
        primary_orientation,
        primary_ned_gravity
    ) = analyze_primary_window(
        time,
        gravity,
        magnetometer
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    print(
        "\nGenerating validation plot..."
    )

    plot_primary_window(
        primary_time,
        primary_orientation,
        primary_ned_gravity
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 5.7 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nStationary orientation validation completed."
    )

    print(
        "Velocity integration : NOT performed"
    )

    print(
        "Position integration : NOT performed"
    )