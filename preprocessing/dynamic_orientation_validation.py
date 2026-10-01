import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.transform import Rotation


# ============================================================
# STEP 5.10 V2
# DYNAMIC QUATERNION ORIENTATION VALIDATION
# ============================================================

print("=" * 70)
print("STEP 5.10 V2")
print("DYNAMIC QUATERNION ORIENTATION VALIDATION")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

ENCODING = "cp1252"

MAX_DT = 0.25

# Stationary window from Step 5.8
BIAS_START = 3133.601
BIAS_END = 3141.700

# Dynamic validation windows
WINDOWS = [
    (0, 30),
    (1000, 1030),
    (2500, 2530),
    (4400, 4430),
    (5000, 5030),
    (7500, 7530),
    (10000, 10030),
]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        encoding=ENCODING
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
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df


# ============================================================
# FIND EXACT SENSOR AXIS
# ============================================================

def find_axis_column(df, sensor, axis):

    """
    Find an exact sensor-axis column.

    Example:

        GRAVITY + X
        -> GRAVITY_X_m_s²

    This avoids accidentally matching the 'Y' inside
    the word 'GRAVITY'.
    """

    prefix = f"{sensor}_{axis}_"

    candidates = [
        column
        for column in df.columns
        if column.startswith(prefix)
    ]

    if len(candidates) == 0:

        print(
            f"\nERROR: Could not find "
            f"{sensor}_{axis}"
        )

        print("\nAvailable columns:")

        for column in df.columns:
            print(f"  {column}")

        raise KeyError(
            f"Missing column: {sensor}_{axis}"
        )

    return candidates[0]


# ============================================================
# FIND GPS SPEED
# ============================================================

def find_gps_speed_column(df):

    candidates = [
        column
        for column in df.columns
        if column.startswith("GPS_SPEED_")
    ]

    if len(candidates) == 0:

        raise KeyError(
            "GPS speed column not found."
        )

    return candidates[0]


# ============================================================
# PREPARE TIMESTAMP
# ============================================================

def prepare_timestamp(df):

    df = df.copy()

    date_candidates = [
        column
        for column in df.columns
        if column.startswith("DATE_")
    ]

    if len(date_candidates) == 0:

        raise KeyError(
            "DATE column not found."
        )

    date_column = date_candidates[0]

    df["timestamp"] = pd.to_datetime(
        df[date_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    invalid = (
        df["timestamp"].isna().sum()
    )

    print("\nTimestamp validation:")

    print(
        f"Invalid timestamps : {invalid}"
    )

    if invalid > 0:

        raise ValueError(
            "Invalid timestamps detected."
        )

    df["time_seconds"] = (
        df["timestamp"]
        -
        df["timestamp"].iloc[0]
    ).dt.total_seconds()

    return df


# ============================================================
# GET SENSOR COLUMNS
# ============================================================

def get_sensor_columns(df):

    # --------------------------------------------------------
    # Accelerometer
    # --------------------------------------------------------

    accel_x = find_axis_column(
        df,
        "ACCELEROMETER",
        "X"
    )

    accel_y = find_axis_column(
        df,
        "ACCELEROMETER",
        "Y"
    )

    accel_z = find_axis_column(
        df,
        "ACCELEROMETER",
        "Z"
    )

    # --------------------------------------------------------
    # Gravity
    # --------------------------------------------------------

    gravity_x = find_axis_column(
        df,
        "GRAVITY",
        "X"
    )

    gravity_y = find_axis_column(
        df,
        "GRAVITY",
        "Y"
    )

    gravity_z = find_axis_column(
        df,
        "GRAVITY",
        "Z"
    )

    # --------------------------------------------------------
    # Magnetometer
    # --------------------------------------------------------

    mag_x = find_axis_column(
        df,
        "MAGNETIC_FIELD",
        "X"
    )

    mag_y = find_axis_column(
        df,
        "MAGNETIC_FIELD",
        "Y"
    )

    mag_z = find_axis_column(
        df,
        "MAGNETIC_FIELD",
        "Z"
    )

    # --------------------------------------------------------
    # Gyroscope
    #
    # Dataset gives:
    #
    #   Yaw
    #   Pitch
    #   Roll
    #
    # Navigation rotation vector must become:
    #
    #   X = Roll
    #   Y = Pitch
    #   Z = Yaw
    # --------------------------------------------------------

    gyro_roll = find_axis_column(
        df,
        "GYROSCOPE",
        "Roll"
    )

    gyro_pitch = find_axis_column(
        df,
        "GYROSCOPE",
        "Pitch"
    )

    gyro_yaw = find_axis_column(
        df,
        "GYROSCOPE",
        "Yaw"
    )

    gps_speed = find_gps_speed_column(
        df
    )

    print("\nSensor columns detected:")

    print(
        f"Accelerometer X : {accel_x}"
    )

    print(
        f"Accelerometer Y : {accel_y}"
    )

    print(
        f"Accelerometer Z : {accel_z}"
    )

    print(
        f"Gravity X       : {gravity_x}"
    )

    print(
        f"Gravity Y       : {gravity_y}"
    )

    print(
        f"Gravity Z       : {gravity_z}"
    )

    print(
        f"Gyroscope X     : {gyro_roll}"
    )

    print(
        f"Gyroscope Y     : {gyro_pitch}"
    )

    print(
        f"Gyroscope Z     : {gyro_yaw}"
    )

    print(
        f"Magnetometer X  : {mag_x}"
    )

    print(
        f"Magnetometer Y  : {mag_y}"
    )

    print(
        f"Magnetometer Z  : {mag_z}"
    )

    print(
        f"GPS Speed       : {gps_speed}"
    )

    return {
        "accel": [
            accel_x,
            accel_y,
            accel_z
        ],

        "gravity": [
            gravity_x,
            gravity_y,
            gravity_z
        ],

        # IMPORTANT:
        # X = Roll
        # Y = Pitch
        # Z = Yaw
        "gyro": [
            gyro_roll,
            gyro_pitch,
            gyro_yaw
        ],

        "magnetometer": [
            mag_x,
            mag_y,
            mag_z
        ],

        "gps_speed": gps_speed
    }


# ============================================================
# EXTRACT SENSOR DATA
# ============================================================

def get_sensor_data(
    df,
    columns
):

    time = df[
        "time_seconds"
    ].to_numpy(
        dtype=float
    )

    accel = df[
        columns["accel"]
    ].to_numpy(
        dtype=float
    )

    gravity = df[
        columns["gravity"]
    ].to_numpy(
        dtype=float
    )

    gyro = df[
        columns["gyro"]
    ].to_numpy(
        dtype=float
    )

    magnetometer = df[
        columns["magnetometer"]
    ].to_numpy(
        dtype=float
    )

    gps_speed = df[
        columns["gps_speed"]
    ].to_numpy(
        dtype=float
    )

    return (
        time,
        accel,
        gravity,
        gyro,
        magnetometer,
        gps_speed
    )


# ============================================================
# ESTIMATE GYRO BIAS
# ============================================================

def estimate_gyro_bias(
    time,
    gyro
):

    mask = (
        (time >= BIAS_START)
        &
        (time <= BIAS_END)
    )

    if np.sum(mask) < 10:

        raise RuntimeError(
            "Not enough stationary "
            "samples for gyro calibration."
        )

    bias = np.mean(
        gyro[mask],
        axis=0
    )

    return bias


# ============================================================
# CLEAN GYRO
# ============================================================

def clean_gyro(
    gyro
):

    gyro = gyro.copy()

    magnitude = np.linalg.norm(
        gyro,
        axis=1
    )

    median = np.median(
        magnitude
    )

    mad = np.median(
        np.abs(
            magnitude - median
        )
    )

    robust_sigma = (
        1.4826 * mad
    )

    threshold = (
        median
        +
        7.0 * robust_sigma
    )

    outlier_mask = (
        magnitude > threshold
    )

    print("\nGyroscope preprocessing:")

    print(
        f"Median magnitude : "
        f"{median:.6f} rad/s"
    )

    print(
        f"MAD              : "
        f"{mad:.6f}"
    )

    print(
        f"Threshold        : "
        f"{threshold:.6f} rad/s"
    )

    print(
        f"Outliers         : "
        f"{np.sum(outlier_mask)}"
    )

    # Interpolate each axis
    # over detected outliers.

    for axis in range(3):

        series = pd.Series(
            gyro[:, axis]
        )

        series[outlier_mask] = np.nan

        series = series.interpolate(
            method="linear",
            limit_direction="both"
        )

        gyro[:, axis] = (
            series.to_numpy()
        )

    return gyro


# ============================================================
# INITIAL ORIENTATION FROM GRAVITY
# ============================================================

def initial_orientation(
    gravity,
    magnetometer
):

    # --------------------------------------------------------
    # Average first samples
    # --------------------------------------------------------

    gravity_mean = np.mean(
        gravity[:50],
        axis=0
    )

    gravity_norm = np.linalg.norm(
        gravity_mean
    )

    if gravity_norm < 1e-8:

        raise RuntimeError(
            "Invalid gravity vector."
        )

    g = (
        gravity_mean
        /
        gravity_norm
    )

    # --------------------------------------------------------
    # Roll and pitch from gravity
    #
    # NED convention:
    #
    # North = X
    # East  = Y
    # Down  = Z
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
    # Magnetometer
    # --------------------------------------------------------

    mag_mean = np.mean(
        magnetometer[:50],
        axis=0
    )

    mag_norm = np.linalg.norm(
        mag_mean
    )

    if mag_norm < 1e-8:

        raise RuntimeError(
            "Invalid magnetometer vector."
        )

    m = (
        mag_mean
        /
        mag_norm
    )

    mx = m[0]
    my = m[1]
    mz = m[2]

    # --------------------------------------------------------
    # Tilt compensated magnetic heading
    # --------------------------------------------------------

    mx_level = (
        mx * np.cos(pitch)
        +
        mz * np.sin(pitch)
    )

    my_level = (
        mx * np.sin(roll)
        *
        np.sin(pitch)
        +
        my * np.cos(roll)
        -
        mz * np.sin(roll)
        *
        np.cos(pitch)
    )

    yaw = np.arctan2(
        -my_level,
        mx_level
    )

    rotation = Rotation.from_euler(
        "ZYX",
        [
            yaw,
            pitch,
            roll
        ]
    )

    return rotation


# ============================================================
# QUATERNION PROPAGATION
# ============================================================

def propagate_quaternion(
    gyro,
    time,
    initial_rotation
):

    n = len(time)

    rotations = [
        initial_rotation
    ]

    current = (
        initial_rotation
    )

    for i in range(1, n):

        dt = (
            time[i]
            -
            time[i - 1]
        )

        # ----------------------------------------------------
        # Protect against timestamp gaps
        # ----------------------------------------------------

        if (
            dt <= 0
            or
            dt > MAX_DT
        ):

            rotations.append(
                current
            )

            continue

        omega = gyro[i]

        rotvec = (
            omega * dt
        )

        delta = (
            Rotation.from_rotvec(
                rotvec
            )
        )

        current = (
            current
            *
            delta
        )

        rotations.append(
            current
        )

    return Rotation.concatenate(
        rotations
    )


# ============================================================
# EULER CONVERSION
# ============================================================

def rotations_to_euler(
    rotations
):

    euler = rotations.as_euler(
        "ZYX",
        degrees=True
    )

    yaw = euler[:, 0]
    pitch = euler[:, 1]
    roll = euler[:, 2]

    return (
        roll,
        pitch,
        yaw
    )


# ============================================================
# CALCULATE NED GRAVITY
# ============================================================

def calculate_ned_gravity(
    rotations
):

    g_body = np.array([
        0.0,
        0.0,
        9.80665
    ])

    gravity_ned = np.zeros(
        (
            len(rotations),
            3
        )
    )

    for i in range(
        len(rotations)
    ):

        gravity_ned[i] = (
            rotations[i]
            .apply(
                g_body
            )
        )

    return gravity_ned


# ============================================================
# ANALYZE ONE WINDOW
# ============================================================

def analyze_window(
    time,
    accel,
    gravity,
    gyro,
    magnetometer,
    gps_speed,
    start,
    end,
    gyro_bias
):

    mask = (
        (time >= start)
        &
        (time <= end)
    )

    indices = np.where(
        mask
    )[0]

    if len(indices) < 20:

        return None

    local_time = time[
        indices
    ]

    local_accel = accel[
        indices
    ]

    local_gravity = gravity[
        indices
    ]

    local_gyro = (
        gyro[indices]
        -
        gyro_bias
    )

    local_mag = (
        magnetometer[
            indices
        ]
    )

    local_speed = (
        gps_speed[
            indices
        ]
    )

    # --------------------------------------------------------
    # Initial orientation
    # --------------------------------------------------------

    initial_rotation = (
        initial_orientation(
            local_gravity,
            local_mag
        )
    )

    # --------------------------------------------------------
    # Quaternion propagation
    # --------------------------------------------------------

    rotations = (
        propagate_quaternion(
            local_gyro,
            local_time,
            initial_rotation
        )
    )

    roll, pitch, yaw = (
        rotations_to_euler(
            rotations
        )
    )

    # --------------------------------------------------------
    # NED gravity
    # --------------------------------------------------------

    gravity_ned = (
        calculate_ned_gravity(
            rotations
        )
    )

    gravity_magnitude = (
        np.linalg.norm(
            gravity_ned,
            axis=1
        )
    )

    horizontal_gravity = np.sqrt(
        gravity_ned[:, 0] ** 2
        +
        gravity_ned[:, 1] ** 2
    )

    gravity_direction_error = (
        np.degrees(
            np.arccos(
                np.clip(
                    gravity_ned[:, 2]
                    /
                    np.maximum(
                        gravity_magnitude,
                        1e-12
                    ),
                    -1.0,
                    1.0
                )
            )
        )
    )

    # --------------------------------------------------------
    # Linear acceleration
    # --------------------------------------------------------

    linear_accel = (
        local_accel
        -
        local_gravity
    )

    linear_accel_magnitude = (
        np.linalg.norm(
            linear_accel,
            axis=1
        )
    )

    gyro_magnitude = (
        np.linalg.norm(
            local_gyro,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"DYNAMIC WINDOW "
        f"{start:.1f}s -> {end:.1f}s"
    )

    print("=" * 70)

    print(
        f"Samples : {len(indices)}"
    )

    print(
        f"GPS speed mean : "
        f"{np.mean(local_speed):.3f} km/h"
    )

    print(
        f"GPS speed max  : "
        f"{np.max(local_speed):.3f} km/h"
    )

    print("\nORIENTATION")

    print(
        f"Roll range  : "
        f"{np.min(roll):.2f}° "
        f"to {np.max(roll):.2f}°"
    )

    print(
        f"Pitch range : "
        f"{np.min(pitch):.2f}° "
        f"to {np.max(pitch):.2f}°"
    )

    print(
        f"Yaw range   : "
        f"{np.min(yaw):.2f}° "
        f"to {np.max(yaw):.2f}°"
    )

    print("\nNED GRAVITY")

    print(
        f"North mean : "
        f"{np.mean(gravity_ned[:, 0]):.4f}"
    )

    print(
        f"East mean  : "
        f"{np.mean(gravity_ned[:, 1]):.4f}"
    )

    print(
        f"Down mean  : "
        f"{np.mean(gravity_ned[:, 2]):.4f}"
    )

    print(
        f"Horizontal median : "
        f"{np.median(horizontal_gravity):.4f}"
    )

    print(
        f"Horizontal max : "
        f"{np.max(horizontal_gravity):.4f}"
    )

    print(
        f"Gravity direction "
        f"median error : "
        f"{np.median(gravity_direction_error):.4f}°"
    )

    print(
        f"Gravity direction "
        f"max error : "
        f"{np.max(gravity_direction_error):.4f}°"
    )

    print("\nMOTION")

    print(
        f"Gyro magnitude mean : "
        f"{np.mean(gyro_magnitude):.4f} rad/s"
    )

    print(
        f"Gyro magnitude max  : "
        f"{np.max(gyro_magnitude):.4f} rad/s"
    )

    print(
        f"Linear accel mean : "
        f"{np.mean(linear_accel_magnitude):.4f} m/s²"
    )

    print(
        f"Linear accel max : "
        f"{np.max(linear_accel_magnitude):.4f} m/s²"
    )

    return {
        "time": local_time,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "gravity_ned": gravity_ned,
        "gravity_direction_error":
            gravity_direction_error,
        "gyro_magnitude":
            gyro_magnitude,
        "linear_accel_magnitude":
            linear_accel_magnitude,
        "gps_speed":
            local_speed
    }


# ============================================================
# PLOT
# ============================================================

def plot_results(
    results
):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 12)
    )

    for start, end, result in results:

        label = (
            f"{start:.0f}-{end:.0f}s"
        )

        axes[0].plot(
            result["time"],
            result["roll"],
            label=label
        )

        axes[1].plot(
            result["time"],
            result["pitch"],
            label=label
        )

        axes[2].plot(
            result["time"],
            result["yaw"],
            label=label
        )

    axes[0].set_title(
        "Dynamic Quaternion Roll"
    )

    axes[1].set_title(
        "Dynamic Quaternion Pitch"
    )

    axes[2].set_title(
        "Dynamic Quaternion Yaw"
    )

    axes[0].set_ylabel(
        "Roll (degrees)"
    )

    axes[1].set_ylabel(
        "Pitch (degrees)"
    )

    axes[2].set_ylabel(
        "Yaw (degrees)"
    )

    axes[2].set_xlabel(
        "Time (seconds)"
    )

    for ax in axes:

        ax.grid(
            True,
            alpha=0.3
        )

        ax.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Clean names
    # --------------------------------------------------------

    df = clean_column_names(
        df
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df = prepare_timestamp(
        df
    )

    print(
        f"Start : "
        f"{df['timestamp'].iloc[0]}"
    )

    print(
        f"End   : "
        f"{df['timestamp'].iloc[-1]}"
    )

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    columns = (
        get_sensor_columns(
            df
        )
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        time,
        accel,
        gravity,
        gyro,
        magnetometer,
        gps_speed
    ) = get_sensor_data(
        df,
        columns
    )

    print("\nSensor shapes:")

    print(
        f"Acceleration : "
        f"{accel.shape}"
    )

    print(
        f"Gravity      : "
        f"{gravity.shape}"
    )

    print(
        f"Gyroscope    : "
        f"{gyro.shape}"
    )

    print(
        f"Magnetometer : "
        f"{magnetometer.shape}"
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    dt = np.diff(
        time
    )

    valid_dt = dt[
        (dt > 0)
        &
        (dt <= MAX_DT)
    ]

    print("\nTimestamp sampling:")

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
        f"{np.max(dt):.4f} s"
    )

    print(
        f"Large gaps > {MAX_DT}s : "
        f"{np.sum(dt > MAX_DT)}"
    )

    # --------------------------------------------------------
    # IMPORTANT SENSOR CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SENSOR AXIS VALIDATION")
    print("=" * 70)

    print(
        "\nGravity median:"
    )

    print(
        np.median(
            gravity,
            axis=0
        )
    )

    print(
        "\nExpected gravity magnitude "
        "should be approximately 9.81 m/s²."
    )

    gravity_mag = np.linalg.norm(
        gravity,
        axis=1
    )

    print(
        f"Gravity magnitude median : "
        f"{np.median(gravity_mag):.6f}"
    )

    print(
        f"Gravity magnitude mean   : "
        f"{np.mean(gravity_mag):.6f}"
    )

    # --------------------------------------------------------
    # Gyroscope preprocessing
    # --------------------------------------------------------

    gyro_clean = clean_gyro(
        gyro
    )

    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------

    gyro_bias = (
        estimate_gyro_bias(
            time,
            gyro_clean
        )
    )

    print("\nGyroscope bias [X,Y,Z]:")

    print(
        gyro_bias
    )

    print(
        f"Bias magnitude : "
        f"{np.linalg.norm(gyro_bias):.8f} rad/s"
    )

    # --------------------------------------------------------
    # Dynamic validation
    # --------------------------------------------------------

    results = []

    for start, end in WINDOWS:

        result = analyze_window(
            time,
            accel,
            gravity,
            gyro_clean,
            magnetometer,
            gps_speed,
            start,
            end,
            gyro_bias
        )

        if result is not None:

            results.append(
                (
                    start,
                    end,
                    result
                )
            )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    print(
        "\nGenerating dynamic orientation plot..."
    )

    plot_results(
        results
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "STEP 5.10 V2 COMPLETE"
    )

    print("=" * 70)

    print(
        "Dynamic quaternion orientation "
        "validation completed."
    )

    print(
        "Velocity integration : NOT performed"
    )

    print(
        "Position integration : NOT performed"
    )

    print(
        "\nNext:"
    )

    print(
        "Review corrected dynamic "
        "orientation results."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()