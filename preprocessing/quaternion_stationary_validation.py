# ============================================================
# quaternion_stationary_validation.py
#
# STEP 5.9
#
# Stationary Quaternion Orientation Validation
#
# Purpose:
#   Validate quaternion-based orientation using:
#
#       1. Gyroscope
#       2. Gyroscope bias calibration
#       3. Gravity
#       4. Magnetometer
#
# IMPORTANT:
#   No velocity integration
#   No position integration
#
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

STATIONARY_START = 3133.60
STATIONARY_END = 3141.70

G = 9.80665


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("STEP 5.9")
    print("STATIONARY QUATERNION ORIENTATION VALIDATION")
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
# FIND COLUMN
# ============================================================

def find_column(df, prefix):

    matches = [
        c
        for c in df.columns
        if c.startswith(prefix)
    ]

    if not matches:

        raise KeyError(
            f"Column not found: {prefix}"
        )

    return matches[0]


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    date_column = "DATE_YYYY-MO-DD_HH-MI-SS_SSS"

    df["timestamp"] = pd.to_datetime(
        df[date_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamp found."
        )

    df["time_seconds"] = (
        df["timestamp"]
        -
        df["timestamp"].iloc[0]
    ).dt.total_seconds()

    return df


# ============================================================
# EXTRACT SENSOR DATA
# ============================================================

def extract_data(df):

    accel = np.column_stack([
        df[
            find_column(
                df,
                "ACCELEROMETER_X"
            )
        ].to_numpy(float),

        df[
            find_column(
                df,
                "ACCELEROMETER_Y"
            )
        ].to_numpy(float),

        df[
            find_column(
                df,
                "ACCELEROMETER_Z"
            )
        ].to_numpy(float)
    ])

    gravity = np.column_stack([
        df[
            find_column(
                df,
                "GRAVITY_X"
            )
        ].to_numpy(float),

        df[
            find_column(
                df,
                "GRAVITY_Y"
            )
        ].to_numpy(float),

        df[
            find_column(
                df,
                "GRAVITY_Z"
            )
        ].to_numpy(float)
    ])

    gyro = np.column_stack([
        df[
            find_column(
                df,
                "GYROSCOPE_Roll"
            )
        ].to_numpy(float),

        df[
            find_column(
                df,
                "GYROSCOPE_Pitch"
            )
        ].to_numpy(float),

        df[
            find_column(
                df,
                "GYROSCOPE_Yaw"
            )
        ].to_numpy(float)
    ])

    magnetometer = np.column_stack([
        df[
            find_column(
                df,
                "MAGNETIC_FIELD_X"
            )
        ].to_numpy(float),

        df[
            find_column(
                df,
                "MAGNETIC_FIELD_Y"
            )
        ].to_numpy(float),

        df[
            find_column(
                df,
                "MAGNETIC_FIELD_Z"
            )
        ].to_numpy(float)
    ])

    time = df[
        "time_seconds"
    ].to_numpy(float)

    return (
        time,
        accel,
        gravity,
        gyro,
        magnetometer
    )


# ============================================================
# SELECT STATIONARY WINDOW
# ============================================================

def select_window(
    time,
    accel,
    gravity,
    gyro,
    magnetometer
):

    mask = (
        (time >= STATIONARY_START)
        &
        (time <= STATIONARY_END)
    )

    return (
        time[mask],
        accel[mask],
        gravity[mask],
        gyro[mask],
        magnetometer[mask]
    )


# ============================================================
# INITIAL ORIENTATION FROM GRAVITY + MAGNETOMETER
# ============================================================

def initial_orientation(
    gravity,
    magnetometer
):

    g = np.mean(
        gravity,
        axis=0
    )

    m = np.mean(
        magnetometer,
        axis=0
    )

    g = g / np.linalg.norm(g)

    m = m / np.linalg.norm(m)

    # --------------------------------------------------------
    # Down vector
    # --------------------------------------------------------

    down = g

    # --------------------------------------------------------
    # Magnetic field projection
    # Remove vertical component
    # --------------------------------------------------------

    magnetic_horizontal = (
        m
        -
        np.dot(m, down) * down
    )

    magnetic_horizontal /= np.linalg.norm(
        magnetic_horizontal
    )

    # --------------------------------------------------------
    # Define East
    # --------------------------------------------------------

    east = np.cross(
        down,
        magnetic_horizontal
    )

    east /= np.linalg.norm(
        east
    )

    # --------------------------------------------------------
    # Define North
    # --------------------------------------------------------

    north = np.cross(
        east,
        down
    )

    north /= np.linalg.norm(
        north
    )

    # --------------------------------------------------------
    # Navigation basis
    #
    # Columns represent body axes expressed in NED.
    # --------------------------------------------------------

    R_nav_body = np.column_stack([
        north,
        east,
        down
    ])

    # --------------------------------------------------------
    # scipy uses standard right-handed rotations.
    #
    # NED has Down instead of Up, therefore we construct
    # orientation carefully and use it primarily as an
    # initialization/reference.
    # --------------------------------------------------------

    yaw = np.arctan2(
        north[1],
        north[0]
    )

    # For stationary phone nearly level:
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

    return (
        roll,
        pitch,
        yaw
    )


# ============================================================
# QUATERNION GYRO INTEGRATION
# ============================================================

def integrate_gyro(
    gyro,
    time,
    gyro_bias,
    initial_rpy
):

    n = len(
        time
    )

    quaternions = np.zeros(
        (n, 4)
    )

    # scipy quaternion:
    # [x, y, z, w]

    initial_rotation = Rotation.from_euler(
        "xyz",
        initial_rpy,
        degrees=False
    )

    quaternions[0] = (
        initial_rotation
        .as_quat()
    )

    for i in range(
        1,
        n
    ):

        dt = (
            time[i]
            -
            time[i - 1]
        )

        # ----------------------------------------------------
        # Protect against bad timestamps
        # ----------------------------------------------------

        if dt <= 0:

            dt = 0.1

        if dt > 0.5:

            dt = 0.1

        # ----------------------------------------------------
        # Bias correction
        # ----------------------------------------------------

        omega = (
            gyro[i - 1]
            -
            gyro_bias
        )

        # ----------------------------------------------------
        # Rotation increment
        # ----------------------------------------------------

        rotation_vector = (
            omega
            *
            dt
        )

        delta_rotation = (
            Rotation
            .from_rotvec(
                rotation_vector
            )
        )

        current_rotation = (
            Rotation
            .from_quat(
                quaternions[i - 1]
            )
        )

        new_rotation = (
            current_rotation
            *
            delta_rotation
        )

        quaternions[i] = (
            new_rotation
            .as_quat()
        )

    return quaternions


# ============================================================
# QUATERNION TO EULER
# ============================================================

def quaternion_to_euler(
    quaternions
):

    rotation = Rotation.from_quat(
        quaternions
    )

    euler = rotation.as_euler(
        "xyz",
        degrees=True
    )

    return euler


# ============================================================
# ANGLE WRAPPING
# ============================================================

def wrap_angle(angle):

    return (
        angle + 180
    ) % 360 - 180


# ============================================================
# STATIONARY DRIFT STATISTICS
# ============================================================

def calculate_statistics(
    euler
):

    roll = euler[:, 0]

    pitch = euler[:, 1]

    yaw = wrap_angle(
        euler[:, 2]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "QUATERNION ORIENTATION STATISTICS"
    )

    print(
        "=" * 70
    )

    for name, values in [
        ("Roll", roll),
        ("Pitch", pitch),
        ("Yaw", yaw)
    ]:

        print(
            f"\n{name}"
        )

        print(
            f"Mean   : "
            f"{np.mean(values):.6f}°"
        )

        print(
            f"Std    : "
            f"{np.std(values):.6f}°"
        )

        print(
            f"Min    : "
            f"{np.min(values):.6f}°"
        )

        print(
            f"Max    : "
            f"{np.max(values):.6f}°"
        )

        print(
            f"Range  : "
            f"{np.ptp(values):.6f}°"
        )

    return (
        roll,
        pitch,
        yaw
    )


# ============================================================
# GRAVITY RECONSTRUCTION
# ============================================================

def reconstruct_gravity(
    euler
):

    gravity = np.zeros(
        (len(euler), 3)
    )

    for i, angles in enumerate(
        euler
    ):

        rotation = Rotation.from_euler(
            "xyz",
            angles,
            degrees=True
        )

        # Body gravity reference.
        body_gravity = np.array([
            0.0,
            0.0,
            G
        ])

        gravity[i] = (
            rotation
            .apply(
                body_gravity
            )
        )

    return gravity


# ============================================================
# PLOT
# ============================================================

def plot_results(
    time,
    euler
):

    relative_time = (
        time
        -
        time[0]
    )

    roll = euler[:, 0]

    pitch = euler[:, 1]

    yaw = np.unwrap(
        np.deg2rad(
            euler[:, 2]
        )
    )

    yaw = np.rad2deg(
        yaw
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        sharex=True
    )

    axes[0].plot(
        relative_time,
        roll
    )

    axes[0].axhline(
        roll[0],
        linestyle="--"
    )

    axes[0].set_ylabel(
        "Roll (deg)"
    )

    axes[0].set_title(
        "Stationary Quaternion Roll"
    )

    axes[0].grid(True)

    axes[1].plot(
        relative_time,
        pitch
    )

    axes[1].axhline(
        pitch[0],
        linestyle="--"
    )

    axes[1].set_ylabel(
        "Pitch (deg)"
    )

    axes[1].set_title(
        "Stationary Quaternion Pitch"
    )

    axes[1].grid(True)

    axes[2].plot(
        relative_time,
        yaw
    )

    axes[2].axhline(
        yaw[0],
        linestyle="--"
    )

    axes[2].set_ylabel(
        "Yaw (deg)"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].set_title(
        "Stationary Quaternion Yaw"
    )

    axes[2].grid(True)

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

    df = clean_column_names(
        df
    )

    df = prepare_data(
        df
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    (
        time,
        accel,
        gravity,
        gyro,
        magnetometer
    ) = extract_data(
        df
    )

    # --------------------------------------------------------
    # Stationary window
    # --------------------------------------------------------

    (
        time,
        accel,
        gravity,
        gyro,
        magnetometer
    ) = select_window(
        time,
        accel,
        gravity,
        gyro,
        magnetometer
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATIONARY WINDOW"
    )

    print(
        "=" * 70
    )

    print(
        f"Start : {time[0]:.3f} s"
    )

    print(
        f"End   : {time[-1]:.3f} s"
    )

    print(
        f"Samples : {len(time)}"
    )

    # --------------------------------------------------------
    # Bias values from Step 5.8
    # --------------------------------------------------------

    gyro_bias = np.mean(
        gyro,
        axis=0
    )

    print(
        "\nGyroscope bias:"
    )

    print(
        gyro_bias
    )

    # --------------------------------------------------------
    # Initial orientation
    # --------------------------------------------------------

    (
        roll0,
        pitch0,
        yaw0
    ) = initial_orientation(
        gravity,
        magnetometer
    )

    initial_rpy = np.array([
        roll0,
        pitch0,
        yaw0
    ])

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INITIAL ORIENTATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRoll  : "
        f"{np.degrees(roll0):.6f}°"
    )

    print(
        f"Pitch : "
        f"{np.degrees(pitch0):.6f}°"
    )

    print(
        f"Yaw   : "
        f"{np.degrees(yaw0):.6f}°"
    )

    # --------------------------------------------------------
    # Quaternion integration
    # --------------------------------------------------------

    print(
        "\nIntegrating calibrated gyroscope..."
    )

    quaternions = integrate_gyro(
        gyro,
        time,
        gyro_bias,
        initial_rpy
    )

    # --------------------------------------------------------
    # Convert to Euler
    # --------------------------------------------------------

    euler = quaternion_to_euler(
        quaternions
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    (
        roll,
        pitch,
        yaw
    ) = calculate_statistics(
        euler
    )

    # --------------------------------------------------------
    # Drift
    # --------------------------------------------------------

    roll_drift = (
        roll[-1]
        -
        roll[0]
    )

    pitch_drift = (
        pitch[-1]
        -
        pitch[0]
    )

    yaw_drift = wrap_angle(
        yaw[-1]
        -
        yaw[0]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATIONARY ORIENTATION DRIFT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRoll drift  : "
        f"{roll_drift:.6f}°"
    )

    print(
        f"Pitch drift : "
        f"{pitch_drift:.6f}°"
    )

    print(
        f"Yaw drift   : "
        f"{yaw_drift:.6f}°"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    print(
        "\nGenerating plot..."
    )

    plot_results(
        time,
        euler
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 5.9 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nQuaternion stationary validation completed."
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
        "Dynamic orientation validation"
    )