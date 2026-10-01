# ============================================================
# orientation_complementary_v2.py
#
# STEP 5.5
# Complementary Orientation Estimation
#
# Sensors:
#   Accelerometer
#   Gravity
#   Gyroscope
#   Magnetometer
#
# Output:
#   Quaternion orientation
#   Roll / Pitch / Yaw
#   NED gravity validation
#   Navigation-frame linear acceleration
#
# IMPORTANT:
#   DO NOT integrate velocity or position yet.
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

GRAVITY_GAIN = 0.02
MAGNETOMETER_GAIN = 0.01

GRAVITY_NORM_MIN = 8.5
GRAVITY_NORM_MAX = 11.2

MAG_MIN = 15.0
MAG_MAX = 100.0

MAX_DT = 0.20

GYRO_ROBUST_SCALE = 7.0


# ============================================================
# VECTOR NORMALIZATION
# ============================================================

def normalize_vector(vector):

    vector = np.asarray(
        vector,
        dtype=float
    )

    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        return vector.copy()

    return vector / norm


# ============================================================
# ANGLE WRAPPING
# ============================================================

def wrap_angle_degrees(angle):

    return (
        (angle + 180.0) % 360.0
    ) - 180.0


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
# LOAD DATASET
# ============================================================

def load_dataset():

    print("\nLoading dataset...")

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
# PREPARE DATASET
# ============================================================

def prepare_dataset(df):

    df = clean_column_names(df)

    date_column = (
        "DATE_YYYY-MO-DD_HH-MI-SS_SSS"
    )

    # --------------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df[date_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    print("\nTimestamp validation:")

    invalid = (
        df["timestamp"].isna().sum()
    )

    print(
        f"Invalid timestamps : {invalid}"
    )

    print(
        f"Start : "
        f"{df['timestamp'].iloc[0]}"
    )

    print(
        f"End   : "
        f"{df['timestamp'].iloc[-1]}"
    )

    # ========================================================
    # IMPORTANT FIX
    #
    # Do NOT use:
    #
    # astype("int64") / 1e9
    #
    # because pandas 3 may store datetime64[us].
    #
    # Use total_seconds() instead.
    # ========================================================

    timestamp_seconds = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    df["timestamp_seconds"] = (
        timestamp_seconds
    )

    # --------------------------------------------------------
    # Validate duration
    # --------------------------------------------------------

    duration = (
        df["timestamp_seconds"].iloc[-1]
        - df["timestamp_seconds"].iloc[0]
    )

    print()

    print(
        f"Dataset duration : "
        f"{duration:.3f} seconds"
    )

    print(
        f"Dataset duration : "
        f"{duration / 60:.2f} minutes"
    )

    print(
        f"Dataset duration : "
        f"{duration / 3600:.2f} hours"
    )

    # --------------------------------------------------------
    # Calculate dt
    # --------------------------------------------------------

    dt = np.diff(
        df["timestamp_seconds"].to_numpy()
    )

    positive_dt = dt[
        dt > 0
    ]

    print()

    print(
        "Timestamp sampling:"
    )

    print(
        f"Median dt : "
        f"{np.median(positive_dt):.4f} s"
    )

    print(
        f"Mean dt   : "
        f"{np.mean(positive_dt):.4f} s"
    )

    print(
        f"Min dt    : "
        f"{np.min(positive_dt):.4f} s"
    )

    print(
        f"Max dt    : "
        f"{np.max(positive_dt):.4f} s"
    )

    return df


# ============================================================
# FIND SENSOR COLUMNS
# ============================================================

def find_columns(df):

    def find_column(prefix):

        matches = [
            column
            for column in df.columns
            if column.startswith(prefix)
        ]

        if not matches:

            raise KeyError(
                f"Could not find column "
                f"starting with: {prefix}"
            )

        return matches[0]

    return {

        "acceleration": [
            find_column(
                "ACCELEROMETER_X"
            ),
            find_column(
                "ACCELEROMETER_Y"
            ),
            find_column(
                "ACCELEROMETER_Z"
            )
        ],

        "gravity": [
            find_column(
                "GRAVITY_X"
            ),
            find_column(
                "GRAVITY_Y"
            ),
            find_column(
                "GRAVITY_Z"
            )
        ],

        "gyro": [
            find_column(
                "GYROSCOPE_Roll"
            ),
            find_column(
                "GYROSCOPE_Pitch"
            ),
            find_column(
                "GYROSCOPE_Yaw"
            )
        ],

        "magnetometer": [
            find_column(
                "MAGNETIC_FIELD_X"
            ),
            find_column(
                "MAGNETIC_FIELD_Y"
            ),
            find_column(
                "MAGNETIC_FIELD_Z"
            )
        ]
    }


# ============================================================
# EXTRACT SENSOR DATA
# ============================================================

def extract_sensors(
    df,
    columns
):

    timestamp = (
        df[
            "timestamp_seconds"
        ]
        .to_numpy(
            dtype=float
        )
    )

    acceleration = (
        df[
            columns["acceleration"]
        ]
        .to_numpy(
            dtype=float
        )
    )

    gravity = (
        df[
            columns["gravity"]
        ]
        .to_numpy(
            dtype=float
        )
    )

    gyroscope = (
        df[
            columns["gyro"]
        ]
        .to_numpy(
            dtype=float
        )
    )

    magnetometer = (
        df[
            columns["magnetometer"]
        ]
        .to_numpy(
            dtype=float
        )
    )

    print("\nSensor shapes:")

    print(
        f"Acceleration : "
        f"{acceleration.shape}"
    )

    print(
        f"Gravity      : "
        f"{gravity.shape}"
    )

    print(
        f"Gyroscope    : "
        f"{gyroscope.shape}"
    )

    print(
        f"Magnetometer : "
        f"{magnetometer.shape}"
    )

    return (
        timestamp,
        acceleration,
        gravity,
        gyroscope,
        magnetometer
    )


# ============================================================
# GYROSCOPE OUTLIER CLEANING
# ============================================================

def clean_gyroscope(gyro):

    print(
        "\nGyroscope preprocessing..."
    )

    gyro_magnitude = (
        np.linalg.norm(
            gyro,
            axis=1
        )
    )

    median = np.median(
        gyro_magnitude
    )

    mad = np.median(
        np.abs(
            gyro_magnitude
            - median
        )
    )

    robust_sigma = (
        1.4826 * mad
    )

    threshold = (
        median
        + GYRO_ROBUST_SCALE
        * robust_sigma
    )

    outlier_mask = (
        gyro_magnitude
        > threshold
    )

    cleaned = gyro.copy()

    # --------------------------------------------------------
    # Interpolate gyro outliers
    # --------------------------------------------------------

    for axis in range(3):

        series = pd.Series(
            cleaned[:, axis]
        )

        series[
            outlier_mask
        ] = np.nan

        series = (
            series
            .interpolate(
                method="linear",
                limit_direction="both"
            )
        )

        cleaned[:, axis] = (
            series.to_numpy()
        )

    print(
        f"Outliers : "
        f"{np.sum(outlier_mask)}"
    )

    print(
        f"Threshold : "
        f"{threshold:.6f} rad/s"
    )

    print(
        f"Clean maximum : "
        f"{np.max(np.linalg.norm(cleaned, axis=1)):.4f} "
        f"rad/s"
    )

    return (
        cleaned,
        outlier_mask
    )


# ============================================================
# INITIAL ORIENTATION
# ============================================================

def calculate_initial_orientation(
    gravity,
    magnetometer
):

    print(
        "\nCalculating initial orientation..."
    )

    samples = min(
        100,
        len(gravity)
    )

    gravity_initial = np.median(
        gravity[:samples],
        axis=0
    )

    magnetometer_initial = np.median(
        magnetometer[:samples],
        axis=0
    )

    gravity_initial = (
        normalize_vector(
            gravity_initial
        )
    )

    magnetometer_initial = (
        normalize_vector(
            magnetometer_initial
        )
    )

    # NED
    north = np.array(
        [1.0, 0.0, 0.0]
    )

    down = np.array(
        [0.0, 0.0, 1.0]
    )

    # --------------------------------------------------------
    # Remove vertical magnetic component
    # --------------------------------------------------------

    magnetic_horizontal = (
        magnetometer_initial
        - np.dot(
            magnetometer_initial,
            gravity_initial
        )
        * gravity_initial
    )

    magnetic_horizontal = (
        normalize_vector(
            magnetic_horizontal
        )
    )

    try:

        rotation = (
            Rotation.align_vectors(
                np.vstack(
                    [
                        down,
                        north
                    ]
                ),
                np.vstack(
                    [
                        gravity_initial,
                        magnetic_horizontal
                    ]
                ),
                weights=[
                    10.0,
                    1.0
                ]
            )[0]
        )

    except Exception:

        rotation = (
            Rotation.align_vectors(
                [down],
                [gravity_initial]
            )[0]
        )

    euler = (
        rotation
        .as_euler(
            "xyz",
            degrees=True
        )
    )

    euler = np.array(
        [
            wrap_angle_degrees(
                euler[0]
            ),
            wrap_angle_degrees(
                euler[1]
            ),
            wrap_angle_degrees(
                euler[2]
            )
        ]
    )

    print(
        "\nInitial orientation:"
    )

    print(
        f"Roll  : {euler[0]:.2f}°"
    )

    print(
        f"Pitch : {euler[1]:.2f}°"
    )

    print(
        f"Yaw   : {euler[2]:.2f}°"
    )

    print(
        "\nInitial gravity vector:"
    )

    print(
        gravity_initial
    )

    print(
        "\nInitial magnetic vector:"
    )

    print(
        magnetometer_initial
    )

    return rotation


# ============================================================
# COMPLEMENTARY FILTER
# ============================================================

def run_complementary_filter(
    timestamp,
    gyro,
    gravity,
    magnetometer,
    initial_rotation
):

    print(
        "\nRunning complementary orientation filter..."
    )

    n = len(timestamp)

    quaternions = np.zeros(
        (n, 4)
    )

    euler_angles = np.zeros(
        (n, 3)
    )

    gravity_error_angles = (
        np.zeros(n)
    )

    magnetometer_corrections = (
        np.zeros(n)
    )

    dt_values = np.zeros(n)

    rotation = (
        initial_rotation
    )

    quaternions[0] = (
        rotation.as_quat()
    )

    euler_angles[0] = (
        rotation.as_euler(
            "xyz",
            degrees=True
        )
    )

    for i in range(1, n):

        # ====================================================
        # TIMESTAMP
        # ====================================================

        dt = (
            timestamp[i]
            - timestamp[i - 1]
        )

        # Protect against bad timestamps
        if (
            not np.isfinite(dt)
            or dt <= 0
        ):

            dt = 0.1

        # Large timestamp gap
        #
        # Do not integrate a giant gyro step.
        #

        if dt > MAX_DT:

            dt_values[i] = dt

            # Keep orientation unchanged
            # across missing data.
            rotation = rotation

        else:

            dt_values[i] = dt

            # =================================================
            # GYRO PROPAGATION
            # =================================================

            angular_velocity = (
                gyro[i]
            )

            rotation_increment = (
                Rotation.from_rotvec(
                    angular_velocity
                    * dt
                )
            )

            rotation = (
                rotation
                * rotation_increment
            )

        # ====================================================
        # GRAVITY CORRECTION
        # ====================================================

        measured_gravity = (
            gravity[i]
        )

        gravity_norm = (
            np.linalg.norm(
                measured_gravity
            )
        )

        if (
            GRAVITY_NORM_MIN
            <= gravity_norm
            <= GRAVITY_NORM_MAX
        ):

            measured_down_body = (
                measured_gravity
                / gravity_norm
            )

            predicted_down_body = (
                rotation.inv().apply(
                    np.array(
                        [0.0, 0.0, 1.0]
                    )
                )
            )

            predicted_down_body = (
                normalize_vector(
                    predicted_down_body
                )
            )

            gravity_error = np.cross(
                predicted_down_body,
                measured_down_body
            )

            error_norm = (
                np.linalg.norm(
                    gravity_error
                )
            )

            if error_norm > 1e-10:

                error_angle = (
                    np.degrees(
                        np.arcsin(
                            np.clip(
                                error_norm,
                                0.0,
                                1.0
                            )
                        )
                    )
                )

                gravity_error_angles[i] = (
                    error_angle
                )

                correction = (
                    Rotation.from_rotvec(
                        -gravity_error
                        * GRAVITY_GAIN
                    )
                )

                rotation = (
                    rotation
                    * correction
                )

        # ====================================================
        # MAGNETOMETER CORRECTION
        # ====================================================

        magnetic_field = (
            magnetometer[i]
        )

        magnetic_norm = (
            np.linalg.norm(
                magnetic_field
            )
        )

        if (
            MAG_MIN
            <= magnetic_norm
            <= MAG_MAX
        ):

            magnetic_body = (
                magnetic_field
                / magnetic_norm
            )

            magnetic_navigation = (
                rotation.apply(
                    magnetic_body
                )
            )

            vertical_component = (
                magnetic_navigation[2]
            )

            magnetic_horizontal = (
                magnetic_navigation
                - vertical_component
                * np.array(
                    [0.0, 0.0, 1.0]
                )
            )

            horizontal_norm = (
                np.linalg.norm(
                    magnetic_horizontal
                )
            )

            if horizontal_norm > 1e-6:

                magnetic_horizontal = (
                    magnetic_horizontal
                    / horizontal_norm
                )

                desired_north = np.array(
                    [1.0, 0.0, 0.0]
                )

                cross_z = np.cross(
                    magnetic_horizontal,
                    desired_north
                )[2]

                dot_value = np.clip(
                    np.dot(
                        magnetic_horizontal,
                        desired_north
                    ),
                    -1.0,
                    1.0
                )

                yaw_error = np.arctan2(
                    cross_z,
                    dot_value
                )

                magnetometer_corrections[i] = (
                    np.degrees(
                        yaw_error
                    )
                )

                yaw_correction = (
                    Rotation.from_rotvec(
                        np.array(
                            [
                                0.0,
                                0.0,
                                yaw_error
                            ]
                        )
                        * MAGNETOMETER_GAIN
                    )
                )

                rotation = (
                    yaw_correction
                    * rotation
                )

        # ====================================================
        # NORMALIZE QUATERNION
        # ====================================================

        q = rotation.as_quat()

        q = (
            q
            / np.linalg.norm(q)
        )

        rotation = (
            Rotation.from_quat(q)
        )

        # ====================================================
        # SAVE
        # ====================================================

        quaternions[i] = q

        euler = (
            rotation
            .as_euler(
                "xyz",
                degrees=True
            )
        )

        euler_angles[i] = np.array(
            [
                wrap_angle_degrees(
                    euler[0]
                ),
                wrap_angle_degrees(
                    euler[1]
                ),
                wrap_angle_degrees(
                    euler[2]
                )
            ]
        )

    return (
        quaternions,
        euler_angles,
        gravity_error_angles,
        magnetometer_corrections,
        dt_values
    )


# ============================================================
# ROTATE BODY VECTOR TO NED
# ============================================================

def rotate_to_navigation_frame(
    quaternions,
    vectors
):

    n = len(vectors)

    output = np.zeros(
        (n, 3)
    )

    for i in range(n):

        rotation = (
            Rotation.from_quat(
                quaternions[i]
            )
        )

        output[i] = (
            rotation.apply(
                vectors[i]
            )
        )

    return output


# ============================================================
# GRAVITY VALIDATION
# ============================================================

def validate_gravity(
    quaternions,
    gravity
):

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

    gravity_navigation = (
        rotate_to_navigation_frame(
            quaternions,
            gravity
        )
    )

    north = (
        gravity_navigation[:, 0]
    )

    east = (
        gravity_navigation[:, 1]
    )

    down = (
        gravity_navigation[:, 2]
    )

    print("\nExpected:")

    print(
        "North ≈ 0"
    )

    print(
        "East  ≈ 0"
    )

    print(
        "Down  ≈ +9.81"
    )

    print("\nMeasured:")

    print(
        f"North mean : "
        f"{np.mean(north):.4f}"
    )

    print(
        f"East mean  : "
        f"{np.mean(east):.4f}"
    )

    print(
        f"Down mean  : "
        f"{np.mean(down):.4f}"
    )

    print()

    print(
        f"North std : "
        f"{np.std(north):.4f}"
    )

    print(
        f"East std  : "
        f"{np.std(east):.4f}"
    )

    print(
        f"Down std  : "
        f"{np.std(down):.4f}"
    )

    horizontal = np.sqrt(
        north ** 2
        + east ** 2
    )

    print()

    print(
        f"Horizontal gravity median : "
        f"{np.median(horizontal):.4f}"
    )

    print(
        f"Horizontal gravity maximum : "
        f"{np.max(horizontal):.4f}"
    )

    # --------------------------------------------------------
    # Direction error
    # --------------------------------------------------------

    magnitude = np.linalg.norm(
        gravity_navigation,
        axis=1
    )

    valid = (
        magnitude > 1e-6
    )

    unit_gravity = (
        gravity_navigation[valid]
        /
        magnitude[valid, None]
    )

    expected_down = np.array(
        [0.0, 0.0, 1.0]
    )

    dot = np.clip(
        np.dot(
            unit_gravity,
            expected_down
        ),
        -1.0,
        1.0
    )

    direction_error = (
        np.degrees(
            np.arccos(dot)
        )
    )

    print()

    print(
        f"Gravity direction mean error : "
        f"{np.mean(direction_error):.4f}°"
    )

    print(
        f"Gravity direction median error : "
        f"{np.median(direction_error):.4f}°"
    )

    return gravity_navigation


# ============================================================
# NAVIGATION ACCELERATION
# ============================================================

def calculate_navigation_acceleration(
    quaternions,
    acceleration,
    gravity
):

    # --------------------------------------------------------
    # Remove gravity in body frame
    # --------------------------------------------------------

    linear_body = (
        acceleration
        - gravity
    )

    # --------------------------------------------------------
    # Rotate into NED
    # --------------------------------------------------------

    linear_navigation = (
        rotate_to_navigation_frame(
            quaternions,
            linear_body
        )
    )

    return (
        linear_body,
        linear_navigation
    )


# ============================================================
# ORIENTATION PLOT
# ============================================================

def plot_orientation(
    timestamp,
    euler_angles
):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 12),
        sharex=True
    )

    axes[0].plot(
        timestamp,
        euler_angles[:, 0],
        linewidth=0.7
    )

    axes[0].set_title(
        "Complementary Filter Roll"
    )

    axes[0].set_ylabel(
        "Roll (degrees)"
    )

    axes[0].grid(True)

    axes[1].plot(
        timestamp,
        euler_angles[:, 1],
        linewidth=0.7
    )

    axes[1].set_title(
        "Complementary Filter Pitch"
    )

    axes[1].set_ylabel(
        "Pitch (degrees)"
    )

    axes[1].grid(True)

    axes[2].plot(
        timestamp,
        euler_angles[:, 2],
        linewidth=0.7
    )

    axes[2].set_title(
        "Complementary Filter Yaw"
    )

    axes[2].set_ylabel(
        "Yaw (degrees)"
    )

    axes[2].set_xlabel(
        "Time (seconds)"
    )

    axes[2].grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# GRAVITY PLOT
# ============================================================

def plot_gravity(
    timestamp,
    gravity_navigation
):

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        timestamp,
        gravity_navigation[:, 0],
        label="North gravity",
        linewidth=0.7
    )

    plt.plot(
        timestamp,
        gravity_navigation[:, 1],
        label="East gravity",
        linewidth=0.7
    )

    plt.plot(
        timestamp,
        gravity_navigation[:, 2],
        label="Down gravity",
        linewidth=0.7
    )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1
    )

    plt.axhline(
        9.80665,
        linestyle="--",
        linewidth=1
    )

    plt.title(
        "NED Gravity Validation - Orientation V2"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Gravity (m/s²)"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# ACCELERATION PLOT
# ============================================================

def plot_navigation_acceleration(
    timestamp,
    acceleration_navigation
):

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        timestamp,
        acceleration_navigation[:, 0],
        label="North acceleration",
        linewidth=0.7
    )

    plt.plot(
        timestamp,
        acceleration_navigation[:, 1],
        label="East acceleration",
        linewidth=0.7
    )

    plt.plot(
        timestamp,
        acceleration_navigation[:, 2],
        label="Down acceleration",
        linewidth=0.7
    )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1
    )

    plt.title(
        "Linear Acceleration in NED Frame"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Acceleration (m/s²)"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPLEMENTARY ORIENTATION V2.2"
    )

    print(
        "GYRO + GRAVITY + MAGNETOMETER"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. PREPARE
    # --------------------------------------------------------

    df = prepare_dataset(df)

    # --------------------------------------------------------
    # 3. FIND COLUMNS
    # --------------------------------------------------------

    columns = find_columns(
        df
    )

    # --------------------------------------------------------
    # 4. EXTRACT SENSORS
    # --------------------------------------------------------

    (
        timestamp,
        acceleration,
        gravity,
        gyro,
        magnetometer
    ) = extract_sensors(
        df,
        columns
    )

    # --------------------------------------------------------
    # 5. CLEAN GYRO
    # --------------------------------------------------------

    gyro_clean, gyro_outliers = (
        clean_gyroscope(
            gyro
        )
    )

    # --------------------------------------------------------
    # 6. INITIAL ORIENTATION
    # --------------------------------------------------------

    initial_rotation = (
        calculate_initial_orientation(
            gravity,
            magnetometer
        )
    )

    # --------------------------------------------------------
    # 7. COMPLEMENTARY FILTER
    # --------------------------------------------------------

    (
        quaternions,
        euler_angles,
        gravity_error_angles,
        magnetometer_corrections,
        dt_values
    ) = run_complementary_filter(
        timestamp,
        gyro_clean,
        gravity,
        magnetometer,
        initial_rotation
    )

    # --------------------------------------------------------
    # 8. ORIENTATION RESULTS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ORIENTATION V2.2 RESULTS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Roll range  : "
        f"{np.min(euler_angles[:, 0]):.2f}° "
        f"to "
        f"{np.max(euler_angles[:, 0]):.2f}°"
    )

    print(
        f"Pitch range : "
        f"{np.min(euler_angles[:, 1]):.2f}° "
        f"to "
        f"{np.max(euler_angles[:, 1]):.2f}°"
    )

    print(
        f"Yaw range   : "
        f"{np.min(euler_angles[:, 2]):.2f}° "
        f"to "
        f"{np.max(euler_angles[:, 2]):.2f}°"
    )

    print()

    print(
        f"Roll median  : "
        f"{np.median(euler_angles[:, 0]):.2f}°"
    )

    print(
        f"Pitch median : "
        f"{np.median(euler_angles[:, 1]):.2f}°"
    )

    print(
        f"Yaw median   : "
        f"{np.median(euler_angles[:, 2]):.2f}°"
    )

    # --------------------------------------------------------
    # 9. GRAVITY VALIDATION
    # --------------------------------------------------------

    gravity_navigation = (
        validate_gravity(
            quaternions,
            gravity
        )
    )

    # --------------------------------------------------------
    # 10. NAVIGATION ACCELERATION
    # --------------------------------------------------------

    (
        linear_body,
        linear_navigation
    ) = calculate_navigation_acceleration(
        quaternions,
        acceleration,
        gravity
    )

    acceleration_magnitude = (
        np.linalg.norm(
            linear_navigation,
            axis=1
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NAVIGATION ACCELERATION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Median magnitude : "
        f"{np.median(acceleration_magnitude):.4f} "
        f"m/s²"
    )

    print(
        f"Maximum magnitude : "
        f"{np.max(acceleration_magnitude):.4f} "
        f"m/s²"
    )

    # --------------------------------------------------------
    # 11. GRAVITY CORRECTION ERROR
    # --------------------------------------------------------

    valid_errors = (
        gravity_error_angles[
            gravity_error_angles > 0
        ]
    )

    if len(valid_errors) > 0:

        print()

        print(
            "Gravity correction:"
        )

        print(
            f"Mean error : "
            f"{np.mean(valid_errors):.4f}°"
        )

        print(
            f"Median error : "
            f"{np.median(valid_errors):.4f}°"
        )

    # --------------------------------------------------------
    # 12. FINAL TIMESTAMP CHECK
    # --------------------------------------------------------

    print()

    print(
        "Final timestamp validation:"
    )

    valid_dt = dt_values[
        dt_values > 0
    ]

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

    large_gaps = (
        valid_dt > MAX_DT
    )

    print(
        f"Large gaps > {MAX_DT}s : "
        f"{np.sum(large_gaps)}"
    )

    # --------------------------------------------------------
    # 13. PLOTS
    # --------------------------------------------------------

    print(
        "\nGenerating orientation plots..."
    )

    plot_orientation(
        timestamp,
        euler_angles
    )

    print(
        "Generating gravity validation plot..."
    )

    plot_gravity(
        timestamp,
        gravity_navigation
    )

    print(
        "Generating navigation acceleration plot..."
    )

    plot_navigation_acceleration(
        timestamp,
        linear_navigation
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 5.5 COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Timestamp scaling corrected."
    )

    print(
        "Orientation estimation completed."
    )

    print(
        "Do NOT integrate velocity or position yet."
    )

    print(
        "Next step: stationary/low-motion validation."
    )