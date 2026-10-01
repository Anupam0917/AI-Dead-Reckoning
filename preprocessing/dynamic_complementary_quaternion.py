import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.transform import Rotation


# ============================================================
# STEP 5.11
# DYNAMIC COMPLEMENTARY QUATERNION FILTER
#
# Gyroscope + Gravity + Magnetometer
#
# State:
#   Quaternion body -> NED
#
# Gyroscope:
#   Fast orientation propagation
#
# Gravity:
#   Roll / pitch correction
#
# Magnetometer:
#   Slow yaw correction
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

OUTPUT_DIR = Path("data/processed")
PLOT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


G = 9.80665

# Complementary filter gains.
#
# Larger value:
#   stronger correction
#
# Smaller value:
#   more trust in gyro
#
GRAVITY_GAIN = 0.025
MAG_GAIN = 0.008

# Maximum acceptable time step.
# Dataset has two large gaps around 1 second.
# We do not integrate across those gaps using the large dt.
MAX_DT = 0.25

# Gyroscope robust outlier threshold.
GYRO_THRESHOLD_MULTIPLIER = 10.0

# Magnetic disturbance threshold.
MAG_MIN_RATIO = 0.80
MAG_MAX_RATIO = 1.20

# Stationary detection thresholds.
STATIONARY_GYRO_THRESHOLD = 0.08       # rad/s
STATIONARY_LINEAR_ACCEL_THRESHOLD = 0.50  # m/s²
STATIONARY_GPS_SPEED_THRESHOLD = 0.30   # km/h
MIN_STATIONARY_DURATION = 2.0          # seconds


# Dynamic windows for validation.
DYNAMIC_WINDOWS = [
    (0.0, 30.0),
    (1000.0, 1030.0),
    (2500.0, 2530.0),
    (4400.0, 4430.0),
    (5000.0, 5030.0),
    (7500.0, 7530.0),
    (10000.0, 10030.0),
]


# ============================================================
# PRINT HEADER
# ============================================================

def print_header():
    print("=" * 70)
    print("STEP 5.11")
    print("DYNAMIC COMPLEMENTARY QUATERNION FILTER")
    print("=" * 70)
    print()
    print("Gyroscope  -> quaternion prediction")
    print("Gravity    -> roll/pitch correction")
    print("Magnetometer -> slow yaw correction")
    print()


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(df, prefix):
    """
    Find a column using its beginning rather than exact encoding.

    This avoids problems such as:
        Î¼T
        ÎÎμT
        etc.
    """

    matches = [
        column
        for column in df.columns
        if column.startswith(prefix)
    ]

    if not matches:
        raise KeyError(
            f"Could not find column beginning with: {prefix}"
        )

    return matches[0]


def detect_columns(df):

    columns = {}

    columns["accel_x"] = find_column(
        df,
        "ACCELEROMETER_X_"
    )

    columns["accel_y"] = find_column(
        df,
        "ACCELEROMETER_Y_"
    )

    columns["accel_z"] = find_column(
        df,
        "ACCELEROMETER_Z_"
    )

    columns["gravity_x"] = find_column(
        df,
        "GRAVITY_X_"
    )

    columns["gravity_y"] = find_column(
        df,
        "GRAVITY_Y_"
    )

    columns["gravity_z"] = find_column(
        df,
        "GRAVITY_Z_"
    )

    columns["gyro_x"] = find_column(
        df,
        "GYROSCOPE_Roll_"
    )

    columns["gyro_y"] = find_column(
        df,
        "GYROSCOPE_Pitch_"
    )

    columns["gyro_z"] = find_column(
        df,
        "GYROSCOPE_Yaw_"
    )

    columns["mag_x"] = find_column(
        df,
        "MAGNETIC_FIELD_X_"
    )

    columns["mag_y"] = find_column(
        df,
        "MAGNETIC_FIELD_Y_"
    )

    columns["mag_z"] = find_column(
        df,
        "MAGNETIC_FIELD_Z_"
    )

    columns["gps_speed"] = find_column(
        df,
        "GPS_SPEED_"
    )

    return columns


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        encoding="cp1252"
    )

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")
    print()

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    # --------------------------------------------------------
    # Timestamp
    #
    # Dataset format:
    #
    # 2019-09-07 09:13:29:506
    #
    # Last field is milliseconds.
    # --------------------------------------------------------

    date_column = "DATE_YYYY-MO-DD_HH-MI-SS_SSS"

    df["timestamp"] = pd.to_datetime(
        df[date_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    invalid = df["timestamp"].isna().sum()

    print("Timestamp validation:")
    print(f"Invalid timestamps : {invalid}")
    print(f"Start : {df['timestamp'].min()}")
    print(f"End   : {df['timestamp'].max()}")

    if invalid > 0:
        raise ValueError(
            "Invalid timestamps detected."
        )

    # --------------------------------------------------------
    # Elapsed time
    # --------------------------------------------------------

    df["time_seconds"] = (
        df["timestamp"] -
        df["timestamp"].iloc[0]
    ).dt.total_seconds()

    print(
        f"Duration : "
        f"{df['time_seconds'].iloc[-1]:.3f} s"
    )

    return df


# ============================================================
# GET SENSOR DATA
# ============================================================

def get_sensor_data(df, columns):

    accel = df[
        [
            columns["accel_x"],
            columns["accel_y"],
            columns["accel_z"],
        ]
    ].to_numpy(dtype=float)

    gravity = df[
        [
            columns["gravity_x"],
            columns["gravity_y"],
            columns["gravity_z"],
        ]
    ].to_numpy(dtype=float)

    gyro = df[
        [
            columns["gyro_x"],
            columns["gyro_y"],
            columns["gyro_z"],
        ]
    ].to_numpy(dtype=float)

    magnetometer = df[
        [
            columns["mag_x"],
            columns["mag_y"],
            columns["mag_z"],
        ]
    ].to_numpy(dtype=float)

    gps_speed = df[
        columns["gps_speed"]
    ].to_numpy(dtype=float)

    time = df[
        "time_seconds"
    ].to_numpy(dtype=float)

    return (
        accel,
        gravity,
        gyro,
        magnetometer,
        gps_speed,
        time
    )


# ============================================================
# TIME STEP
# ============================================================

def calculate_dt(time):

    dt = np.diff(time)

    median_dt = np.median(
        dt[dt > 0]
    )

    # First sample
    dt_full = np.zeros(len(time))

    dt_full[0] = median_dt

    dt_full[1:] = dt

    # Invalid / very large dt
    invalid_mask = (
        (dt_full <= 0) |
        (dt_full > MAX_DT)
    )

    number_invalid = np.sum(
        invalid_mask
    )

    # Replace problematic gaps with normal sampling period.
    dt_full[invalid_mask] = median_dt

    print()
    print("Timestamp sampling:")
    print(
        f"Median dt : {median_dt:.4f} s"
    )
    print(
        f"Mean dt   : {np.mean(dt):.4f} s"
    )
    print(
        f"Min dt    : {np.min(dt):.4f} s"
    )
    print(
        f"Max dt    : {np.max(dt):.4f} s"
    )
    print(
        f"Large/invalid dt corrected : "
        f"{number_invalid}"
    )

    return dt_full, median_dt


# ============================================================
# GYROSCOPE OUTLIER CLEANING
# ============================================================

def clean_gyro(gyro):

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

    robust_sigma = 1.4826 * mad

    threshold = (
        median +
        GYRO_THRESHOLD_MULTIPLIER *
        robust_sigma
    )

    outlier_mask = (
        magnitude > threshold
    )

    print()
    print("Gyroscope preprocessing:")
    print(
        f"Median magnitude : "
        f"{median:.6f} rad/s"
    )

    print(
        f"MAD              : "
        f"{mad:.6f}"
    )

    print(
        f"Robust sigma      : "
        f"{robust_sigma:.6f}"
    )

    print(
        f"Threshold         : "
        f"{threshold:.6f} rad/s"
    )

    print(
        f"Outliers          : "
        f"{np.sum(outlier_mask)}"
    )

    clean = gyro.copy()

    # Replace outliers with NaN
    clean[outlier_mask] = np.nan

    # Interpolate each axis
    for axis in range(3):

        series = pd.Series(
            clean[:, axis]
        )

        series = series.interpolate(
            method="linear",
            limit_direction="both"
        )

        clean[:, axis] = (
            series.to_numpy()
        )

    return (
        clean,
        outlier_mask,
        threshold
    )


# ============================================================
# AUTOMATIC STATIONARY DETECTION
# ============================================================

def find_stationary_window(
    gyro,
    accel,
    gravity,
    gps_speed,
    time
):

    gyro_mag = np.linalg.norm(
        gyro,
        axis=1
    )

    gravity_mag = np.linalg.norm(
        gravity,
        axis=1
    )

    linear_accel = (
        accel - gravity
    )

    linear_mag = np.linalg.norm(
        linear_accel,
        axis=1
    )

    stationary = (
        (gyro_mag < STATIONARY_GYRO_THRESHOLD)
        &
        (linear_mag < STATIONARY_LINEAR_ACCEL_THRESHOLD)
        &
        (np.abs(gps_speed) < STATIONARY_GPS_SPEED_THRESHOLD)
        &
        (np.abs(gravity_mag - G) < 0.05)
    )

    # Find contiguous stationary segments
    windows = []

    start = None

    for i, flag in enumerate(stationary):

        if flag and start is None:
            start = i

        elif not flag and start is not None:

            end = i - 1

            duration = (
                time[end] -
                time[start]
            )

            if duration >= MIN_STATIONARY_DURATION:

                windows.append(
                    (
                        start,
                        end,
                        duration
                    )
                )

            start = None

    # Last segment
    if start is not None:

        end = len(stationary) - 1

        duration = (
            time[end] -
            time[start]
        )

        if duration >= MIN_STATIONARY_DURATION:

            windows.append(
                (
                    start,
                    end,
                    duration
                )
            )

    if not windows:
        raise RuntimeError(
            "No stationary window found."
        )

    # Longest window
    best = max(
        windows,
        key=lambda x: x[2]
    )

    start_idx = best[0]
    end_idx = best[1]

    print()
    print("=" * 70)
    print("STATIONARY WINDOW FOR GYRO BIAS")
    print("=" * 70)

    print(
        f"Start    : "
        f"{time[start_idx]:.3f} s"
    )

    print(
        f"End      : "
        f"{time[end_idx]:.3f} s"
    )

    print(
        f"Duration : "
        f"{best[2]:.3f} s"
    )

    print(
        f"Samples  : "
        f"{end_idx - start_idx + 1}"
    )

    return start_idx, end_idx


# ============================================================
# GYRO BIAS
# ============================================================

def estimate_gyro_bias(
    gyro,
    start_idx,
    end_idx
):

    bias = np.mean(
        gyro[
            start_idx:end_idx + 1
        ],
        axis=0
    )

    print()
    print("Gyroscope bias:")
    print(bias)

    print(
        "Bias magnitude : "
        f"{np.linalg.norm(bias):.8f} rad/s"
    )

    return bias


# ============================================================
# QUATERNION HELPERS
# ============================================================

def normalize_quaternion(q):

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        return np.array(
            [0.0, 0.0, 0.0, 1.0]
        )

    return q / norm


def quaternion_nlerp(q1, q2, alpha):

    # Prevent taking the long path.
    if np.dot(q1, q2) < 0:
        q2 = -q2

    q = (
        (1.0 - alpha) * q1
        +
        alpha * q2
    )

    return normalize_quaternion(q)


# ============================================================
# INITIAL ORIENTATION
# ============================================================

def initial_orientation(
    gravity,
    magnetometer,
    start_idx,
    end_idx
):

    g = np.mean(
        gravity[
            start_idx:end_idx + 1
        ],
        axis=0
    )

    m = np.mean(
        magnetometer[
            start_idx:end_idx + 1
        ],
        axis=0
    )

    g = g / np.linalg.norm(g)

    # Remove vertical component from magnetic vector.
    m_horizontal = (
        m -
        np.dot(m, g) * g
    )

    m_norm = np.linalg.norm(
        m_horizontal
    )

    if m_norm < 1e-9:

        print(
            "WARNING: Magnetometer "
            "cannot determine initial heading."
        )

        # Identity orientation
        return np.array(
            [0.0, 0.0, 0.0, 1.0]
        )

    north_b = (
        m_horizontal /
        m_norm
    )

    # NED:
    #
    # north x
    # east  y
    # down  z
    #
    east_b = np.cross(
        g,
        north_b
    )

    east_b /= np.linalg.norm(
        east_b
    )

    # Re-orthogonalize north.
    north_b = np.cross(
        east_b,
        g
    )

    north_b /= np.linalg.norm(
        north_b
    )

    # Rotation body -> NED.
    #
    # Rows contain the NED basis
    # vectors expressed in body frame.
    R_bn = np.vstack(
        [
            north_b,
            east_b,
            g
        ]
    )

    # Numerical orthogonalization.
    U, _, Vt = np.linalg.svd(
        R_bn
    )

    R_bn = U @ Vt

    q = Rotation.from_matrix(
        R_bn
    ).as_quat()

    q = normalize_quaternion(q)

    return q


# ============================================================
# GRAVITY REFERENCE ORIENTATION
# ============================================================

def gravity_reference_quaternion(
    gravity_body,
    current_quaternion
):

    norm = np.linalg.norm(
        gravity_body
    )

    if norm < 1e-6:
        return current_quaternion

    g = gravity_body / norm

    # Current orientation's yaw.
    current_euler = (
        Rotation.from_quat(
            current_quaternion
        ).as_euler(
            "xyz",
            degrees=False
        )
    )

    current_yaw = current_euler[2]

    # Gravity in NED should be:
    #
    # [0, 0, +1]
    #
    # Roll / pitch are derived from
    # measured body gravity.

    roll = np.arctan2(
        g[1],
        g[2]
    )

    pitch = np.arctan2(
        -g[0],
        np.sqrt(
            g[1] ** 2 +
            g[2] ** 2
        )
    )

    reference = Rotation.from_euler(
        "xyz",
        [
            roll,
            pitch,
            current_yaw
        ]
    )

    return reference.as_quat()


# ============================================================
# MAGNETIC REFERENCE
# ============================================================

def magnetic_reference_quaternion(
    gravity_body,
    magnetometer_body,
    current_quaternion
):

    g_norm = np.linalg.norm(
        gravity_body
    )

    m_norm = np.linalg.norm(
        magnetometer_body
    )

    if g_norm < 1e-6:
        return current_quaternion, False

    if m_norm < 1e-6:
        return current_quaternion, False

    g = (
        gravity_body /
        g_norm
    )

    m_horizontal = (
        magnetometer_body
        -
        np.dot(
            magnetometer_body,
            g
        ) * g
    )

    horizontal_norm = np.linalg.norm(
        m_horizontal
    )

    if horizontal_norm < 1e-6:
        return current_quaternion, False

    north_b = (
        m_horizontal /
        horizontal_norm
    )

    east_b = np.cross(
        g,
        north_b
    )

    east_norm = np.linalg.norm(
        east_b
    )

    if east_norm < 1e-6:
        return current_quaternion, False

    east_b /= east_norm

    north_b = np.cross(
        east_b,
        g
    )

    north_b /= np.linalg.norm(
        north_b
    )

    R_bn = np.vstack(
        [
            north_b,
            east_b,
            g
        ]
    )

    U, _, Vt = np.linalg.svd(
        R_bn
    )

    R_bn = U @ Vt

    reference = Rotation.from_matrix(
        R_bn
    )

    return reference.as_quat(), True


# ============================================================
# DYNAMIC COMPLEMENTARY FILTER
# ============================================================

def run_filter(
    gyro,
    gravity,
    magnetometer,
    dt,
    gyro_bias
):

    n = len(gyro)

    quaternions = np.zeros(
        (n, 4)
    )

    roll = np.zeros(n)
    pitch = np.zeros(n)
    yaw = np.zeros(n)

    ned_gravity = np.zeros(
        (n, 3)
    )

    magnetic_valid = np.zeros(
        n,
        dtype=bool
    )

    correction_strength = np.zeros(
        n
    )

    # --------------------------------------------------------
    # Initial orientation
    # --------------------------------------------------------

    q = initial_orientation(
        gravity,
        magnetometer,
        0,
        min(
            n,
            100
        )
    )

    q = normalize_quaternion(q)

    quaternions[0] = q

    # --------------------------------------------------------
    # Reference magnetic magnitude
    # --------------------------------------------------------

    first_count = min(
        n,
        100
    )

    reference_mag = np.median(
        np.linalg.norm(
            magnetometer[:first_count],
            axis=1
        )
    )

    print()
    print(
        "Reference magnetic magnitude : "
        f"{reference_mag:.4f} µT"
    )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for i in range(1, n):

        # ----------------------------------------------------
        # 1. Gyroscope prediction
        # ----------------------------------------------------

        omega = (
            gyro[i]
            - gyro_bias
        )

        # Remove impossible values.
        if not np.all(
            np.isfinite(omega)
        ):
            omega = np.zeros(3)

        delta_rotation = (
            Rotation.from_rotvec(
                omega * dt[i]
            )
        )

        current_rotation = (
            Rotation.from_quat(q)
        )

        predicted_rotation = (
            current_rotation *
            delta_rotation
        )

        q_pred = (
            predicted_rotation
            .as_quat()
        )

        q_pred = normalize_quaternion(
            q_pred
        )

        # ----------------------------------------------------
        # 2. Gravity validity
        # ----------------------------------------------------

        gravity_norm = np.linalg.norm(
            gravity[i]
        )

        gravity_valid = (
            np.isfinite(gravity_norm)
            and
            abs(gravity_norm - G) < 0.5
        )

        # ----------------------------------------------------
        # 3. Gravity correction
        # ----------------------------------------------------

        q_corrected = q_pred.copy()

        if gravity_valid:

            q_gravity = (
                gravity_reference_quaternion(
                    gravity[i],
                    q_pred
                )
            )

            q_corrected = (
                quaternion_nlerp(
                    q_corrected,
                    q_gravity,
                    GRAVITY_GAIN
                )
            )

        # ----------------------------------------------------
        # 4. Magnetometer validation
        # ----------------------------------------------------

        mag_norm = np.linalg.norm(
            magnetometer[i]
        )

        magnetic_valid_now = False

        if (
            np.isfinite(mag_norm)
            and
            reference_mag > 1e-6
        ):

            ratio = (
                mag_norm /
                reference_mag
            )

            if (
                MAG_MIN_RATIO
                <= ratio
                <= MAG_MAX_RATIO
            ):

                magnetic_valid_now = True

        # ----------------------------------------------------
        # 5. Magnetometer yaw correction
        # ----------------------------------------------------

        if (
            magnetic_valid_now
            and
            gravity_valid
        ):

            q_mag, valid = (
                magnetic_reference_quaternion(
                    gravity[i],
                    magnetometer[i],
                    q_corrected
                )
            )

            if valid:

                q_corrected = (
                    quaternion_nlerp(
                        q_corrected,
                        q_mag,
                        MAG_GAIN
                    )
                )

                magnetic_valid[i] = True

        # ----------------------------------------------------
        # 6. Normalize
        # ----------------------------------------------------

        q = normalize_quaternion(
            q_corrected
        )

        quaternions[i] = q

        # ----------------------------------------------------
        # 7. Euler representation
        #
        # ONLY for plotting.
        # Quaternion remains the actual state.
        # ----------------------------------------------------

        euler = (
            Rotation.from_quat(q)
            .as_euler(
                "xyz",
                degrees=True
            )
        )

        roll[i] = euler[0]
        pitch[i] = euler[1]
        yaw[i] = euler[2]

        # ----------------------------------------------------
        # 8. Transform gravity body -> NED
        # ----------------------------------------------------

        R_bn = (
            Rotation.from_quat(q)
            .as_matrix()
        )

        ned_gravity[i] = (
            R_bn @ gravity[i]
        )

        correction_strength[i] = (
            GRAVITY_GAIN
            +
            (
                MAG_GAIN
                if magnetic_valid[i]
                else 0.0
            )
        )

    # Initial gravity
    R_bn = (
        Rotation.from_quat(
            quaternions[0]
        ).as_matrix()
    )

    ned_gravity[0] = (
        R_bn @ gravity[0]
    )

    euler = (
        Rotation.from_quat(
            quaternions[0]
        ).as_euler(
            "xyz",
            degrees=True
        )
    )

    roll[0] = euler[0]
    pitch[0] = euler[1]
    yaw[0] = euler[2]

    return (
        quaternions,
        roll,
        pitch,
        yaw,
        ned_gravity,
        magnetic_valid,
        correction_strength
    )


# ============================================================
# WINDOW VALIDATION
# ============================================================

def validate_window(
    start,
    end,
    time,
    gps_speed,
    gyro,
    accel,
    roll,
    pitch,
    yaw,
    ned_gravity,
    magnetic_valid
):

    mask = (
        (time >= start)
        &
        (time <= end)
    )

    if np.sum(mask) < 2:
        print(
            f"Window {start}-{end} "
            "has insufficient samples."
        )
        return

    indices = np.where(mask)[0]

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    r = roll[mask]
    p = pitch[mask]
    y = yaw[mask]

    # --------------------------------------------------------
    # Motion
    # --------------------------------------------------------

    gyro_mag = np.linalg.norm(
        gyro[mask],
        axis=1
    )

    linear_accel = (
        accel[mask]
        -
        np.linalg.norm(
            accel[mask],
            axis=1
        )[:, None]
        * 0.0
    )

    # We report raw acceleration magnitude
    # separately from gravity magnitude.
    accel_mag = np.linalg.norm(
        accel[mask],
        axis=1
    )

    # --------------------------------------------------------
    # NED gravity
    # --------------------------------------------------------

    g_ned = ned_gravity[mask]

    north = g_ned[:, 0]
    east = g_ned[:, 1]
    down = g_ned[:, 2]

    horizontal = np.sqrt(
        north ** 2 +
        east ** 2
    )

    direction_error = np.degrees(
        np.arccos(
            np.clip(
                down / G,
                -1.0,
                1.0
            )
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 70)
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
        f"{np.mean(gps_speed[mask]):.3f} km/h"
    )

    print(
        f"GPS speed max  : "
        f"{np.max(gps_speed[mask]):.3f} km/h"
    )

    print()
    print("ORIENTATION")

    print(
        f"Roll range  : "
        f"{np.min(r):.2f}° to "
        f"{np.max(r):.2f}°"
    )

    print(
        f"Pitch range : "
        f"{np.min(p):.2f}° to "
        f"{np.max(p):.2f}°"
    )

    print(
        f"Yaw range   : "
        f"{np.min(y):.2f}° to "
        f"{np.max(y):.2f}°"
    )

    print()
    print("MOTION")

    print(
        f"Gyro magnitude mean : "
        f"{np.mean(gyro_mag):.4f} rad/s"
    )

    print(
        f"Gyro magnitude max  : "
        f"{np.max(gyro_mag):.4f} rad/s"
    )

    print(
        f"Accel magnitude mean : "
        f"{np.mean(accel_mag):.4f} m/s²"
    )

    print(
        f"Accel magnitude max  : "
        f"{np.max(accel_mag):.4f} m/s²"
    )

    print()
    print("NED GRAVITY")

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

    print(
        f"Horizontal median : "
        f"{np.median(horizontal):.4f}"
    )

    print(
        f"Horizontal max : "
        f"{np.max(horizontal):.4f}"
    )

    print(
        f"Gravity direction "
        f"median error : "
        f"{np.median(direction_error):.4f}°"
    )

    print(
        f"Gravity direction "
        f"max error : "
        f"{np.max(direction_error):.4f}°"
    )

    print()
    print(
        "Magnetic reference valid : "
        f"{np.mean(magnetic_valid[mask]) * 100:.2f}%"
    )


# ============================================================
# PLOT
# ============================================================

def generate_plot(
    time,
    roll,
    pitch,
    yaw,
    ned_gravity,
    magnetic_valid
):

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 13),
        sharex=True
    )

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    axes[0].plot(
        time,
        roll,
        label="Roll"
    )

    axes[0].plot(
        time,
        pitch,
        label="Pitch"
    )

    axes[0].plot(
        time,
        yaw,
        label="Yaw"
    )

    axes[0].set_title(
        "Dynamic Complementary Quaternion Orientation"
    )

    axes[0].set_ylabel(
        "Angle (degrees)"
    )

    axes[0].legend()
    axes[0].grid(True)

    # --------------------------------------------------------
    # NED gravity
    # --------------------------------------------------------

    axes[1].plot(
        time,
        ned_gravity[:, 0],
        label="North gravity"
    )

    axes[1].plot(
        time,
        ned_gravity[:, 1],
        label="East gravity"
    )

    axes[1].plot(
        time,
        ned_gravity[:, 2],
        label="Down gravity"
    )

    axes[1].axhline(
        0.0,
        linestyle="--"
    )

    axes[1].axhline(
        G,
        linestyle="--"
    )

    axes[1].set_title(
        "NED Gravity Validation"
    )

    axes[1].set_ylabel(
        "Gravity (m/s²)"
    )

    axes[1].legend()
    axes[1].grid(True)

    # --------------------------------------------------------
    # Horizontal gravity
    # --------------------------------------------------------

    horizontal = np.sqrt(
        ned_gravity[:, 0] ** 2
        +
        ned_gravity[:, 1] ** 2
    )

    axes[2].plot(
        time,
        horizontal,
        label="Horizontal gravity"
    )

    axes[2].axhline(
        0.0,
        linestyle="--"
    )

    axes[2].set_title(
        "Horizontal Gravity Error"
    )

    axes[2].set_ylabel(
        "m/s²"
    )

    axes[2].legend()
    axes[2].grid(True)

    # --------------------------------------------------------
    # Magnetometer validity
    # --------------------------------------------------------

    axes[3].plot(
        time,
        magnetic_valid.astype(int),
        label="Magnetometer valid"
    )

    axes[3].set_title(
        "Magnetometer Reliability"
    )

    axes[3].set_ylabel(
        "Valid"
    )

    axes[3].set_xlabel(
        "Time (seconds)"
    )

    axes[3].set_yticks(
        [0, 1]
    )

    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()

    output_file = (
        PLOT_DIR /
        "dynamic_complementary_quaternion.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    print()
    print(
        f"Plot saved to: {output_file}"
    )

    plt.show()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    time,
    quaternions,
    roll,
    pitch,
    yaw,
    ned_gravity,
    magnetic_valid
):

    result = pd.DataFrame(
        {
            "time_seconds": time,

            "quat_x": quaternions[:, 0],
            "quat_y": quaternions[:, 1],
            "quat_z": quaternions[:, 2],
            "quat_w": quaternions[:, 3],

            "roll_deg": roll,
            "pitch_deg": pitch,
            "yaw_deg": yaw,

            "ned_gravity_n": ned_gravity[:, 0],
            "ned_gravity_e": ned_gravity[:, 1],
            "ned_gravity_d": ned_gravity[:, 2],

            "magnetometer_valid":
                magnetic_valid.astype(int),
        }
    )

    output_file = (
        OUTPUT_DIR /
        "dynamic_complementary_quaternion.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"Results saved to: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Detect columns
    # --------------------------------------------------------

    columns = detect_columns(df)

    print()
    print("Sensor columns detected:")

    for name, column in columns.items():
        print(
            f"{name:15s}: {column}"
        )

    # --------------------------------------------------------
    # Sensors
    # --------------------------------------------------------

    (
        accel,
        gravity,
        gyro,
        magnetometer,
        gps_speed,
        time
    ) = get_sensor_data(
        df,
        columns
    )

    print()
    print("Sensor shapes:")

    print(
        f"Acceleration : {accel.shape}"
    )

    print(
        f"Gravity      : {gravity.shape}"
    )

    print(
        f"Gyroscope    : {gyro.shape}"
    )

    print(
        f"Magnetometer : {magnetometer.shape}"
    )

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    dt, median_dt = calculate_dt(
        time
    )

    # --------------------------------------------------------
    # Sensor axis validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SENSOR AXIS VALIDATION")
    print("=" * 70)

    gravity_magnitude = np.linalg.norm(
        gravity,
        axis=1
    )

    print(
        "Gravity median:"
    )

    print(
        np.median(
            gravity,
            axis=0
        )
    )

    print(
        f"Gravity magnitude median : "
        f"{np.median(gravity_magnitude):.6f}"
    )

    print(
        f"Gravity magnitude mean   : "
        f"{np.mean(gravity_magnitude):.6f}"
    )

    # --------------------------------------------------------
    # Gyro cleaning
    # --------------------------------------------------------

    (
        gyro_clean,
        gyro_outliers,
        gyro_threshold
    ) = clean_gyro(
        gyro
    )

    # --------------------------------------------------------
    # Stationary window
    # --------------------------------------------------------

    stationary_start, stationary_end = (
        find_stationary_window(
            gyro_clean,
            accel,
            gravity,
            gps_speed,
            time
        )
    )

    # --------------------------------------------------------
    # Gyro bias
    # --------------------------------------------------------

    gyro_bias = estimate_gyro_bias(
        gyro_clean,
        stationary_start,
        stationary_end
    )

    # --------------------------------------------------------
    # Dynamic complementary filter
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING DYNAMIC COMPLEMENTARY QUATERNION FILTER")
    print("=" * 70)

    (
        quaternions,
        roll,
        pitch,
        yaw,
        ned_gravity,
        magnetic_valid,
        correction_strength
    ) = run_filter(
        gyro_clean,
        gravity,
        magnetometer,
        dt,
        gyro_bias
    )

    # --------------------------------------------------------
    # Initial orientation
    # --------------------------------------------------------

    print()
    print("Initial quaternion:")

    print(
        quaternions[0]
    )

    initial_euler = (
        Rotation.from_quat(
            quaternions[0]
        ).as_euler(
            "xyz",
            degrees=True
        )
    )

    print()
    print("Initial orientation:")

    print(
        f"Roll  : {initial_euler[0]:.4f}°"
    )

    print(
        f"Pitch : {initial_euler[1]:.4f}°"
    )

    print(
        f"Yaw   : {initial_euler[2]:.4f}°"
    )

    # --------------------------------------------------------
    # Global NED validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GLOBAL NED GRAVITY VALIDATION")
    print("=" * 70)

    north = ned_gravity[:, 0]
    east = ned_gravity[:, 1]
    down = ned_gravity[:, 2]

    horizontal = np.sqrt(
        north ** 2 +
        east ** 2
    )

    direction_error = np.degrees(
        np.arccos(
            np.clip(
                down / G,
                -1.0,
                1.0
            )
        )
    )

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

    print(
        f"North std  : "
        f"{np.std(north):.4f}"
    )

    print(
        f"East std   : "
        f"{np.std(east):.4f}"
    )

    print(
        f"Down std   : "
        f"{np.std(down):.4f}"
    )

    print(
        f"Horizontal gravity median : "
        f"{np.median(horizontal):.4f}"
    )

    print(
        f"Horizontal gravity max : "
        f"{np.max(horizontal):.4f}"
    )

    print(
        f"Gravity direction "
        f"median error : "
        f"{np.median(direction_error):.4f}°"
    )

    print(
        f"Gravity direction "
        f"mean error : "
        f"{np.mean(direction_error):.4f}°"
    )

    print(
        f"Gravity direction "
        f"max error : "
        f"{np.max(direction_error):.4f}°"
    )

    print()
    print(
        "Magnetometer valid percentage : "
        f"{np.mean(magnetic_valid) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Dynamic windows
    # --------------------------------------------------------

    for start, end in DYNAMIC_WINDOWS:

        validate_window(
            start,
            end,
            time,
            gps_speed,
            gyro_clean,
            accel,
            roll,
            pitch,
            yaw,
            ned_gravity,
            magnetic_valid
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        time,
        quaternions,
        roll,
        pitch,
        yaw,
        ned_gravity,
        magnetic_valid
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    print()
    print("Generating validation plot...")

    generate_plot(
        time,
        roll,
        pitch,
        yaw,
        ned_gravity,
        magnetic_valid
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 5.11 COMPLETE")
    print("=" * 70)

    print()
    print("Dynamic complementary quaternion filter completed.")

    print()
    print("Velocity integration : NOT performed")
    print("Position integration : NOT performed")

    print()
    print("Next:")
    print("Review NED gravity and dynamic orientation.")
    print("Only after validation will we proceed to Step 6.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()