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

GYRO_OUTLIER_THRESHOLD = 0.433104
LARGE_GAP_THRESHOLD = 0.2

# Correction strengths
GRAVITY_CORRECTION_GAIN = 0.08
MAGNETOMETER_CORRECTION_GAIN = 0.02

INITIAL_SAMPLES = 100


# ============================================================
# HELPER
# ============================================================

def find_column(df, prefix):

    matches = [
        column
        for column in df.columns
        if column.startswith(prefix)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely find {prefix}. "
            f"Found: {matches}"
        )

    return matches[0]


def wrap_angle(angle):

    return (
        (angle + 180.0) % 360.0
    ) - 180.0


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("COMPLEMENTARY ORIENTATION ESTIMATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

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


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_", regex=False)
)


# ============================================================
# TIMESTAMP
# ============================================================

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
    df["timestamp"] -
    df["timestamp"].iloc[0]
).dt.total_seconds()


df["dt"] = (
    df["time_seconds"].diff()
)


print("\nTimestamp validation:")
print(
    f"Invalid timestamps : "
    f"{df['timestamp'].isna().sum()}"
)

print(
    f"Start : {df['timestamp'].iloc[0]}"
)

print(
    f"End   : {df['timestamp'].iloc[-1]}"
)


# ============================================================
# SENSOR COLUMNS
# ============================================================

acceleration = df[
    [
        "ACCELEROMETER_X_m_s²",
        "ACCELEROMETER_Y_m_s²",
        "ACCELEROMETER_Z_m_s²"
    ]
].to_numpy(dtype=float)


gravity = df[
    [
        "GRAVITY_X_m_s²",
        "GRAVITY_Y_m_s²",
        "GRAVITY_Z_m_s²"
    ]
].to_numpy(dtype=float)


gyro = df[
    [
        "GYROSCOPE_Roll_rad_s",
        "GYROSCOPE_Pitch_rad_s",
        "GYROSCOPE_Yaw_rad_s"
    ]
].to_numpy(dtype=float)


magnetometer_columns = [
    find_column(df, "MAGNETIC_FIELD_X"),
    find_column(df, "MAGNETIC_FIELD_Y"),
    find_column(df, "MAGNETIC_FIELD_Z")
]


magnetometer = df[
    magnetometer_columns
].to_numpy(dtype=float)


print("\nSensor shapes:")

print(
    f"Acceleration  : {acceleration.shape}"
)

print(
    f"Gravity       : {gravity.shape}"
)

print(
    f"Gyroscope     : {gyro.shape}"
)

print(
    f"Magnetometer  : {magnetometer.shape}"
)


# ============================================================
# CLEAN GYROSCOPE
# ============================================================

gyro_magnitude = np.linalg.norm(
    gyro,
    axis=1
)

outlier_mask = (
    gyro_magnitude >
    GYRO_OUTLIER_THRESHOLD
)


gyro_clean = gyro.copy()


for axis in range(3):

    series = pd.Series(
        gyro_clean[:, axis]
    )

    series[outlier_mask] = np.nan

    series = series.interpolate(
        method="linear",
        limit_direction="both"
    )

    gyro_clean[:, axis] = (
        series.to_numpy()
    )


print("\nGyroscope preprocessing:")

print(
    f"Outliers : "
    f"{np.sum(outlier_mask)}"
)

print(
    f"Clean maximum : "
    f"{np.max(np.linalg.norm(gyro_clean, axis=1)):.4f} rad/s"
)


# ============================================================
# INITIAL GRAVITY
# ============================================================

initial_gravity = np.median(
    gravity[:INITIAL_SAMPLES],
    axis=0
)


gravity_norm = np.linalg.norm(
    initial_gravity
)


initial_down = (
    initial_gravity /
    gravity_norm
)


# ============================================================
# INITIAL MAGNETOMETER
# ============================================================

initial_magnetic = np.median(
    magnetometer[:INITIAL_SAMPLES],
    axis=0
)


# Remove gravity component from magnetic field

magnetic_horizontal = (
    initial_magnetic
    -
    np.dot(
        initial_magnetic,
        initial_down
    )
    *
    initial_down
)


magnetic_horizontal_norm = np.linalg.norm(
    magnetic_horizontal
)


if magnetic_horizontal_norm < 1e-6:

    raise ValueError(
        "Magnetometer horizontal component "
        "is too small."
    )


initial_north = (
    magnetic_horizontal /
    magnetic_horizontal_norm
)


initial_east = np.cross(
    initial_down,
    initial_north
)


initial_east /= np.linalg.norm(
    initial_east
)


initial_north = np.cross(
    initial_east,
    initial_down
)


initial_north /= np.linalg.norm(
    initial_north
)


# Body -> Navigation matrix

initial_matrix = np.vstack(
    [
        initial_north,
        initial_east,
        initial_down
    ]
)


rotation = Rotation.from_matrix(
    initial_matrix
)


# ============================================================
# INITIAL EULER
# ============================================================

initial_euler = rotation.as_euler(
    "xyz",
    degrees=True
)


print("\nInitial orientation:")

print(
    f"Roll  : {initial_euler[0]:.2f}°"
)

print(
    f"Pitch : {initial_euler[1]:.2f}°"
)

print(
    f"Yaw   : {initial_euler[2]:.2f}°"
)


# ============================================================
# STORAGE
# ============================================================

N = len(df)


euler_angles = np.zeros(
    (N, 3)
)


navigation_gravity = np.zeros(
    (N, 3)
)


navigation_acceleration = np.zeros(
    (N, 3)
)


# ============================================================
# COMPLEMENTARY FILTER
# ============================================================

print(
    "\nRunning gyro + gravity + magnetometer "
    "orientation filter..."
)


for i in range(N):

    if i > 0:

        dt = df["dt"].iloc[i]

        if (
            np.isfinite(dt)
            and dt > 0
            and dt <= LARGE_GAP_THRESHOLD
        ):

            delta_rotation = Rotation.from_rotvec(
                gyro_clean[i] * dt
            )

            rotation = (
                rotation *
                delta_rotation
            )


    # --------------------------------------------------------
    # CURRENT GRAVITY IN BODY FRAME
    # --------------------------------------------------------

    g_body = gravity[i]

    g_norm = np.linalg.norm(
        g_body
    )


    if g_norm > 1e-6:

        g_body = (
            g_body /
            g_norm
        )


        # Gravity predicted in navigation frame

        predicted_gravity = (
            rotation.apply(
                np.array([0.0, 0.0, 1.0]),
                inverse=True
            )
        )


        # Error between measured and predicted gravity

        gravity_error = np.cross(
            predicted_gravity,
            g_body
        )


        gravity_correction = (
            Rotation.from_rotvec(
                gravity_error
                *
                GRAVITY_CORRECTION_GAIN
            )
        )


        rotation = (
            rotation *
            gravity_correction
        )


    # --------------------------------------------------------
    # MAGNETOMETER CORRECTION
    # --------------------------------------------------------

    magnetic = magnetometer[i]

    magnetic_norm = np.linalg.norm(
        magnetic
    )


    if magnetic_norm > 1e-6:

        magnetic = (
            magnetic /
            magnetic_norm
        )


        magnetic_horizontal = (
            magnetic
            -
            np.dot(
                magnetic,
                g_body
            )
            *
            g_body
        )


        magnetic_horizontal_norm = (
            np.linalg.norm(
                magnetic_horizontal
            )
        )


        if magnetic_horizontal_norm > 1e-6:

            magnetic_horizontal /= (
                magnetic_horizontal_norm
            )


            # Current magnetic vector in navigation frame

            magnetic_nav = (
                rotation.apply(
                    magnetic
                )
            )


            magnetic_nav_horizontal = (
                magnetic_nav.copy()
            )

            magnetic_nav_horizontal[2] = 0.0


            nav_norm = np.linalg.norm(
                magnetic_nav_horizontal
            )


            if nav_norm > 1e-6:

                magnetic_nav_horizontal /= (
                    nav_norm
                )


                desired_north = np.array(
                    [1.0, 0.0, 0.0]
                )


                yaw_error = np.arctan2(
                    magnetic_nav_horizontal[1],
                    magnetic_nav_horizontal[0]
                )


                yaw_correction = (
                    Rotation.from_euler(
                        "z",
                        -yaw_error
                        *
                        MAGNETOMETER_CORRECTION_GAIN,
                        degrees=False
                    )
                )


                rotation = (
                    yaw_correction *
                    rotation
                )


    # --------------------------------------------------------
    # STORE ORIENTATION
    # --------------------------------------------------------

    euler = rotation.as_euler(
        "xyz",
        degrees=True
    )


    euler_angles[i] = euler


    # --------------------------------------------------------
    # ROTATE GRAVITY
    # --------------------------------------------------------

    navigation_gravity[i] = (
        rotation.apply(
            gravity[i]
        )
    )


    # --------------------------------------------------------
    # LINEAR ACCELERATION
    # --------------------------------------------------------

    linear_acceleration = (
        acceleration[i]
        -
        gravity[i]
    )


    navigation_acceleration[i] = (
        rotation.apply(
            linear_acceleration
        )
    )


# ============================================================
# EXTRACT EULER
# ============================================================

roll = euler_angles[:, 0]

pitch = euler_angles[:, 1]

yaw = wrap_angle(
    euler_angles[:, 2]
)


# ============================================================
# GRAVITY RESULTS
# ============================================================

north_gravity = (
    navigation_gravity[:, 0]
)

east_gravity = (
    navigation_gravity[:, 1]
)

down_gravity = (
    navigation_gravity[:, 2]
)


horizontal_gravity = np.sqrt(
    north_gravity ** 2
    +
    east_gravity ** 2
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ORIENTATION RESULTS")
print("=" * 70)


print(
    f"\nRoll range  : "
    f"{np.min(roll):.2f}° to "
    f"{np.max(roll):.2f}°"
)

print(
    f"Pitch range : "
    f"{np.min(pitch):.2f}° to "
    f"{np.max(pitch):.2f}°"
)

print(
    f"Yaw range   : "
    f"{np.min(yaw):.2f}° to "
    f"{np.max(yaw):.2f}°"
)


# ============================================================
# GRAVITY VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "GRAVITY VALIDATION"
)

print(
    "=" * 70
)


print(
    f"\nNorth mean : "
    f"{np.mean(north_gravity):.4f} m/s²"
)

print(
    f"East mean  : "
    f"{np.mean(east_gravity):.4f} m/s²"
)

print(
    f"Down mean  : "
    f"{np.mean(down_gravity):.4f} m/s²"
)


print(
    f"\nNorth std : "
    f"{np.std(north_gravity):.4f}"
)

print(
    f"East std  : "
    f"{np.std(east_gravity):.4f}"
)

print(
    f"Down std  : "
    f"{np.std(down_gravity):.4f}"
)


print(
    f"\nHorizontal gravity median : "
    f"{np.median(horizontal_gravity):.4f} m/s²"
)

print(
    f"Horizontal gravity maximum : "
    f"{np.max(horizontal_gravity):.4f} m/s²"
)


# ============================================================
# NAVIGATION ACCELERATION
# ============================================================

acceleration_magnitude = np.linalg.norm(
    navigation_acceleration,
    axis=1
)


print(
    "\n" + "=" * 70
)

print(
    "NAVIGATION ACCELERATION"
)

print(
    "=" * 70
)


print(
    f"\nMedian magnitude : "
    f"{np.median(acceleration_magnitude):.4f} m/s²"
)

print(
    f"Maximum magnitude : "
    f"{np.max(acceleration_magnitude):.4f} m/s²"
)


# ============================================================
# PLOT ROLL
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    roll,
    label="Complementary Roll"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Roll (degrees)"
)

plt.title(
    "Complementary Filter Roll"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# PLOT PITCH
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    pitch,
    label="Complementary Pitch"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Pitch (degrees)"
)

plt.title(
    "Complementary Filter Pitch"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# PLOT YAW
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    yaw,
    label="Complementary Yaw"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Yaw (degrees)"
)

plt.title(
    "Complementary Filter Yaw"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# PLOT GRAVITY
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    north_gravity,
    label="North gravity"
)

plt.plot(
    df["time_seconds"],
    east_gravity,
    label="East gravity"
)

plt.plot(
    df["time_seconds"],
    down_gravity,
    label="Down gravity"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.axhline(
    9.80665,
    linestyle="--"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Gravity (m/s²)"
)

plt.title(
    "Gravity After Complementary Orientation Correction"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# PLOT NAVIGATION ACCELERATION
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    navigation_acceleration[:, 0],
    label="North acceleration"
)

plt.plot(
    df["time_seconds"],
    navigation_acceleration[:, 1],
    label="East acceleration"
)

plt.plot(
    df["time_seconds"],
    navigation_acceleration[:, 2],
    label="Down acceleration"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Acceleration (m/s²)"
)

plt.title(
    "Linear Acceleration in Navigation Frame"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5.2 COMPLETE")
print("=" * 70)

print(
    "\nComplementary orientation estimation completed."
)

print(
    "\nDo NOT integrate velocity yet."
)

print(
    "First inspect the gravity validation results."
)