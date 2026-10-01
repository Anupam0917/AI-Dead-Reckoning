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
INITIAL_SAMPLES = 100


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("QUATERNION ORIENTATION ESTIMATION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252"
)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


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

invalid_timestamps = df["timestamp"].isna().sum()

if invalid_timestamps > 0:
    raise ValueError(
        f"Found {invalid_timestamps} invalid timestamps."
    )

first_timestamp = df["timestamp"].iloc[0]

df["time_seconds"] = (
    df["timestamp"] - first_timestamp
).dt.total_seconds()

df["dt"] = df["time_seconds"].diff()


print("\nTimestamp validation:")
print(
    f"Invalid timestamps : "
    f"{invalid_timestamps}"
)

print(
    f"Start : "
    f"{df['timestamp'].iloc[0]}"
)

print(
    f"End   : "
    f"{df['timestamp'].iloc[-1]}"
)


# ============================================================
# EXTRACT ACCELEROMETER
# ============================================================

acceleration_columns = [
    "ACCELEROMETER_X_m_s²",
    "ACCELEROMETER_Y_m_s²",
    "ACCELEROMETER_Z_m_s²"
]

for column in acceleration_columns:

    if column not in df.columns:
        raise KeyError(
            f"Missing accelerometer column: {column}"
        )

acceleration = df[
    acceleration_columns
].to_numpy(dtype=float)


# ============================================================
# EXTRACT GRAVITY
# ============================================================

gravity_columns = [
    "GRAVITY_X_m_s²",
    "GRAVITY_Y_m_s²",
    "GRAVITY_Z_m_s²"
]

for column in gravity_columns:

    if column not in df.columns:
        raise KeyError(
            f"Missing gravity column: {column}"
        )

gravity = df[
    gravity_columns
].to_numpy(dtype=float)


# ============================================================
# EXTRACT MAGNETOMETER
# ============================================================

# The original dataset contains unusual Unicode characters
# in the magnetic-field unit names.
#
# Therefore we do NOT hard-code the complete column names.
#
# Instead, find columns beginning with:
#
# MAGNETIC_FIELD_X
# MAGNETIC_FIELD_Y
# MAGNETIC_FIELD_Z

magnetic_columns = []

for axis in ["X", "Y", "Z"]:

    matching_columns = [
        column
        for column in df.columns
        if column.startswith(
            f"MAGNETIC_FIELD_{axis}"
        )
    ]

    if len(matching_columns) != 1:

        raise ValueError(
            f"Could not uniquely identify "
            f"magnetometer {axis} column.\n"
            f"Found: {matching_columns}"
        )

    magnetic_columns.append(
        matching_columns[0]
    )


print("\nMagnetometer columns detected:")

for column in magnetic_columns:
    print(f" - {column}")


magnetometer = df[
    magnetic_columns
].to_numpy(dtype=float)


# ============================================================
# EXTRACT GYROSCOPE
# ============================================================

# Dataset labels:
#
# GYROSCOPE_Yaw
# GYROSCOPE_Pitch
# GYROSCOPE_Roll
#
# For the XYZ sensor vector we use:
#
# X = Roll
# Y = Pitch
# Z = Yaw
#
# This mapping will be validated later.

gyro_columns = [
    "GYROSCOPE_Roll_rad_s",
    "GYROSCOPE_Pitch_rad_s",
    "GYROSCOPE_Yaw_rad_s"
]

for column in gyro_columns:

    if column not in df.columns:

        raise KeyError(
            f"Missing gyroscope column: {column}"
        )

gyro_raw = df[
    gyro_columns
].to_numpy(dtype=float)


# ============================================================
# SENSOR SHAPE CHECK
# ============================================================

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
    f"{gyro_raw.shape}"
)

print(
    f"Magnetometer : "
    f"{magnetometer.shape}"
)


# ============================================================
# GYROSCOPE OUTLIER DETECTION
# ============================================================

gyro_magnitude = np.linalg.norm(
    gyro_raw,
    axis=1
)

gyro_outlier_mask = (
    gyro_magnitude >
    GYRO_OUTLIER_THRESHOLD
)


print("\nGyroscope outliers:")

print(
    f"Outliers : "
    f"{np.sum(gyro_outlier_mask)}"
)

print(
    f"Clean    : "
    f"{len(gyro_outlier_mask) - np.sum(gyro_outlier_mask)}"
)


# ============================================================
# CLEAN GYROSCOPE
# ============================================================

gyro = gyro_raw.copy()

for axis in range(3):

    series = pd.Series(
        gyro[:, axis]
    )

    # Replace outliers with NaN
    series[gyro_outlier_mask] = np.nan

    # Interpolate over corrupted samples
    series = series.interpolate(
        method="linear",
        limit_direction="both"
    )

    gyro[:, axis] = series.to_numpy()


clean_gyro_magnitude = np.linalg.norm(
    gyro,
    axis=1
)

print(
    f"Clean gyro maximum : "
    f"{np.max(clean_gyro_magnitude):.4f} rad/s"
)


# ============================================================
# GRAVITY COMPENSATION
# ============================================================

linear_acceleration = (
    acceleration - gravity
)

linear_acceleration_magnitude = np.linalg.norm(
    linear_acceleration,
    axis=1
)


print("\nLinear acceleration:")

print(
    f"Median : "
    f"{np.median(linear_acceleration_magnitude):.4f} m/s²"
)

print(
    f"Maximum: "
    f"{np.max(linear_acceleration_magnitude):.4f} m/s²"
)


# ============================================================
# TIMESTAMP GAP CHECK
# ============================================================

gap_mask = (
    df["dt"] > LARGE_GAP_THRESHOLD
)

gap_indices = np.where(
    gap_mask.to_numpy()
)[0]


print("\nTimestamp gaps:")

print(
    f"Gaps > {LARGE_GAP_THRESHOLD:.1f}s : "
    f"{len(gap_indices)}"
)

for index in gap_indices:

    print(
        f"Row {index}: "
        f"{df['dt'].iloc[index]:.3f} seconds"
    )


# ============================================================
# INITIAL SENSOR ESTIMATION
# ============================================================

print("\n" + "=" * 60)
print("INITIAL ORIENTATION")
print("=" * 60)

print(
    f"Using first {INITIAL_SAMPLES} samples."
)


# Median is more robust against individual sensor spikes.

initial_gravity = np.median(
    gravity[:INITIAL_SAMPLES],
    axis=0
)

initial_magnetic = np.median(
    magnetometer[:INITIAL_SAMPLES],
    axis=0
)


print(
    "\nInitial gravity vector:"
)

print(
    initial_gravity
)


print(
    "\nInitial magnetic vector:"
)

print(
    initial_magnetic
)


# ============================================================
# NORMALIZE GRAVITY
# ============================================================

gravity_norm = np.linalg.norm(
    initial_gravity
)

if gravity_norm < 1e-6:

    raise ValueError(
        "Gravity magnitude is too small."
    )

gravity_unit = (
    initial_gravity /
    gravity_norm
)


# ============================================================
# NORMALIZE MAGNETOMETER
# ============================================================

magnetic_norm = np.linalg.norm(
    initial_magnetic
)

if magnetic_norm < 1e-6:

    raise ValueError(
        "Magnetometer magnitude is too small."
    )

magnetic_unit = (
    initial_magnetic /
    magnetic_norm
)


# ============================================================
# REMOVE VERTICAL COMPONENT FROM MAGNETOMETER
# ============================================================

# We only want the horizontal magnetic direction
# for estimating heading.

magnetic_horizontal = (
    magnetic_unit
    -
    np.dot(
        magnetic_unit,
        gravity_unit
    )
    * gravity_unit
)


magnetic_horizontal_norm = np.linalg.norm(
    magnetic_horizontal
)

if magnetic_horizontal_norm < 1e-6:

    raise ValueError(
        "Magnetometer horizontal component "
        "is too small to estimate heading."
    )


magnetic_horizontal = (
    magnetic_horizontal /
    magnetic_horizontal_norm
)


# ============================================================
# BUILD INITIAL NED FRAME
# ============================================================

# Navigation frame:
#
# X = North
# Y = East
# Z = Down
#
# Gravity provides Down.
# Magnetometer provides approximate magnetic North.

body_down = gravity_unit

body_north = magnetic_horizontal


# East direction
body_east = np.cross(
    body_down,
    body_north
)


east_norm = np.linalg.norm(
    body_east
)

if east_norm < 1e-6:

    raise ValueError(
        "Could not construct East direction."
    )


body_east /= east_norm


# Re-orthogonalize North
body_north = np.cross(
    body_east,
    body_down
)

body_north /= np.linalg.norm(
    body_north
)


# ============================================================
# BODY BASIS MATRIX
# ============================================================

body_basis = np.column_stack(
    [
        body_north,
        body_east,
        body_down
    ]
)


# ============================================================
# INITIAL ROTATION
# ============================================================

# Rotation maps body-frame vectors
# into navigation-frame vectors.

initial_rotation_matrix = (
    body_basis.T
)


initial_rotation = Rotation.from_matrix(
    initial_rotation_matrix
)


# scipy quaternion format:
#
# [x, y, z, w]

initial_quaternion = (
    initial_rotation.as_quat()
)


print(
    "\nInitial quaternion [x, y, z, w]:"
)

print(
    initial_quaternion
)


# ============================================================
# QUATERNION INTEGRATION
# ============================================================

print("\nIntegrating quaternion orientation...")

N = len(df)

rotations = [
    initial_rotation
]

current_rotation = (
    initial_rotation
)


for i in range(1, N):

    dt = df["dt"].iloc[i]


    # --------------------------------------------------------
    # INVALID TIME STEP
    # --------------------------------------------------------

    if not np.isfinite(dt) or dt <= 0:

        rotations.append(
            current_rotation
        )

        continue


    # --------------------------------------------------------
    # LARGE TIME GAP
    # --------------------------------------------------------

    if dt > LARGE_GAP_THRESHOLD:

        # There is no reliable IMU information during
        # the missing interval.
        #
        # Do not integrate through the gap.

        rotations.append(
            current_rotation
        )

        continue


    # --------------------------------------------------------
    # ANGULAR DISPLACEMENT
    # --------------------------------------------------------

    angular_velocity = gyro[i]

    rotation_vector = (
        angular_velocity * dt
    )


    # --------------------------------------------------------
    # DELTA ROTATION
    # --------------------------------------------------------

    delta_rotation = (
        Rotation.from_rotvec(
            rotation_vector
        )
    )


    # --------------------------------------------------------
    # UPDATE QUATERNION
    # --------------------------------------------------------

    current_rotation = (
        current_rotation *
        delta_rotation
    )


    rotations.append(
        current_rotation
    )


# ============================================================
# CONVERT QUATERNIONS TO EULER ANGLES
# ============================================================

print(
    "\nConverting orientation to Euler angles..."
)


euler_angles = np.zeros(
    (N, 3)
)


for i, rotation in enumerate(rotations):

    euler_angles[i] = (
        rotation.as_euler(
            "xyz",
            degrees=True
        )
    )


roll = euler_angles[:, 0]

pitch = euler_angles[:, 1]

yaw = euler_angles[:, 2]


# ============================================================
# NORMALIZE YAW
# ============================================================

yaw = (
    (yaw + 180.0) % 360.0
) - 180.0


# ============================================================
# ORIENTATION STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("ORIENTATION RESULTS")
print("=" * 60)


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


print(
    f"Roll median : "
    f"{np.median(roll):.2f}°"
)


print(
    f"Pitch median: "
    f"{np.median(pitch):.2f}°"
)


print(
    f"Yaw median  : "
    f"{np.median(yaw):.2f}°"
)


# ============================================================
# ROTATE LINEAR ACCELERATION
# ============================================================

print(
    "\nRotating linear acceleration "
    "into navigation frame..."
)


navigation_acceleration = np.zeros(
    (N, 3)
)


for i, rotation in enumerate(rotations):

    navigation_acceleration[i] = (
        rotation.apply(
            linear_acceleration[i]
        )
    )


# ============================================================
# NAVIGATION ACCELERATION STATISTICS
# ============================================================

navigation_acceleration_magnitude = (
    np.linalg.norm(
        navigation_acceleration,
        axis=1
    )
)


print(
    "\nNavigation-frame acceleration:"
)


print(
    f"Median magnitude : "
    f"{np.median(navigation_acceleration_magnitude):.4f} m/s²"
)


print(
    f"Maximum magnitude: "
    f"{np.max(navigation_acceleration_magnitude):.4f} m/s²"
)


# ============================================================
# PLOT 1: ROLL
# ============================================================

print("\nGenerating Roll plot...")

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    roll,
    linewidth=0.8
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Roll (degrees)"
)

plt.title(
    "Quaternion Estimated Roll"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 2: PITCH
# ============================================================

print("Generating Pitch plot...")

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    pitch,
    linewidth=0.8
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Pitch (degrees)"
)

plt.title(
    "Quaternion Estimated Pitch"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 3: YAW
# ============================================================

print("Generating Yaw plot...")

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    yaw,
    linewidth=0.8
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Yaw (degrees)"
)

plt.title(
    "Quaternion Estimated Yaw"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 4: NAVIGATION ACCELERATION
# ============================================================

print(
    "Generating navigation acceleration plot..."
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    navigation_acceleration[:, 0],
    label="North acceleration",
    linewidth=0.8
)

plt.plot(
    df["time_seconds"],
    navigation_acceleration[:, 1],
    label="East acceleration",
    linewidth=0.8
)

plt.plot(
    df["time_seconds"],
    navigation_acceleration[:, 2],
    label="Down acceleration",
    linewidth=0.8
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

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 COMPLETE")
print("=" * 60)

print(
    "Quaternion orientation estimation completed."
)

print(
    "\nImportant:"
)

print(
    "This is a baseline orientation estimator."
)

print(
    "Yaw will accumulate drift because the current "
    "implementation propagates orientation mainly "
    "using the gyroscope."
)

print(
    "\nNext step:"
)

print(
    "Validate orientation before velocity and "
    "position integration."
)