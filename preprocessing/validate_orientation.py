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
# HELPER FUNCTIONS
# ============================================================

def find_column(df, prefix):
    """
    Find exactly one column beginning with a given prefix.
    This avoids problems caused by broken Unicode characters
    in the original dataset column names.
    """

    matches = [
        column
        for column in df.columns
        if column.startswith(prefix)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely find column "
            f"'{prefix}'. Found: {matches}"
        )

    return matches[0]


def wrap_angle(angle):
    """
    Wrap angle to [-180, 180] degrees.
    """

    return (
        (angle + 180.0) % 360.0
    ) - 180.0


def angular_error_deg(estimated, reference):
    """
    Calculate wrapped angular error.
    """

    return wrap_angle(
        estimated - reference
    )


def remove_constant_offset(
    estimated,
    reference
):
    """
    Remove the initial median offset.

    This is useful because the dataset orientation and our
    navigation frame may use different absolute heading
    conventions.

    We are NOT changing the estimated trajectory.
    This is only for comparison.
    """

    initial_count = min(
        INITIAL_SAMPLES,
        len(estimated)
    )

    error = angular_error_deg(
        estimated[:initial_count],
        reference[:initial_count]
    )

    offset = np.median(error)

    corrected = wrap_angle(
        estimated - offset
    )

    return corrected, offset


def calculate_rmse(error):
    """
    Root Mean Square Error.
    """

    return np.sqrt(
        np.mean(
            np.square(error)
        )
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("ORIENTATION VALIDATION")
print("=" * 70)


# ============================================================
# LOAD DATASET
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


first_timestamp = df["timestamp"].iloc[0]

df["time_seconds"] = (
    df["timestamp"] -
    first_timestamp
).dt.total_seconds()

df["dt"] = (
    df["time_seconds"].diff()
)


# ============================================================
# EXTRACT ACCELERATION
# ============================================================

acceleration_columns = [
    "ACCELEROMETER_X_m_s²",
    "ACCELEROMETER_Y_m_s²",
    "ACCELEROMETER_Z_m_s²"
]

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

gravity = df[
    gravity_columns
].to_numpy(dtype=float)


# ============================================================
# EXTRACT MAGNETOMETER
# ============================================================

magnetic_columns = []

for axis in ["X", "Y", "Z"]:

    column = find_column(
        df,
        f"MAGNETIC_FIELD_{axis}"
    )

    magnetic_columns.append(
        column
    )


magnetometer = df[
    magnetic_columns
].to_numpy(dtype=float)


# ============================================================
# EXTRACT GYROSCOPE
# ============================================================

# Baseline mapping currently being tested:
#
# X = Roll
# Y = Pitch
# Z = Yaw

gyro_columns = [
    "GYROSCOPE_Roll_rad_s",
    "GYROSCOPE_Pitch_rad_s",
    "GYROSCOPE_Yaw_rad_s"
]

gyro_raw = df[
    gyro_columns
].to_numpy(dtype=float)


# ============================================================
# CLEAN GYROSCOPE
# ============================================================

gyro_magnitude = np.linalg.norm(
    gyro_raw,
    axis=1
)

gyro_outlier_mask = (
    gyro_magnitude >
    GYRO_OUTLIER_THRESHOLD
)

gyro = gyro_raw.copy()


for axis in range(3):

    series = pd.Series(
        gyro[:, axis]
    )

    series[
        gyro_outlier_mask
    ] = np.nan

    series = series.interpolate(
        method="linear",
        limit_direction="both"
    )

    gyro[:, axis] = (
        series.to_numpy()
    )


print("\nGyroscope preprocessing:")

print(
    f"Outliers : "
    f"{np.sum(gyro_outlier_mask)}"
)


# ============================================================
# GRAVITY COMPENSATION
# ============================================================

linear_acceleration = (
    acceleration -
    gravity
)


# ============================================================
# INITIAL ORIENTATION
# ============================================================

print("\nCalculating initial orientation...")

initial_gravity = np.median(
    gravity[:INITIAL_SAMPLES],
    axis=0
)

initial_magnetic = np.median(
    magnetometer[:INITIAL_SAMPLES],
    axis=0
)


# Normalize gravity

gravity_norm = np.linalg.norm(
    initial_gravity
)

gravity_unit = (
    initial_gravity /
    gravity_norm
)


# Normalize magnetometer

magnetic_norm = np.linalg.norm(
    initial_magnetic
)

magnetic_unit = (
    initial_magnetic /
    magnetic_norm
)


# ============================================================
# INITIAL MAGNETIC NORTH
# ============================================================

magnetic_horizontal = (
    magnetic_unit
    -
    np.dot(
        magnetic_unit,
        gravity_unit
    )
    *
    gravity_unit
)


magnetic_horizontal_norm = (
    np.linalg.norm(
        magnetic_horizontal
    )
)


if magnetic_horizontal_norm < 1e-6:

    raise ValueError(
        "Magnetometer horizontal component "
        "is too small."
    )


magnetic_horizontal /= (
    magnetic_horizontal_norm
)


# ============================================================
# INITIAL NED FRAME
# ============================================================

body_down = gravity_unit

body_north = magnetic_horizontal

body_east = np.cross(
    body_down,
    body_north
)

body_east /= np.linalg.norm(
    body_east
)

body_north = np.cross(
    body_east,
    body_down
)

body_north /= np.linalg.norm(
    body_north
)


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

initial_rotation_matrix = (
    body_basis.T
)

current_rotation = (
    Rotation.from_matrix(
        initial_rotation_matrix
    )
)


initial_quaternion = (
    current_rotation.as_quat()
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

print(
    "\nIntegrating quaternion..."
)

N = len(df)

rotations = [
    current_rotation
]


for i in range(1, N):

    dt = df["dt"].iloc[i]


    # --------------------------------------------------------
    # INVALID DT
    # --------------------------------------------------------

    if (
        not np.isfinite(dt)
        or dt <= 0
    ):

        rotations.append(
            current_rotation
        )

        continue


    # --------------------------------------------------------
    # LARGE GAP
    # --------------------------------------------------------

    if dt > LARGE_GAP_THRESHOLD:

        rotations.append(
            current_rotation
        )

        continue


    # --------------------------------------------------------
    # GYRO ROTATION
    # --------------------------------------------------------

    angular_velocity = gyro[i]

    rotation_vector = (
        angular_velocity *
        dt
    )


    delta_rotation = (
        Rotation.from_rotvec(
            rotation_vector
        )
    )


    current_rotation = (
        current_rotation *
        delta_rotation
    )


    rotations.append(
        current_rotation
    )


# ============================================================
# CONVERT QUATERNION TO EULER
# ============================================================

print(
    "Converting quaternion to Euler angles..."
)


quaternion_euler = np.zeros(
    (N, 3)
)


for i, rotation in enumerate(
    rotations
):

    quaternion_euler[i] = (
        rotation.as_euler(
            "xyz",
            degrees=True
        )
    )


quat_roll = (
    quaternion_euler[:, 0]
)

quat_pitch = (
    quaternion_euler[:, 1]
)

quat_yaw = wrap_angle(
    quaternion_euler[:, 2]
)


# ============================================================
# DATASET ORIENTATION
# ============================================================

print(
    "\nExtracting dataset orientation..."
)


dataset_roll_column = find_column(
    df,
    "ORIENTATION_Roll"
)

dataset_pitch_column = find_column(
    df,
    "ORIENTATION_Pitch"
)

dataset_yaw_column = find_column(
    df,
    "ORIENTATION_Yaw"
)


dataset_roll = pd.to_numeric(
    df[dataset_roll_column],
    errors="coerce"
).to_numpy(dtype=float)


dataset_pitch = pd.to_numeric(
    df[dataset_pitch_column],
    errors="coerce"
).to_numpy(dtype=float)


dataset_yaw = pd.to_numeric(
    df[dataset_yaw_column],
    errors="coerce"
).to_numpy(dtype=float)


# Wrap reference angles

dataset_roll = wrap_angle(
    dataset_roll
)

dataset_pitch = wrap_angle(
    dataset_pitch
)

dataset_yaw = wrap_angle(
    dataset_yaw
)


print(
    "\nDataset orientation columns:"
)

print(
    f"Roll  : {dataset_roll_column}"
)

print(
    f"Pitch : {dataset_pitch_column}"
)

print(
    f"Yaw   : {dataset_yaw_column}"
)


# ============================================================
# INITIAL OFFSET CORRECTION
# ============================================================

print(
    "\nCalculating initial orientation offsets..."
)


quat_roll_aligned, roll_offset = (
    remove_constant_offset(
        quat_roll,
        dataset_roll
    )
)


quat_pitch_aligned, pitch_offset = (
    remove_constant_offset(
        quat_pitch,
        dataset_pitch
    )
)


quat_yaw_aligned, yaw_offset = (
    remove_constant_offset(
        quat_yaw,
        dataset_yaw
    )
)


print(
    f"Roll offset  : {roll_offset:.2f}°"
)

print(
    f"Pitch offset : {pitch_offset:.2f}°"
)

print(
    f"Yaw offset   : {yaw_offset:.2f}°"
)


# ============================================================
# ANGULAR ERRORS
# ============================================================

roll_error = angular_error_deg(
    quat_roll_aligned,
    dataset_roll
)

pitch_error = angular_error_deg(
    quat_pitch_aligned,
    dataset_pitch
)

yaw_error = angular_error_deg(
    quat_yaw_aligned,
    dataset_yaw
)


# ============================================================
# RMSE
# ============================================================

roll_rmse = calculate_rmse(
    roll_error
)

pitch_rmse = calculate_rmse(
    pitch_error
)

yaw_rmse = calculate_rmse(
    yaw_error
)


print("\n" + "=" * 70)
print("ORIENTATION COMPARISON")
print("=" * 70)


print(
    f"\nRoll RMSE  : "
    f"{roll_rmse:.2f}°"
)

print(
    f"Pitch RMSE : "
    f"{pitch_rmse:.2f}°"
)

print(
    f"Yaw RMSE   : "
    f"{yaw_rmse:.2f}°"
)


# ============================================================
# EARLY VS LATE ERROR
# ============================================================

early_count = min(
    1000,
    N
)

late_start = max(
    N - 1000,
    0
)


early_roll_rmse = calculate_rmse(
    roll_error[:early_count]
)

late_roll_rmse = calculate_rmse(
    roll_error[late_start:]
)


early_pitch_rmse = calculate_rmse(
    pitch_error[:early_count]
)

late_pitch_rmse = calculate_rmse(
    pitch_error[late_start:]
)


early_yaw_rmse = calculate_rmse(
    yaw_error[:early_count]
)

late_yaw_rmse = calculate_rmse(
    yaw_error[late_start:]
)


print("\nEarly vs late error:")

print(
    f"Roll  : "
    f"{early_roll_rmse:.2f}° → "
    f"{late_roll_rmse:.2f}°"
)

print(
    f"Pitch : "
    f"{early_pitch_rmse:.2f}° → "
    f"{late_pitch_rmse:.2f}°"
)

print(
    f"Yaw   : "
    f"{early_yaw_rmse:.2f}° → "
    f"{late_yaw_rmse:.2f}°"
)


# ============================================================
# GRAVITY VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "GRAVITY CONSISTENCY TEST"
)

print(
    "=" * 70
)


navigation_gravity = np.zeros(
    (N, 3)
)


for i, rotation in enumerate(
    rotations
):

    navigation_gravity[i] = (
        rotation.apply(
            gravity[i]
        )
    )


north_gravity = (
    navigation_gravity[:, 0]
)

east_gravity = (
    navigation_gravity[:, 1]
)

down_gravity = (
    navigation_gravity[:, 2]
)


print(
    "\nExpected NED gravity:"
)

print(
    "North ≈ 0 m/s²"
)

print(
    "East  ≈ 0 m/s²"
)

print(
    "Down  ≈ +9.81 m/s²"
)


print(
    "\nMeasured statistics:"
)

print(
    f"North gravity mean : "
    f"{np.mean(north_gravity):.4f}"
)

print(
    f"East gravity mean  : "
    f"{np.mean(east_gravity):.4f}"
)

print(
    f"Down gravity mean  : "
    f"{np.mean(down_gravity):.4f}"
)


print(
    f"\nNorth gravity std  : "
    f"{np.std(north_gravity):.4f}"
)

print(
    f"East gravity std   : "
    f"{np.std(east_gravity):.4f}"
)

print(
    f"Down gravity std   : "
    f"{np.std(down_gravity):.4f}"
)


# ============================================================
# GRAVITY ERROR
# ============================================================

horizontal_gravity = np.sqrt(
    north_gravity ** 2
    +
    east_gravity ** 2
)


print(
    "\nHorizontal gravity magnitude:"
)

print(
    f"Median : "
    f"{np.median(horizontal_gravity):.4f} m/s²"
)

print(
    f"Maximum: "
    f"{np.max(horizontal_gravity):.4f} m/s²"
)


# ============================================================
# NAVIGATION ACCELERATION
# ============================================================

navigation_acceleration = np.zeros(
    (N, 3)
)


for i, rotation in enumerate(
    rotations
):

    navigation_acceleration[i] = (
        rotation.apply(
            linear_acceleration[i]
        )
    )


# ============================================================
# ACCELERATION MAGNITUDE CHECK
# ============================================================

body_acceleration_magnitude = (
    np.linalg.norm(
        linear_acceleration,
        axis=1
    )
)

nav_acceleration_magnitude = (
    np.linalg.norm(
        navigation_acceleration,
        axis=1
    )
)


magnitude_difference = (
    nav_acceleration_magnitude
    -
    body_acceleration_magnitude
)


print(
    "\n" + "=" * 70
)

print(
    "ACCELERATION ROTATION CONSISTENCY"
)

print(
    "=" * 70
)


print(
    "\nBecause rotation should preserve vector magnitude:"
)

print(
    f"Maximum magnitude difference: "
    f"{np.max(np.abs(magnitude_difference)):.8f} m/s²"
)

print(
    f"Mean magnitude difference: "
    f"{np.mean(np.abs(magnitude_difference)):.8f} m/s²"
)


# ============================================================
# PLOT 1: ROLL COMPARISON
# ============================================================

print(
    "\nGenerating Roll comparison..."
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    dataset_roll,
    label="Dataset Roll",
    alpha=0.7
)

plt.plot(
    df["time_seconds"],
    quat_roll_aligned,
    label="Quaternion Roll",
    alpha=0.8
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Roll (degrees)"
)

plt.title(
    "Dataset vs Quaternion Roll"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 2: PITCH COMPARISON
# ============================================================

print(
    "Generating Pitch comparison..."
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    dataset_pitch,
    label="Dataset Pitch",
    alpha=0.7
)

plt.plot(
    df["time_seconds"],
    quat_pitch_aligned,
    label="Quaternion Pitch",
    alpha=0.8
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Pitch (degrees)"
)

plt.title(
    "Dataset vs Quaternion Pitch"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 3: YAW COMPARISON
# ============================================================

print(
    "Generating Yaw comparison..."
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    dataset_yaw,
    label="Dataset Yaw",
    alpha=0.7
)

plt.plot(
    df["time_seconds"],
    quat_yaw_aligned,
    label="Quaternion Yaw",
    alpha=0.8
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Yaw (degrees)"
)

plt.title(
    "Dataset vs Quaternion Yaw"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 4: ORIENTATION ERROR
# ============================================================

print(
    "Generating orientation error plot..."
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    df["time_seconds"],
    roll_error,
    label="Roll error"
)

plt.plot(
    df["time_seconds"],
    pitch_error,
    label="Pitch error"
)

plt.plot(
    df["time_seconds"],
    yaw_error,
    label="Yaw error"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Angular error (degrees)"
)

plt.title(
    "Quaternion Orientation Error"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 5: NAVIGATION GRAVITY
# ============================================================

print(
    "Generating gravity validation plot..."
)


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
    "Gravity in Navigation Frame"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 5.1 COMPLETE")
print("=" * 70)

print(
    "\nOrientation validation completed."
)

print(
    "\nThe results will determine whether the current "
    "quaternion baseline is suitable for navigation."
)

print(
    "\nDo NOT integrate velocity or position yet."
)