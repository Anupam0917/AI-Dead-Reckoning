import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V9.4 STEP 1
# STATIONARY DETECTION + IMU BIAS ESTIMATION
# ============================================================

print("=" * 70)
print("V9.4 STATIONARY DETECTION + IMU BIAS ESTIMATION")
print("=" * 70)


# ------------------------------------------------------------
# PATH
# ------------------------------------------------------------

DATA_PATH = (
    "data/raw/"
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

OUTPUT_DIR = "outputs"
PROCESSED_DIR = "data/processed"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading IO-VNBD dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252"
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print(
    "Rows loaded:",
    len(df)
)


# ------------------------------------------------------------
# COLUMN NAMES
# ------------------------------------------------------------

timestamp_col = (
    "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
)

acc_cols = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)"
]

gravity_cols = [
    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)"
]

gyro_cols = [
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)"
]


required = (
    [timestamp_col]
    + acc_cols
    + gravity_cols
    + gyro_cols
)


missing = [
    c for c in required
    if c not in df.columns
]


if missing:

    print("\nMissing columns:")

    for c in missing:
        print(" ", c)

    raise ValueError(
        "Required columns are missing."
    )


# ------------------------------------------------------------
# TIMESTAMP
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df[timestamp_col],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)


# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

for col in (
    acc_cols
    + gravity_cols
    + gyro_cols
):

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ------------------------------------------------------------
# CLEAN SENSOR DATA
# ------------------------------------------------------------

df[
    acc_cols
    + gravity_cols
    + gyro_cols
] = (
    df[
        acc_cols
        + gravity_cols
        + gyro_cols
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .interpolate(
        method="linear",
        limit_direction="both"
    )
)


# ------------------------------------------------------------
# ACCELEROMETER
# ------------------------------------------------------------

ax = df[
    acc_cols[0]
].values

ay = df[
    acc_cols[1]
].values

az = df[
    acc_cols[2]
].values


# ------------------------------------------------------------
# GRAVITY
# ------------------------------------------------------------

gx = df[
    gravity_cols[0]
].values

gy = df[
    gravity_cols[1]
].values

gz = df[
    gravity_cols[2]
].values


# ------------------------------------------------------------
# GYROSCOPE
# ------------------------------------------------------------

gyro_x = df[
    gyro_cols[0]
].values

gyro_y = df[
    gyro_cols[1]
].values

gyro_z = df[
    gyro_cols[2]
].values


# ------------------------------------------------------------
# LINEAR ACCELERATION
#
# Accelerometer ≈ gravity + linear acceleration
# ------------------------------------------------------------

lin_x = ax - gx
lin_y = ay - gy
lin_z = az - gz


# ------------------------------------------------------------
# MAGNITUDES
# ------------------------------------------------------------

acc_mag = np.sqrt(
    ax**2 +
    ay**2 +
    az**2
)

gravity_mag = np.sqrt(
    gx**2 +
    gy**2 +
    gz**2
)

linear_acc_mag = np.sqrt(
    lin_x**2 +
    lin_y**2 +
    lin_z**2
)

gyro_mag = np.sqrt(
    gyro_x**2 +
    gyro_y**2 +
    gyro_z**2
)


# ------------------------------------------------------------
# STATIONARY DETECTION
# ------------------------------------------------------------

print("\nDetecting stationary periods...")


# We use short rolling windows.
#
# A stationary phone/vehicle should generally have:
#
# 1. Low linear acceleration
# 2. Low angular velocity
# 3. Accelerometer magnitude close to gravity


window = 20


linear_acc_mean = (
    pd.Series(linear_acc_mag)
    .rolling(
        window=window,
        min_periods=1
    )
    .mean()
    .values
)


gyro_mean = (
    pd.Series(gyro_mag)
    .rolling(
        window=window,
        min_periods=1
    )
    .mean()
    .values
)


acc_mag_error = np.abs(
    acc_mag - gravity_mag
)


acc_mag_error_mean = (
    pd.Series(acc_mag_error)
    .rolling(
        window=window,
        min_periods=1
    )
    .mean()
    .values
)


# ------------------------------------------------------------
# THRESHOLDS
# ------------------------------------------------------------

# These are intentionally conservative starting thresholds.
#
# We will inspect the resulting stationary percentage
# before using these values inside the EKF.


LINEAR_ACC_THRESHOLD = 0.35

GYRO_THRESHOLD = 0.08

GRAVITY_ERROR_THRESHOLD = 0.30


stationary = (
    (linear_acc_mean < LINEAR_ACC_THRESHOLD)
    &
    (gyro_mean < GYRO_THRESHOLD)
    &
    (
        acc_mag_error_mean
        < GRAVITY_ERROR_THRESHOLD
    )
)


# ------------------------------------------------------------
# REMOVE VERY SHORT STATIONARY EVENTS
# ------------------------------------------------------------

# Require at least ~1 second of continuous
# stationary detection.


min_stationary_samples = 10


stationary_series = (
    pd.Series(
        stationary.astype(int)
    )
)


groups = (
    stationary_series
    .ne(
        stationary_series.shift()
    )
    .cumsum()
)


group_sizes = (
    stationary_series
    .groupby(groups)
    .transform("size")
)


stationary = (
    stationary
    &
    (
        group_sizes.values
        >= min_stationary_samples
    )
)


# ------------------------------------------------------------
# STORE FEATURES
# ------------------------------------------------------------

df["linear_acc_mag"] = (
    linear_acc_mag
)

df["gyro_mag"] = (
    gyro_mag
)

df["acc_mag"] = (
    acc_mag
)

df["gravity_mag"] = (
    gravity_mag
)

df["stationary"] = (
    stationary
)


# ------------------------------------------------------------
# BASIC STATISTICS
# ------------------------------------------------------------

stationary_count = int(
    stationary.sum()
)

stationary_percentage = (
    stationary_count
    /
    len(df)
    *
    100
)


print("\n" + "=" * 70)
print("STATIONARY DETECTION RESULTS")
print("=" * 70)

print(
    "\nStationary samples:",
    stationary_count
)

print(
    "Total samples:",
    len(df)
)

print(
    "Stationary percentage:",
    f"{stationary_percentage:.2f}%"
)


# ------------------------------------------------------------
# BIAS ESTIMATION
# ------------------------------------------------------------

if stationary_count < 20:

    print(
        "\nWARNING:"
    )

    print(
        "Too few stationary samples "
        "for reliable bias estimation."
    )

    print(
        "We will NOT create a bias estimate."
    )

else:

    print(
        "\nEstimating IMU bias from "
        "stationary samples..."
    )


    # --------------------------------------------------------
    # GYROSCOPE BIAS
    # --------------------------------------------------------

    gyro_bias_x = np.mean(
        gyro_x[stationary]
    )

    gyro_bias_y = np.mean(
        gyro_y[stationary]
    )

    gyro_bias_z = np.mean(
        gyro_z[stationary]
    )


    # --------------------------------------------------------
    # ACCELEROMETER RESIDUAL
    # --------------------------------------------------------

    # IMPORTANT:
    #
    # Raw accelerometer contains gravity.
    #
    # Therefore we do NOT call its mean
    # "accelerometer bias" directly.
    #
    # Instead we calculate:
    #
    # accelerometer - gravity
    #
    # during stationary periods.


    accel_residual_x = (
        lin_x[stationary]
    )

    accel_residual_y = (
        lin_y[stationary]
    )

    accel_residual_z = (
        lin_z[stationary]
    )


    accel_bias_x = np.mean(
        accel_residual_x
    )

    accel_bias_y = np.mean(
        accel_residual_y
    )

    accel_bias_z = np.mean(
        accel_residual_z
    )


    # --------------------------------------------------------
    # STANDARD DEVIATION
    # --------------------------------------------------------

    accel_std_x = np.std(
        accel_residual_x
    )

    accel_std_y = np.std(
        accel_residual_y
    )

    accel_std_z = np.std(
        accel_residual_z
    )


    gyro_std_x = np.std(
        gyro_x[stationary]
    )

    gyro_std_y = np.std(
        gyro_y[stationary]
    )

    gyro_std_z = np.std(
        gyro_z[stationary]
    )


    # --------------------------------------------------------
    # PRINT BIAS
    # --------------------------------------------------------

    print("\nAccelerometer residual / bias estimate:")

    print(
        f"  X: {accel_bias_x:.6f} m/s²"
    )

    print(
        f"  Y: {accel_bias_y:.6f} m/s²"
    )

    print(
        f"  Z: {accel_bias_z:.6f} m/s²"
    )


    print("\nAccelerometer noise:")

    print(
        f"  X: {accel_std_x:.6f} m/s²"
    )

    print(
        f"  Y: {accel_std_y:.6f} m/s²"
    )

    print(
        f"  Z: {accel_std_z:.6f} m/s²"
    )


    print("\nGyroscope bias estimate:")

    print(
        f"  X: {gyro_bias_x:.6f} rad/s"
    )

    print(
        f"  Y: {gyro_bias_y:.6f} rad/s"
    )

    print(
        f"  Z: {gyro_bias_z:.6f} rad/s"
    )


    print("\nGyroscope noise:")

    print(
        f"  X: {gyro_std_x:.6f} rad/s"
    )

    print(
        f"  Y: {gyro_std_y:.6f} rad/s"
    )

    print(
        f"  Z: {gyro_std_z:.6f} rad/s"
    )


    # --------------------------------------------------------
    # SAVE BIAS
    # --------------------------------------------------------

    bias_data = pd.DataFrame({

        "sensor": [
            "accelerometer_x",
            "accelerometer_y",
            "accelerometer_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z"
        ],

        "bias": [
            accel_bias_x,
            accel_bias_y,
            accel_bias_z,
            gyro_bias_x,
            gyro_bias_y,
            gyro_bias_z
        ],

        "noise_std": [
            accel_std_x,
            accel_std_y,
            accel_std_z,
            gyro_std_x,
            gyro_std_y,
            gyro_std_z
        ]
    })


    bias_path = (
        "data/processed/"
        "prototype_v9_imu_bias.csv"
    )


    bias_data.to_csv(
        bias_path,
        index=False
    )


    print(
        "\nSaved bias estimates:"
    )

    print(
        bias_path
    )


# ------------------------------------------------------------
# SAVE STATIONARY DATA
# ------------------------------------------------------------

stationary_output = df[
    [
        "timestamp",
        "linear_acc_mag",
        "gyro_mag",
        "acc_mag",
        "gravity_mag",
        "stationary"
    ]
].copy()


stationary_path = (
    "data/processed/"
    "prototype_v9_stationary_detection.csv"
)


stationary_output.to_csv(
    stationary_path,
    index=False
)


# ------------------------------------------------------------
# PLOT 1
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    df["timestamp"],
    linear_acc_mag,
    label="Linear acceleration"
)

plt.axhline(
    LINEAR_ACC_THRESHOLD,
    linestyle="--",
    label="Stationary threshold"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Linear acceleration (m/s²)"
)

plt.title(
    "V9.4 Linear Acceleration and Stationary Threshold"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plot1 = (
    "outputs/"
    "prototype_v9_stationary_acceleration.png"
)


plt.savefig(
    plot1,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# PLOT 2
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    df["timestamp"],
    gyro_mag,
    label="Gyroscope magnitude"
)

plt.axhline(
    GYRO_THRESHOLD,
    linestyle="--",
    label="Stationary threshold"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Angular velocity (rad/s)"
)

plt.title(
    "V9.4 Gyroscope Activity and Stationary Threshold"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plot2 = (
    "outputs/"
    "prototype_v9_stationary_gyro.png"
)


plt.savefig(
    plot2,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# PLOT 3
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 3)
)

plt.plot(
    df["timestamp"],
    stationary.astype(int)
)

plt.yticks(
    [0, 1],
    ["Moving", "Stationary"]
)

plt.xlabel(
    "Time"
)

plt.title(
    "V9.4 Stationary Detection"
)

plt.grid(True)

plt.tight_layout()


plot3 = (
    "outputs/"
    "prototype_v9_stationary_detection.png"
)


plt.savefig(
    plot3,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V9.4 STEP 1 COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(
    "1.",
    stationary_path
)

if stationary_count >= 20:

    print(
        "2.",
        "data/processed/"
        "prototype_v9_imu_bias.csv"
    )

print(
    "3.",
    plot1
)

print(
    "4.",
    plot2
)

print(
    "5.",
    plot3
)

print("\nNext step:")
print(
    "Use stationary periods to initialize "
    "and update the bias-aware EKF."
)

print("=" * 70)