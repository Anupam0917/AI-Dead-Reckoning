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

GYRO_OUTLIER_THRESHOLD = 0.433104


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("IMU PREPROCESSING VISUALIZATION")
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

first_timestamp = df["timestamp"].iloc[0]

df["time_seconds"] = (
    df["timestamp"] - first_timestamp
).dt.total_seconds()

df["dt"] = df["time_seconds"].diff()


# ============================================================
# SENSOR DATA
# ============================================================

acceleration = df[
    [
        "ACCELEROMETER_X_m_s²",
        "ACCELEROMETER_Y_m_s²",
        "ACCELEROMETER_Z_m_s²"
    ]
].to_numpy(dtype=float)


gyro = df[
    [
        "GYROSCOPE_Yaw_rad_s",
        "GYROSCOPE_Pitch_rad_s",
        "GYROSCOPE_Roll_rad_s"
    ]
].to_numpy(dtype=float)


gravity = df[
    [
        "GRAVITY_X_m_s²",
        "GRAVITY_Y_m_s²",
        "GRAVITY_Z_m_s²"
    ]
].to_numpy(dtype=float)


# ============================================================
# GYRO CLEANING
# ============================================================

gyro_magnitude = np.linalg.norm(
    gyro,
    axis=1
)

gyro_outlier_mask = (
    gyro_magnitude > GYRO_OUTLIER_THRESHOLD
)


gyro_clean = gyro.copy()


for axis in range(3):

    series = pd.Series(
        gyro_clean[:, axis]
    )

    series[gyro_outlier_mask] = np.nan

    series = series.interpolate(
        method="linear",
        limit_direction="both"
    )

    gyro_clean[:, axis] = series.to_numpy()


# ============================================================
# LINEAR ACCELERATION
# ============================================================

linear_acceleration = (
    acceleration - gravity
)


# ============================================================
# MAGNITUDES
# ============================================================

acceleration_magnitude = np.linalg.norm(
    acceleration,
    axis=1
)

linear_acceleration_magnitude = np.linalg.norm(
    linear_acceleration,
    axis=1
)

gyro_clean_magnitude = np.linalg.norm(
    gyro_clean,
    axis=1
)


# ============================================================
# PLOT 1
# RAW VS CLEANED GYROSCOPE
# ============================================================

print("\nGenerating gyro plot...")

plt.figure(figsize=(14, 7))

plt.plot(
    df["time_seconds"],
    gyro[:, 0],
    label="Raw Gyro X",
    alpha=0.5
)

plt.plot(
    df["time_seconds"],
    gyro_clean[:, 0],
    label="Clean Gyro X",
    linewidth=1
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Raw vs Cleaned Gyroscope X")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 2
# ACCELERATION + GRAVITY
# ============================================================

print("Generating acceleration plot...")

plt.figure(figsize=(14, 7))

plt.plot(
    df["time_seconds"],
    acceleration[:, 0],
    label="Acceleration X",
    alpha=0.6
)

plt.plot(
    df["time_seconds"],
    acceleration[:, 1],
    label="Acceleration Y",
    alpha=0.6
)

plt.plot(
    df["time_seconds"],
    acceleration[:, 2],
    label="Acceleration Z",
    alpha=0.6
)

plt.plot(
    df["time_seconds"],
    gravity[:, 0],
    label="Gravity X",
    linewidth=1
)

plt.plot(
    df["time_seconds"],
    gravity[:, 1],
    label="Gravity Y",
    linewidth=1
)

plt.plot(
    df["time_seconds"],
    gravity[:, 2],
    label="Gravity Z",
    linewidth=1
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Accelerometer and Gravity")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 3
# LINEAR ACCELERATION
# ============================================================

print("Generating linear acceleration plot...")

plt.figure(figsize=(14, 7))

plt.plot(
    df["time_seconds"],
    linear_acceleration[:, 0],
    label="Linear Acc X"
)

plt.plot(
    df["time_seconds"],
    linear_acceleration[:, 1],
    label="Linear Acc Y"
)

plt.plot(
    df["time_seconds"],
    linear_acceleration[:, 2],
    label="Linear Acc Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Linear acceleration (m/s²)")
plt.title("Gravity-Compensated Linear Acceleration")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 4
# TIMESTAMP INTERVAL
# ============================================================

print("Generating timestamp interval plot...")

plt.figure(figsize=(14, 7))

plt.plot(
    df["time_seconds"],
    df["dt"],
    linewidth=0.8
)

plt.axhline(
    0.1,
    linestyle="--",
    label="Expected ≈ 0.1 s"
)

plt.axhline(
    0.2,
    linestyle="--",
    label="Large gap threshold"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Δt (seconds)")
plt.title("IMU Sampling Interval")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# OUTLIER INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZATION SUMMARY")
print("=" * 60)

print(
    f"Gyro outliers      : "
    f"{np.sum(gyro_outlier_mask)}"
)

print(
    f"Clean gyro maximum : "
    f"{np.max(gyro_clean_magnitude):.4f} rad/s"
)

print(
    f"Linear accel median: "
    f"{np.median(linear_acceleration_magnitude):.4f} m/s²"
)

print(
    f"Linear accel max   : "
    f"{np.max(linear_acceleration_magnitude):.4f} m/s²"
)

print(
    f"Median dt          : "
    f"{df['dt'].median():.4f} s"
)

print(
    f"Maximum dt         : "
    f"{df['dt'].max():.4f} s"
)

print("\nSTEP 4 COMPLETE")