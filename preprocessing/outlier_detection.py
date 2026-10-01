import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from data_loader import (
    load_io_vnbd,
    clean_column_names,
    prepare_data
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("IMU OUTLIER DETECTION")
print("=" * 60)

df = load_io_vnbd()

df = clean_column_names(df)
df = prepare_data(df)


# ============================================================
# TIME
# ============================================================

df["time_seconds"] = (
    df["timestamp"] - df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# SENSOR COLUMNS
# ============================================================

acc_columns = [
    "ACCELEROMETER_X_m_s²",
    "ACCELEROMETER_Y_m_s²",
    "ACCELEROMETER_Z_m_s²"
]

gyro_columns = [
    "GYROSCOPE_Yaw_rad_s",
    "GYROSCOPE_Pitch_rad_s",
    "GYROSCOPE_Roll_rad_s"
]


# ============================================================
# CALCULATE SENSOR MAGNITUDES
# ============================================================

df["acceleration_magnitude"] = np.sqrt(
    df[acc_columns[0]] ** 2 +
    df[acc_columns[1]] ** 2 +
    df[acc_columns[2]] ** 2
)

df["gyro_magnitude"] = np.sqrt(
    df[gyro_columns[0]] ** 2 +
    df[gyro_columns[1]] ** 2 +
    df[gyro_columns[2]] ** 2
)


# ============================================================
# ROBUST OUTLIER DETECTION
# ============================================================

def robust_threshold(series, multiplier=6):

    median = series.median()

    mad = np.median(
        np.abs(series - median)
    )

    if mad == 0:
        return median, np.inf

    threshold = median + multiplier * 1.4826 * mad

    return median, threshold


# ------------------------------------------------------------
# Acceleration threshold
# ------------------------------------------------------------

acc_median, acc_threshold = robust_threshold(
    df["acceleration_magnitude"]
)


# ------------------------------------------------------------
# Gyroscope threshold
# ------------------------------------------------------------

gyro_median, gyro_threshold = robust_threshold(
    df["gyro_magnitude"]
)


# ============================================================
# CREATE FLAGS
# ============================================================

df["acceleration_outlier"] = (
    df["acceleration_magnitude"] > acc_threshold
)

df["gyro_outlier"] = (
    df["gyro_magnitude"] > gyro_threshold
)

df["imu_outlier"] = (
    df["acceleration_outlier"] |
    df["gyro_outlier"]
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLDS")
print("=" * 60)

print(
    f"Acceleration median : {acc_median:.4f} m/s²"
)

print(
    f"Acceleration threshold : {acc_threshold:.4f} m/s²"
)

print(
    f"Gyroscope median : {gyro_median:.4f} rad/s"
)

print(
    f"Gyroscope threshold : {gyro_threshold:.4f} rad/s"
)


# ============================================================
# OUTLIER COUNT
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER COUNTS")
print("=" * 60)

print(
    f"Acceleration outliers : "
    f"{df['acceleration_outlier'].sum()}"
)

print(
    f"Gyroscope outliers    : "
    f"{df['gyro_outlier'].sum()}"
)

print(
    f"Total IMU outliers    : "
    f"{df['imu_outlier'].sum()}"
)


# ============================================================
# TOP SUSPICIOUS EVENTS
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 SUSPICIOUS IMU EVENTS")
print("=" * 60)

events = df[df["imu_outlier"]].copy()

events = events.sort_values(
    "acceleration_magnitude",
    ascending=False
)

columns_to_show = [
    "time_seconds",
    "acceleration_magnitude",
    "gyro_magnitude",
    "GPS_SPEED_Kmh",
    "GPS_ACCURACY_m",
    "ACCELEROMETER_X_m_s²",
    "ACCELEROMETER_Y_m_s²",
    "ACCELEROMETER_Z_m_s²"
]

print(
    events[columns_to_show]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# TIME GAPS
# ============================================================

print("\n" + "=" * 60)
print("LARGE TIME GAPS")
print("=" * 60)

df["dt"] = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
)

large_gaps = df[df["dt"] > 0.2]

print(
    f"Number of gaps > 0.2 seconds: "
    f"{len(large_gaps)}"
)

if len(large_gaps) > 0:

    print(
        large_gaps[
            ["time_seconds", "dt"]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER DETECTION COMPLETE")
print("=" * 60)