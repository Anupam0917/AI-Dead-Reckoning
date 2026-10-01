import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from data_loader import (
    load_io_vnbd,
    clean_column_names,
    prepare_data
)


# ============================================================
# ROBUST GYROSCOPE PREPROCESSING
# ============================================================

print("=" * 60)
print("ROBUST GYROSCOPE PREPROCESSING")
print("=" * 60)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = load_io_vnbd()

df = clean_column_names(df)
df = prepare_data(df)


# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

df["time_seconds"] = (
    df["timestamp"] - df["timestamp"].iloc[0]
).dt.total_seconds()


# ------------------------------------------------------------
# Gyroscope channels
# ------------------------------------------------------------

gyro_columns = [
    "GYROSCOPE_Yaw_rad_s",
    "GYROSCOPE_Pitch_rad_s",
    "GYROSCOPE_Roll_rad_s"
]

gyro = df[gyro_columns].to_numpy(dtype=float)


# ------------------------------------------------------------
# Gyroscope magnitude
# ------------------------------------------------------------

gyro_magnitude = np.linalg.norm(
    gyro,
    axis=1
)

df["gyro_magnitude"] = gyro_magnitude


# ------------------------------------------------------------
# Robust threshold using Median Absolute Deviation
# ------------------------------------------------------------

median = np.median(gyro_magnitude)

mad = np.median(
    np.abs(gyro_magnitude - median)
)

robust_sigma = 1.4826 * mad

threshold = median + 6 * robust_sigma


print("\nGyroscope quality:")
print(f"Median magnitude       : {median:.6f} rad/s")
print(f"MAD                    : {mad:.6f}")
print(f"Robust sigma           : {robust_sigma:.6f}")
print(f"Outlier threshold      : {threshold:.6f} rad/s")


# ------------------------------------------------------------
# Detect outliers
# ------------------------------------------------------------

outlier_mask = (
    gyro_magnitude > threshold
)

df["gyro_outlier"] = outlier_mask


outlier_count = outlier_mask.sum()

print(f"Gyroscope outliers     : {outlier_count}")
print(
    f"Clean samples          : "
    f"{len(df) - outlier_count}"
)


# ------------------------------------------------------------
# Replace outliers using interpolation
# ------------------------------------------------------------

gyro_clean = gyro.copy()

for axis in range(3):

    series = pd.Series(
        gyro[:, axis]
    )

    # Mark suspicious samples as missing
    series[outlier_mask] = np.nan

    # Interpolate using surrounding valid samples
    series = (
        series
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )

    gyro_clean[:, axis] = series.to_numpy()


# ------------------------------------------------------------
# Store cleaned gyroscope
# ------------------------------------------------------------

for i, column in enumerate(gyro_columns):

    df[
        column + "_CLEAN"
    ] = gyro_clean[:, i]


# ------------------------------------------------------------
# Statistics after cleaning
# ------------------------------------------------------------

clean_magnitude = np.linalg.norm(
    gyro_clean,
    axis=1
)

print("\nAfter outlier removal:")

print(
    pd.Series(clean_magnitude)
    .describe()
    .round(6)
)


# ------------------------------------------------------------
# Check suspicious event
# ------------------------------------------------------------

event_start = 4408
event_end = 4418

event = df[
    (df["time_seconds"] >= event_start) &
    (df["time_seconds"] <= event_end)
]


print("\nGyroscope around suspicious event:")

print(
    event[
        [
            "time_seconds",
            "GYROSCOPE_Yaw_rad_s",
            "GYROSCOPE_Pitch_rad_s",
            "GYROSCOPE_Roll_rad_s",
            "GYROSCOPE_Yaw_rad_s_CLEAN",
            "GYROSCOPE_Pitch_rad_s_CLEAN",
            "GYROSCOPE_Roll_rad_s_CLEAN"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# Plot raw vs cleaned gyro magnitude
# ------------------------------------------------------------

plt.figure(figsize=(14, 7))

plt.plot(
    df["time_seconds"],
    gyro_magnitude,
    label="Raw gyro magnitude"
)

plt.plot(
    df["time_seconds"],
    clean_magnitude,
    label="Cleaned gyro magnitude"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Outlier threshold"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity magnitude (rad/s)")

plt.title(
    "Raw vs Cleaned Gyroscope"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


print("\n" + "=" * 60)
print("GYROSCOPE PREPROCESSING COMPLETE")
print("=" * 60)