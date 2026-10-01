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
# ORIENTATION V2 - STEP 1
# SENSOR AXIS INVESTIGATION
# ============================================================

print("=" * 60)
print("ORIENTATION V2 - SENSOR AXIS INVESTIGATION")
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
# Gyroscope
# ------------------------------------------------------------

gyro_columns = [
    "GYROSCOPE_Yaw_rad_s",
    "GYROSCOPE_Pitch_rad_s",
    "GYROSCOPE_Roll_rad_s"
]

print("\nGyroscope columns:")
for column in gyro_columns:
    print(" -", column)


print("\nGyroscope statistics:")

print(
    df[gyro_columns]
    .describe()
    .round(4)
)


# ------------------------------------------------------------
# Accelerometer
# ------------------------------------------------------------

acc_columns = [
    "ACCELEROMETER_X_m_s²",
    "ACCELEROMETER_Y_m_s²",
    "ACCELEROMETER_Z_m_s²"
]

print("\nAccelerometer statistics:")

print(
    df[acc_columns]
    .describe()
    .round(4)
)


# ------------------------------------------------------------
# Plot gyroscope channels
# ------------------------------------------------------------

time = df["time_seconds"].to_numpy()

plt.figure(figsize=(14, 7))

plt.plot(
    time,
    df["GYROSCOPE_Yaw_rad_s"],
    label="Gyroscope Yaw"
)

plt.plot(
    time,
    df["GYROSCOPE_Pitch_rad_s"],
    label="Gyroscope Pitch"
)

plt.plot(
    time,
    df["GYROSCOPE_Roll_rad_s"],
    label="Gyroscope Roll"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")

plt.title(
    "Orientation V2 - Gyroscope Channels"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# Calculate gyroscope magnitude
# ------------------------------------------------------------

gyro = df[gyro_columns].to_numpy()

gyro_magnitude = np.linalg.norm(
    gyro,
    axis=1
)


print("\nGyroscope magnitude:")

print(
    pd.Series(gyro_magnitude)
    .describe()
    .round(4)
)


# ------------------------------------------------------------
# Find strongest gyroscope events
# ------------------------------------------------------------

top_indices = np.argsort(
    gyro_magnitude
)[-10:][::-1]


print("\nTop 10 gyroscope events:")

print(
    df.iloc[
        top_indices
    ][
        [
            "time_seconds",
            "GYROSCOPE_Yaw_rad_s",
            "GYROSCOPE_Pitch_rad_s",
            "GYROSCOPE_Roll_rad_s",
            "GPS_SPEED_Kmh"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)