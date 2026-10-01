import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).parent))

from data_loader import (
    load_io_vnbd,
    clean_column_names,
    prepare_data
)


# ============================================================
# QUATERNION-BASED IMU ORIENTATION
# ============================================================

print("=" * 60)
print("QUATERNION-BASED IMU ORIENTATION")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = load_io_vnbd()

df = clean_column_names(df)
df = prepare_data(df)


# ------------------------------------------------------------
# 2. Time
# ------------------------------------------------------------

df["time_seconds"] = (
    df["timestamp"] - df["timestamp"].iloc[0]
).dt.total_seconds()

time = df["time_seconds"].to_numpy()


# ------------------------------------------------------------
# 3. Accelerometer
# ------------------------------------------------------------

accel_columns = [
    "ACCELEROMETER_X_m_s²",
    "ACCELEROMETER_Y_m_s²",
    "ACCELEROMETER_Z_m_s²"
]

accel = df[
    accel_columns
].to_numpy(dtype=float)


# ------------------------------------------------------------
# 4. Magnetometer
# ------------------------------------------------------------

mag_columns = [
    "MAGNETIC_FIELD_X_Î¼T",
    "MAGNETIC_FIELD_Y_Î¼T",
    "MAGNETIC_FIELD_Z_Î¼T"
]

mag = df[
    mag_columns
].to_numpy(dtype=float)


# ------------------------------------------------------------
# 5. Clean gyroscope
#
# Same robust threshold used previously.
# ------------------------------------------------------------

gyro_columns = [
    "GYROSCOPE_Yaw_rad_s",
    "GYROSCOPE_Pitch_rad_s",
    "GYROSCOPE_Roll_rad_s"
]

gyro_raw = df[
    gyro_columns
].to_numpy(dtype=float)


gyro_magnitude = np.linalg.norm(
    gyro_raw,
    axis=1
)

median = np.median(gyro_magnitude)

mad = np.median(
    np.abs(gyro_magnitude - median)
)

robust_sigma = 1.4826 * mad

gyro_threshold = (
    median + 6 * robust_sigma
)

gyro_outlier = (
    gyro_magnitude > gyro_threshold
)


# ------------------------------------------------------------
# Interpolate gyro outliers
# ------------------------------------------------------------

gyro_clean = gyro_raw.copy()

for axis in range(3):

    series = pd.Series(
        gyro_raw[:, axis]
    )

    series[gyro_outlier] = np.nan

    series = series.interpolate(
        method="linear",
        limit_direction="both"
    )

    gyro_clean[:, axis] = (
        series.to_numpy()
    )


print("\nGyroscope preprocessing:")
print(
    f"Outlier threshold : "
    f"{gyro_threshold:.6f} rad/s"
)

print(
    f"Outlier samples   : "
    f"{gyro_outlier.sum()}"
)


# ------------------------------------------------------------
# 6. Gyroscope axis mapping
#
# Dataset:
#   Yaw   -> angular motion around Z
#   Pitch -> angular motion around Y
#   Roll  -> angular motion around X
#
# Therefore:
#   X = Roll
#   Y = Pitch
#   Z = Yaw
# ------------------------------------------------------------

gyro_xyz = np.column_stack([
    gyro_clean[:, 2],   # Roll -> X
    gyro_clean[:, 1],   # Pitch -> Y
    gyro_clean[:, 0]    # Yaw -> Z
])


# ------------------------------------------------------------
# 7. Initial orientation from accelerometer
# ------------------------------------------------------------

# Use the first 100 samples rather than one noisy sample.

initial_accel = np.median(
    accel[:100],
    axis=0
)

initial_mag = np.median(
    mag[:100],
    axis=0
)


ax, ay, az = initial_accel


# Roll
roll0 = np.arctan2(
    ay,
    az
)


# Pitch
pitch0 = np.arctan2(
    -ax,
    np.sqrt(
        ay**2 + az**2
    )
)


# ------------------------------------------------------------
# 8. Tilt-compensated magnetometer yaw
# ------------------------------------------------------------

mx, my, mz = initial_mag

# Tilt compensation

mx_level = (
    mx * np.cos(pitch0)
    + mz * np.sin(pitch0)
)

my_level = (
    mx * np.sin(roll0) * np.sin(pitch0)
    + my * np.cos(roll0)
    - mz * np.sin(roll0) * np.cos(pitch0)
)


yaw0 = np.arctan2(
    -my_level,
    mx_level
)


print("\nInitial orientation:")

print(
    f"Roll  : "
    f"{np.degrees(roll0):.2f}°"
)

print(
    f"Pitch : "
    f"{np.degrees(pitch0):.2f}°"
)

print(
    f"Yaw   : "
    f"{np.degrees(yaw0):.2f}°"
)


# ------------------------------------------------------------
# 9. Initial quaternion
# ------------------------------------------------------------

orientation = R.from_euler(
    "xyz",
    [
        roll0,
        pitch0,
        yaw0
    ]
)


# Store Euler angles

euler_angles = np.zeros(
    (len(df), 3)
)

euler_angles[0] = (
    orientation.as_euler(
        "xyz",
        degrees=True
    )
)


# ------------------------------------------------------------
# 10. Quaternion integration
# ------------------------------------------------------------

print("\nIntegrating gyroscope...")


for i in range(1, len(df)):

    dt = (
        time[i] - time[i - 1]
    )

    # Protect against invalid timestamps
    if dt <= 0 or dt > 0.2:

        euler_angles[i] = (
            euler_angles[i - 1]
        )

        orientation = orientation

        continue


    # Angular velocity
    omega = gyro_xyz[i]


    # Rotation during this timestep
    delta_rotation = R.from_rotvec(
        omega * dt
    )


    # Update orientation
    orientation = (
        orientation * delta_rotation
    )


    # Convert quaternion to Euler
    euler_angles[i] = (
        orientation.as_euler(
            "xyz",
            degrees=True
        )
    )


# ------------------------------------------------------------
# 11. Store results
# ------------------------------------------------------------

df["ESTIMATED_ROLL_deg"] = (
    euler_angles[:, 0]
)

df["ESTIMATED_PITCH_deg"] = (
    euler_angles[:, 1]
)

df["ESTIMATED_YAW_deg"] = (
    euler_angles[:, 2]
)


# ------------------------------------------------------------
# 12. Print statistics
# ------------------------------------------------------------

print("\nOrientation statistics:")

print(
    f"Roll range  : "
    f"{euler_angles[:, 0].min():.2f}° "
    f"to "
    f"{euler_angles[:, 0].max():.2f}°"
)

print(
    f"Pitch range : "
    f"{euler_angles[:, 1].min():.2f}° "
    f"to "
    f"{euler_angles[:, 1].max():.2f}°"
)

print(
    f"Yaw range   : "
    f"{euler_angles[:, 2].min():.2f}° "
    f"to "
    f"{euler_angles[:, 2].max():.2f}°"
)


# ------------------------------------------------------------
# 13. Plot orientation
# ------------------------------------------------------------

plt.figure(
    figsize=(15, 8)
)

plt.plot(
    time,
    euler_angles[:, 0],
    label="Estimated Roll"
)

plt.plot(
    time,
    euler_angles[:, 1],
    label="Estimated Pitch"
)

plt.plot(
    time,
    euler_angles[:, 2],
    label="Estimated Yaw"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Angle (degrees)"
)

plt.title(
    "IO-VNBD Quaternion IMU Orientation"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 14. Suspicious-event plot
# ------------------------------------------------------------

event_start = 4408
event_end = 4418

mask = (
    (time >= event_start)
    &
    (time <= event_end)
)


plt.figure(
    figsize=(15, 7)
)

plt.plot(
    time[mask],
    euler_angles[mask, 0],
    label="Roll"
)

plt.plot(
    time[mask],
    euler_angles[mask, 1],
    label="Pitch"
)

plt.plot(
    time[mask],
    euler_angles[mask, 2],
    label="Yaw"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Angle (degrees)"
)

plt.title(
    "Quaternion Orientation Around "
    "Suspicious IMU Event"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


print("\n" + "=" * 60)
print("QUATERNION ORIENTATION COMPLETE")
print("=" * 60)