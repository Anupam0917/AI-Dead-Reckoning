import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from data_loader import (
    load_io_vnbd,
    clean_column_names,
    prepare_data
)


# ============================================================
# CONFIGURATION
# ============================================================

ALPHA = 0.98

# Expected magnetic field magnitude range.
# We use a broad range initially because the dataset
# contains substantial magnetic disturbances.
MAG_MIN = 30.0
MAG_MAX = 60.0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("BASELINE IMU ORIENTATION ESTIMATOR")
print("=" * 60)

df = load_io_vnbd()

df = clean_column_names(df)
df = prepare_data(df)


# ============================================================
# TIME
# ============================================================

df["time_seconds"] = (
    df["timestamp"]
    - df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# SENSOR COLUMNS
# ============================================================

acc_x = "ACCELEROMETER_X_m_s²"
acc_y = "ACCELEROMETER_Y_m_s²"
acc_z = "ACCELEROMETER_Z_m_s²"

gyro_x = "GYROSCOPE_Yaw_rad_s"
gyro_y = "GYROSCOPE_Pitch_rad_s"
gyro_z = "GYROSCOPE_Roll_rad_s"

mag_x = "MAGNETIC_FIELD_X_Î¼T"
mag_y = "MAGNETIC_FIELD_Y_Î¼T"
mag_z = "MAGNETIC_FIELD_Z_Î¼T"


# ============================================================
# CONVERT TO NUMPY
# ============================================================

acc = df[[acc_x, acc_y, acc_z]].to_numpy()

gyro = df[[gyro_x, gyro_y, gyro_z]].to_numpy()

mag = df[[mag_x, mag_y, mag_z]].to_numpy()

time = df["time_seconds"].to_numpy()


# ============================================================
# MAGNETIC FIELD MAGNITUDE
# ============================================================

mag_magnitude = np.linalg.norm(mag, axis=1)


# ============================================================
# MAGNETOMETER CONFIDENCE
# ============================================================

mag_valid = (
    (mag_magnitude >= MAG_MIN) &
    (mag_magnitude <= MAG_MAX)
)

mag_confidence = mag_valid.astype(float)


# ============================================================
# INITIAL ORIENTATION FROM ACCELEROMETER
# ============================================================

ax, ay, az = acc[0]

roll = np.arctan2(
    ay,
    az
)

pitch = np.arctan2(
    -ax,
    np.sqrt(ay ** 2 + az ** 2)
)


# ============================================================
# INITIAL YAW FROM MAGNETOMETER
# ============================================================

mx, my, mz = mag[0]

yaw = np.arctan2(
    my,
    mx
)


# ============================================================
# STORAGE
# ============================================================

roll_history = np.zeros(len(df))
pitch_history = np.zeros(len(df))
yaw_history = np.zeros(len(df))


roll_history[0] = roll
pitch_history[0] = pitch
yaw_history[0] = yaw


# ============================================================
# ORIENTATION ESTIMATION
# ============================================================

for i in range(1, len(df)):

    # --------------------------------------------------------
    # Time difference
    # --------------------------------------------------------

    dt = time[i] - time[i - 1]

    # Ignore invalid / unusually large gaps
    if dt <= 0 or dt > 0.2:
        dt = 0.1


    # --------------------------------------------------------
    # Gyroscope integration
    # --------------------------------------------------------

    gx, gy, gz = gyro[i]

    gyro_roll = roll + gx * dt
    gyro_pitch = pitch + gy * dt
    gyro_yaw = yaw + gz * dt


    # --------------------------------------------------------
    # Accelerometer tilt estimate
    # --------------------------------------------------------

    ax, ay, az = acc[i]

    accel_roll = np.arctan2(
        ay,
        az
    )

    accel_pitch = np.arctan2(
        -ax,
        np.sqrt(ay ** 2 + az ** 2)
    )


    # --------------------------------------------------------
    # Complementary filter
    # --------------------------------------------------------

    roll = (
        ALPHA * gyro_roll +
        (1 - ALPHA) * accel_roll
    )

    pitch = (
        ALPHA * gyro_pitch +
        (1 - ALPHA) * accel_pitch
    )


    # --------------------------------------------------------
    # Magnetometer yaw correction
    # --------------------------------------------------------

    if mag_valid[i]:

        mx, my, mz = mag[i]

        mag_yaw = np.arctan2(
            my,
            mx
        )

        # Circular angle difference
        yaw_error = np.arctan2(
            np.sin(mag_yaw - gyro_yaw),
            np.cos(mag_yaw - gyro_yaw)
        )

        # Small correction
        yaw = gyro_yaw + 0.02 * yaw_error

    else:

        # Magnetometer unreliable
        yaw = gyro_yaw


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    roll_history[i] = roll
    pitch_history[i] = pitch
    yaw_history[i] = yaw


# ============================================================
# CONVERT TO DEGREES
# ============================================================

roll_deg = np.degrees(roll_history)

pitch_deg = np.degrees(pitch_history)

yaw_deg = np.degrees(yaw_history)


# ============================================================
# WRAP ANGLES
# ============================================================

yaw_deg = (
    (yaw_deg + 180) % 360
) - 180


# ============================================================
# RESULTS
# ============================================================

print("\nMagnetometer quality:")

print(
    f"Reliable samples : "
    f"{mag_valid.sum()} / {len(mag_valid)}"
)

print(
    f"Reliable percentage : "
    f"{100 * mag_valid.mean():.2f}%"
)


print("\nBaseline orientation statistics:")

print(
    f"Roll  range  : "
    f"{roll_deg.min():.2f}° to {roll_deg.max():.2f}°"
)

print(
    f"Pitch range  : "
    f"{pitch_deg.min():.2f}° to {pitch_deg.max():.2f}°"
)

print(
    f"Yaw range    : "
    f"{yaw_deg.min():.2f}° to {yaw_deg.max():.2f}°"
)


# ============================================================
# PLOT ORIENTATION
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    time,
    roll_deg,
    label="Estimated Roll"
)

plt.plot(
    time,
    pitch_deg,
    label="Estimated Pitch"
)

plt.plot(
    time,
    yaw_deg,
    label="Estimated Yaw"
)

plt.xlabel("Time (seconds)")

plt.ylabel("Angle (degrees)")

plt.title(
    "IO-VNBD Baseline IMU Orientation"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# MAGNETOMETER CONFIDENCE
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    time,
    mag_magnitude,
    label="Magnetic Field Magnitude"
)

plt.axhline(
    MAG_MIN,
    linestyle="--",
    label="Lower threshold"
)

plt.axhline(
    MAG_MAX,
    linestyle="--",
    label="Upper threshold"
)

plt.xlabel("Time (seconds)")

plt.ylabel("Magnetic Field (µT)")

plt.title(
    "Magnetometer Reliability Check"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


print("\nOrientation baseline complete.")