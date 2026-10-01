import sys
from pathlib import Path

import matplotlib.pyplot as plt


# Allow importing data_loader.py
sys.path.append(str(Path(__file__).parent))

from data_loader import (
    load_io_vnbd,
    clean_column_names,
    prepare_data
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading IO-VNBD dataset...")

df = load_io_vnbd()

df = clean_column_names(df)
df = prepare_data(df)

print("Dataset ready for visualization.")


# ============================================================
# USE RELATIVE TIME
# ============================================================

df["time_seconds"] = (
    df["timestamp"] - df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# 1. ACCELEROMETER
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["time_seconds"],
    df["ACCELEROMETER_X_m_s²"],
    label="Accelerometer X"
)

plt.plot(
    df["time_seconds"],
    df["ACCELEROMETER_Y_m_s²"],
    label="Accelerometer Y"
)

plt.plot(
    df["time_seconds"],
    df["ACCELEROMETER_Z_m_s²"],
    label="Accelerometer Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("IO-VNBD Accelerometer Data")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 2. GYROSCOPE
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["time_seconds"],
    df["GYROSCOPE_Yaw_rad_s"],
    label="Gyroscope Yaw"
)

plt.plot(
    df["time_seconds"],
    df["GYROSCOPE_Pitch_rad_s"],
    label="Gyroscope Pitch"
)

plt.plot(
    df["time_seconds"],
    df["GYROSCOPE_Roll_rad_s"],
    label="Gyroscope Roll"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("IO-VNBD Gyroscope Data")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 3. GPS SPEED
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["time_seconds"],
    df["GPS_SPEED_Kmh"],
    label="GPS Speed"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Speed (km/h)")
plt.title("IO-VNBD GPS Speed")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 4. GPS ACCURACY
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["time_seconds"],
    df["GPS_ACCURACY_m"],
    label="GPS Accuracy"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Accuracy (m)")
plt.title("IO-VNBD GPS Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\nVisualization complete.")