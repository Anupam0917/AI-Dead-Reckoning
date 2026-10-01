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
# LOAD DATA
# ============================================================

print("=" * 60)
print("GRAVITY COMPENSATION BASELINE")
print("=" * 60)

df = load_io_vnbd()

df = clean_column_names(df)
df = prepare_data(df)


# ============================================================
# RAW ACCELERATION
# ============================================================

acc_x = df["ACCELEROMETER_X_m_s²"]
acc_y = df["ACCELEROMETER_Y_m_s²"]
acc_z = df["ACCELEROMETER_Z_m_s²"]


# ============================================================
# GRAVITY VECTOR
# ============================================================

gravity_x = df["GRAVITY_X_m_s²"]
gravity_y = df["GRAVITY_Y_m_s²"]
gravity_z = df["GRAVITY_Z_m_s²"]


# ============================================================
# REMOVE GRAVITY
# ============================================================

df["LINEAR_ACCEL_X_m_s²"] = (
    acc_x - gravity_x
)

df["LINEAR_ACCEL_Y_m_s²"] = (
    acc_y - gravity_y
)

df["LINEAR_ACCEL_Z_m_s²"] = (
    acc_z - gravity_z
)


# ============================================================
# MAGNITUDES
# ============================================================

df["RAW_ACCEL_MAGNITUDE"] = np.sqrt(
    acc_x**2 +
    acc_y**2 +
    acc_z**2
)

df["GRAVITY_MAGNITUDE"] = np.sqrt(
    gravity_x**2 +
    gravity_y**2 +
    gravity_z**2
)

df["LINEAR_ACCEL_MAGNITUDE"] = np.sqrt(
    df["LINEAR_ACCEL_X_m_s²"]**2 +
    df["LINEAR_ACCEL_Y_m_s²"]**2 +
    df["LINEAR_ACCEL_Z_m_s²"]**2
)


# ============================================================
# STATISTICS
# ============================================================

print("\nRaw acceleration magnitude:")
print(
    df["RAW_ACCEL_MAGNITUDE"]
    .describe()
    .round(4)
)

print("\nGravity magnitude:")
print(
    df["GRAVITY_MAGNITUDE"]
    .describe()
    .round(4)
)

print("\nLinear acceleration magnitude:")
print(
    df["LINEAR_ACCEL_MAGNITUDE"]
    .describe()
    .round(4)
)


# ============================================================
# FIRST 10 SAMPLES
# ============================================================

print("\nFirst 10 linear acceleration samples:")

print(
    df[
        [
            "LINEAR_ACCEL_X_m_s²",
            "LINEAR_ACCEL_Y_m_s²",
            "LINEAR_ACCEL_Z_m_s²"
        ]
    ]
    .head(10)
)


# ============================================================
# PLOT
# ============================================================

df["time_seconds"] = (
    df["timestamp"] -
    df["timestamp"].iloc[0]
).dt.total_seconds()


plt.figure(figsize=(12, 6))

plt.plot(
    df["time_seconds"],
    df["LINEAR_ACCEL_X_m_s²"],
    label="Linear Acceleration X"
)

plt.plot(
    df["time_seconds"],
    df["LINEAR_ACCEL_Y_m_s²"],
    label="Linear Acceleration Y"
)

plt.plot(
    df["time_seconds"],
    df["LINEAR_ACCEL_Z_m_s²"],
    label="Linear Acceleration Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Linear acceleration (m/s²)")

plt.title(
    "Gravity-Compensated Linear Acceleration"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\nGravity compensation complete.")