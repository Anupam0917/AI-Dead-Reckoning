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
print("GRAVITY / ORIENTATION CONSISTENCY CHECK")
print("=" * 60)

df = load_io_vnbd()

df = clean_column_names(df)
df = prepare_data(df)


# ============================================================
# TIME
# ============================================================

df["time_seconds"] = (
    df["timestamp"] -
    df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# GRAVITY COMPONENTS
# ============================================================

gravity_columns = [
    "GRAVITY_X_m_s²",
    "GRAVITY_Y_m_s²",
    "GRAVITY_Z_m_s²"
]

print("\nGravity statistics:")

print(
    df[gravity_columns]
    .describe()
    .round(4)
)


# ============================================================
# GRAVITY MAGNITUDE
# ============================================================

df["gravity_magnitude"] = np.sqrt(
    df["GRAVITY_X_m_s²"] ** 2 +
    df["GRAVITY_Y_m_s²"] ** 2 +
    df["GRAVITY_Z_m_s²"] ** 2
)


print("\nGravity magnitude statistics:")

print(
    df["gravity_magnitude"]
    .describe()
    .round(6)
)


# ============================================================
# ORIENTATION
# ============================================================

orientation_columns = [
    "ORIENTATION_Yaw_Â°",
    "ORIENTATION_Pitch_Â°",
    "ORIENTATION_Roll__Â°"
]


# ============================================================
# PLOT GRAVITY VECTOR
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["time_seconds"],
    df["GRAVITY_X_m_s²"],
    label="Gravity X"
)

plt.plot(
    df["time_seconds"],
    df["GRAVITY_Y_m_s²"],
    label="Gravity Y"
)

plt.plot(
    df["time_seconds"],
    df["GRAVITY_Z_m_s²"],
    label="Gravity Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Gravity (m/s²)")

plt.title("Gravity Vector Components")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# PITCH / ROLL VS GRAVITY
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["time_seconds"],
    df["GRAVITY_X_m_s²"],
    label="Gravity X"
)

plt.plot(
    df["time_seconds"],
    df["GRAVITY_Y_m_s²"],
    label="Gravity Y"
)

plt.plot(
    df["time_seconds"],
    df["GRAVITY_Z_m_s²"],
    label="Gravity Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Gravity component (m/s²)")

plt.title("Gravity Components Used for Orientation")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# CHECK CORRELATION
# ============================================================

print("\nCorrelation between gravity components and orientation:")

for gravity_column in gravity_columns:

    print(f"\n{gravity_column}")

    print(
        df[
            [gravity_column] + orientation_columns
        ]
        .corr()
        .loc[
            gravity_column,
            orientation_columns
        ]
        .round(4)
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)