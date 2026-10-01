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
print("MAGNETOMETER ANALYSIS")
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
# MAGNETOMETER COLUMNS
# ============================================================

mag_x = "MAGNETIC_FIELD_X_Î¼T"
mag_y = "MAGNETIC_FIELD_Y_Î¼T"
mag_z = "MAGNETIC_FIELD_Z_Î¼T"


# ============================================================
# MAGNETIC FIELD MAGNITUDE
# ============================================================

df["magnetic_magnitude"] = np.sqrt(
    df[mag_x] ** 2 +
    df[mag_y] ** 2 +
    df[mag_z] ** 2
)


# ============================================================
# STATISTICS
# ============================================================

print("\nMagnetometer statistics:")

print(
    df[
        [mag_x, mag_y, mag_z]
    ]
    .describe()
    .round(4)
)


print("\nMagnetic field magnitude:")

print(
    df["magnetic_magnitude"]
    .describe()
    .round(4)
)


# ============================================================
# PLOT COMPONENTS
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["time_seconds"],
    df[mag_x],
    label="Magnetic X"
)

plt.plot(
    df["time_seconds"],
    df[mag_y],
    label="Magnetic Y"
)

plt.plot(
    df["time_seconds"],
    df[mag_z],
    label="Magnetic Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Magnetic field (µT)")

plt.title("IO-VNBD Magnetometer Data")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# MAGNITUDE
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["time_seconds"],
    df["magnetic_magnitude"],
    label="Magnetic Field Magnitude"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Magnitude (µT)")

plt.title("IO-VNBD Magnetic Field Magnitude")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\nMagnetometer analysis complete.")