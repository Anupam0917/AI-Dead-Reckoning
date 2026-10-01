import sys
from pathlib import Path

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
print("PHONE ORIENTATION ANALYSIS")
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
# ORIENTATION STATISTICS
# ============================================================

orientation_columns = [
    "ORIENTATION_Yaw_Â°",
    "ORIENTATION_Pitch_Â°",
    "ORIENTATION_Roll__Â°"
]

print("\nOrientation statistics:")

print(
    df[orientation_columns]
    .describe()
    .round(3)
)


# ============================================================
# PLOT ORIENTATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["time_seconds"],
    df["ORIENTATION_Yaw_Â°"],
    label="Yaw"
)

plt.plot(
    df["time_seconds"],
    df["ORIENTATION_Pitch_Â°"],
    label="Pitch"
)

plt.plot(
    df["time_seconds"],
    df["ORIENTATION_Roll__Â°"],
    label="Roll"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angle (degrees)")

plt.title("IO-VNBD Smartphone Orientation")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\nOrientation analysis complete.")