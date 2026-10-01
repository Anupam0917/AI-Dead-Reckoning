import sys
from pathlib import Path

import numpy as np

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
print("SENSOR QUALITY ANALYSIS")
print("=" * 60)

df = load_io_vnbd()

df = clean_column_names(df)
df = prepare_data(df)


# ============================================================
# SENSOR MAGNITUDES
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


df["acceleration_magnitude"] = np.sqrt(
    df["ACCELEROMETER_X_m_s²"] ** 2 +
    df["ACCELEROMETER_Y_m_s²"] ** 2 +
    df["ACCELEROMETER_Z_m_s²"] ** 2
)


df["gyro_magnitude"] = np.sqrt(
    df["GYROSCOPE_Yaw_rad_s"] ** 2 +
    df["GYROSCOPE_Pitch_rad_s"] ** 2 +
    df["GYROSCOPE_Roll_rad_s"] ** 2
)


# ============================================================
# QUANTILES
# ============================================================

print("\n" + "=" * 60)
print("ACCELERATION MAGNITUDE DISTRIBUTION")
print("=" * 60)

print(
    df["acceleration_magnitude"]
    .quantile([
        0.50,
        0.90,
        0.95,
        0.99,
        0.995,
        0.999,
        1.00
    ])
)


print("\n" + "=" * 60)
print("GYROSCOPE MAGNITUDE DISTRIBUTION")
print("=" * 60)

print(
    df["gyro_magnitude"]
    .quantile([
        0.50,
        0.90,
        0.95,
        0.99,
        0.995,
        0.999,
        1.00
    ])
)


# ============================================================
# EXTREME EVENTS
# ============================================================

print("\n" + "=" * 60)
print("EXTREME ACCELERATION EVENTS")
print("=" * 60)

print(
    df.nlargest(
        10,
        "acceleration_magnitude"
    )[
        [
            "acceleration_magnitude",
            "gyro_magnitude",
            "GPS_SPEED_Kmh",
            "GPS_ACCURACY_m"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 60)
print("EXTREME GYROSCOPE EVENTS")
print("=" * 60)

print(
    df.nlargest(
        10,
        "gyro_magnitude"
    )[
        [
            "acceleration_magnitude",
            "gyro_magnitude",
            "GPS_SPEED_Kmh",
            "GPS_ACCURACY_m"
        ]
    ].to_string(index=False)
)


# ============================================================
# TIME GAPS
# ============================================================

df["dt"] = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
)


print("\n" + "=" * 60)
print("TIME GAP DISTRIBUTION")
print("=" * 60)

print(
    df["dt"]
    .quantile([
        0.50,
        0.90,
        0.95,
        0.99,
        1.00
    ])
)


print("\n" + "=" * 60)
print("GAPS GREATER THAN 0.2 SECONDS")
print("=" * 60)

large_gaps = df[df["dt"] > 0.2]

print(
    large_gaps[
        [
            "timestamp",
            "dt"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 60)
print("SENSOR QUALITY ANALYSIS COMPLETE")
print("=" * 60)