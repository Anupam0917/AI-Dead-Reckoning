import sys
from pathlib import Path

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

print("=" * 60)
print("IO-VNBD DATA ANALYSIS")
print("=" * 60)

df = load_io_vnbd()

df = clean_column_names(df)
df = prepare_data(df)


# ============================================================
# 1. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("1. BASIC INFORMATION")
print("=" * 60)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\nColumn names:")

for column in df.columns:
    print(" -", column)


# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)


# ============================================================
# 3. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)
print("3. DUPLICATE ROWS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")


# ============================================================
# 4. SAMPLING INTERVAL
# ============================================================

print("\n" + "=" * 60)
print("4. SAMPLING INTERVAL")
print("=" * 60)

time_diff = df["timestamp"].diff().dt.total_seconds()

print(f"Mean interval   : {time_diff.mean():.4f} seconds")
print(f"Median interval : {time_diff.median():.4f} seconds")
print(f"Min interval    : {time_diff.min():.4f} seconds")
print(f"Max interval    : {time_diff.max():.4f} seconds")


# ============================================================
# 5. SENSOR STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("5. SENSOR STATISTICS")
print("=" * 60)

sensor_columns = [
    "ACCELEROMETER_X_m_s²",
    "ACCELEROMETER_Y_m_s²",
    "ACCELEROMETER_Z_m_s²",

    "GYROSCOPE_Yaw_rad_s",
    "GYROSCOPE_Pitch_rad_s",
    "GYROSCOPE_Roll_rad_s",

    "MAGNETIC_FIELD_X_Î¼T",
    "MAGNETIC_FIELD_Y_Î¼T",
    "MAGNETIC_FIELD_Z_Î¼T"
]

print(df[sensor_columns].describe().round(4))


# ============================================================
# 6. GNSS STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("6. GNSS STATISTICS")
print("=" * 60)

gps_columns = [
    "GPS_LATITUDE_degrees",
    "GPS_LONGITUDE_degrees",
    "GPS_ALTITUDE_m",
    "GPS_SPEED_Kmh",
    "GPS_ACCURACY_m",
    "GPS_ORIENTATION_Â°"
]

print(df[gps_columns].describe().round(4))


# ============================================================
# 7. SPEED INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("7. SPEED INFORMATION")
print("=" * 60)

speed = df["GPS_SPEED_Kmh"]

print(f"Minimum speed : {speed.min():.2f} km/h")
print(f"Maximum speed : {speed.max():.2f} km/h")
print(f"Mean speed    : {speed.mean():.2f} km/h")
print(f"Median speed  : {speed.median():.2f} km/h")


# ============================================================
# 8. GPS ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("8. GPS ACCURACY")
print("=" * 60)

accuracy = df["GPS_ACCURACY_m"]

print(f"Minimum accuracy : {accuracy.min():.2f} m")
print(f"Maximum accuracy : {accuracy.max():.2f} m")
print(f"Mean accuracy    : {accuracy.mean():.2f} m")
print(f"Median accuracy  : {accuracy.median():.2f} m")


# ============================================================
# 9. TIME RANGE
# ============================================================

print("\n" + "=" * 60)
print("9. DATASET TIME RANGE")
print("=" * 60)

start_time = df["timestamp"].iloc[0]
end_time = df["timestamp"].iloc[-1]

duration = end_time - start_time

print(f"Start    : {start_time}")
print(f"End      : {end_time}")
print(f"Duration : {duration}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("DATA ANALYSIS COMPLETE")
print("=" * 60)