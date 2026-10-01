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

df = load_io_vnbd()

df = clean_column_names(df)
df = prepare_data(df)

df["time_seconds"] = (
    df["timestamp"] - df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# SELECT SUSPICIOUS EVENT
# ============================================================

EVENT_TIME = 4400

WINDOW_BEFORE = 30
WINDOW_AFTER = 30

start_time = EVENT_TIME - WINDOW_BEFORE
end_time = EVENT_TIME + WINDOW_AFTER

event_df = df[
    (df["time_seconds"] >= start_time) &
    (df["time_seconds"] <= end_time)
].copy()


print("=" * 60)
print("IMU EVENT ANALYSIS")
print("=" * 60)

print(f"Event time : {EVENT_TIME} seconds")
print(f"Window     : {start_time} to {end_time} seconds")
print(f"Samples    : {len(event_df)}")


# ============================================================
# EVENT STATISTICS
# ============================================================

print("\nMaximum absolute sensor values in window:")

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

for column in acc_columns + gyro_columns:

    value = event_df[column].abs().max()

    print(f"{column}: {value:.4f}")


# ============================================================
# 1. ACCELEROMETER EVENT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    event_df["time_seconds"],
    event_df["ACCELEROMETER_X_m_s²"],
    label="Accelerometer X"
)

plt.plot(
    event_df["time_seconds"],
    event_df["ACCELEROMETER_Y_m_s²"],
    label="Accelerometer Y"
)

plt.plot(
    event_df["time_seconds"],
    event_df["ACCELEROMETER_Z_m_s²"],
    label="Accelerometer Z"
)

plt.axvline(
    EVENT_TIME,
    linestyle="--",
    label="Suspicious event"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Accelerometer Around Suspicious Event")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 2. GYROSCOPE EVENT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    event_df["time_seconds"],
    event_df["GYROSCOPE_Yaw_rad_s"],
    label="Gyroscope Yaw"
)

plt.plot(
    event_df["time_seconds"],
    event_df["GYROSCOPE_Pitch_rad_s"],
    label="Gyroscope Pitch"
)

plt.plot(
    event_df["time_seconds"],
    event_df["GYROSCOPE_Roll_rad_s"],
    label="Gyroscope Roll"
)

plt.axvline(
    EVENT_TIME,
    linestyle="--",
    label="Suspicious event"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Gyroscope Around Suspicious Event")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 3. GPS SPEED
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    event_df["time_seconds"],
    event_df["GPS_SPEED_Kmh"],
    label="GPS Speed"
)

plt.axvline(
    EVENT_TIME,
    linestyle="--",
    label="Suspicious event"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Speed (km/h)")
plt.title("GPS Speed Around Suspicious Event")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\nEvent analysis complete.")