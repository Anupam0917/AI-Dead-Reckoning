import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SIH 2026 - AI/ML Intelligent Dead Reckoning
# Prototype V2
#
# AI Speed + Smartphone Heading
#        ↓
# Dead Reckoning
#        ↓
# GNSS outage simulation
#        ↓
# Position error evaluation
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Synchronised V abd S datasets",
    "Categorised IOVNB Dataset",
    "M (Driver B)",
    "S-M.csv"
)

AI_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "prototype_ai_speed_results.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

OUTAGE_DURATION = 60.0

# Same outage region used in Prototype V1
OUTAGE_START = 7420.42
OUTAGE_END = OUTAGE_START + OUTAGE_DURATION

EARTH_RADIUS = 6371000.0


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("SIH PROTOTYPE V2 - DEAD RECKONING")
print("=" * 70)

print("\nLoading raw dataset...")

df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

print(f"Raw rows: {len(df)}")


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_", regex=False)
)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

def find_column(keyword):
    matches = [
        c for c in df.columns
        if keyword.lower() in c.lower()
    ]

    if not matches:
        raise ValueError(
            f"Could not find column containing: {keyword}"
        )

    return matches[0]


LAT_COL = find_column("GPS_LATITUDE")
LON_COL = find_column("GPS_LONGITUDE")
GPS_SPEED_COL = find_column("GPS_SPEED")
GPS_HEADING_COL = find_column("GPS_ORIENTATION")

PHONE_HEADING_COL = find_column("ORIENTATION_Yaw")

TIME_COL = find_column("DATE")


print("\nDetected columns:")
print("Latitude :", LAT_COL)
print("Longitude:", LON_COL)
print("GPS speed:", GPS_SPEED_COL)
print("GPS heading:", GPS_HEADING_COL)
print("Phone yaw:", PHONE_HEADING_COL)
print("Timestamp:", TIME_COL)


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df[TIME_COL],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)

df = df.dropna(subset=["timestamp"]).copy()

df["time_seconds"] = (
    df["timestamp"] - df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in [
    LAT_COL,
    LON_COL,
    GPS_SPEED_COL,
    GPS_HEADING_COL,
    PHONE_HEADING_COL
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# LOAD AI SPEED RESULTS
# ============================================================

print("\nLoading AI speed predictions...")

ai_df = pd.read_csv(AI_FILE)

print("AI result columns:")
print(ai_df.columns.tolist())


# ============================================================
# FIND AI COLUMNS
# ============================================================

AI_SPEED_COL = None

for c in ai_df.columns:

    if "predicted" in c.lower() and "speed" in c.lower():
        AI_SPEED_COL = c
        break

if AI_SPEED_COL is None:

    for c in ai_df.columns:

        if "ai_speed" in c.lower():
            AI_SPEED_COL = c
            break


if AI_SPEED_COL is None:

    raise ValueError(
        "Could not find AI speed prediction column."
    )


print("AI speed column:", AI_SPEED_COL)


# ============================================================
# ALIGN AI RESULTS WITH RAW DATA
# ============================================================

if "time_seconds" in ai_df.columns:

    ai_df["time_seconds"] = pd.to_numeric(
        ai_df["time_seconds"],
        errors="coerce"
    )

    ai_speed = np.interp(
        df["time_seconds"].values,
        ai_df["time_seconds"].values,
        ai_df[AI_SPEED_COL].values
    )

else:

    # Fallback if AI result has same number of rows
    if len(ai_df) != len(df):

        raise ValueError(
            "AI result does not contain time_seconds and "
            "row count does not match raw dataset."
        )

    ai_speed = ai_df[AI_SPEED_COL].values


df["ai_speed_kmh"] = ai_speed


# ============================================================
# HEADING PREPROCESSING
# ============================================================

print("\nProcessing smartphone heading...")

phone_heading = df[PHONE_HEADING_COL].values


# Remove invalid values
phone_heading = np.where(
    np.isfinite(phone_heading),
    phone_heading,
    np.nan
)


# Forward/backward fill
phone_heading_series = pd.Series(phone_heading)

phone_heading_series = (
    phone_heading_series
    .interpolate(limit_direction="both")
)

phone_heading = phone_heading_series.values


# ============================================================
# GPS HEADING
# ============================================================

gps_heading = df[GPS_HEADING_COL].values

gps_heading_series = pd.Series(gps_heading)

gps_heading_series = (
    gps_heading_series
    .interpolate(limit_direction="both")
)

gps_heading = gps_heading_series.values


# ============================================================
# HEADING CALIBRATION
#
# Smartphone yaw can have a constant reference offset.
# We estimate that offset using GNSS BEFORE the outage.
#
# Important:
# GNSS heading is NOT used during the outage.
# ============================================================

pre_outage_mask = (
    (df["time_seconds"] >= OUTAGE_START - 60)
    &
    (df["time_seconds"] < OUTAGE_START)
    &
    (df[GPS_SPEED_COL] > 2.0)
    &
    np.isfinite(phone_heading)
    &
    np.isfinite(gps_heading)
)


def circular_difference(a, b):

    return (
        (a - b + 180.0) % 360.0
    ) - 180.0


heading_difference = circular_difference(
    gps_heading[pre_outage_mask],
    phone_heading[pre_outage_mask]
)


heading_offset = np.median(
    heading_difference[
        np.isfinite(heading_difference)
    ]
)

print(
    f"\nEstimated smartphone heading offset: "
    f"{heading_offset:.2f} degrees"
)


# Apply calibration
heading = (
    phone_heading + heading_offset
) % 360.0


# ============================================================
# LOCAL COORDINATE SYSTEM
#
# Latitude/longitude → local North/East meters
# ============================================================

lat = df[LAT_COL].values
lon = df[LON_COL].values


# Reference point
lat0 = lat[0]
lon0 = lon[0]

lat0_rad = np.radians(lat0)


north = (
    np.radians(lat - lat0)
    * EARTH_RADIUS
)

east = (
    np.radians(lon - lon0)
    * EARTH_RADIUS
    * np.cos(lat0_rad)
)


# ============================================================
# DEAD RECKONING INITIALIZATION
# ============================================================

time = df["time_seconds"].values

gps_north = north.copy()
gps_east = east.copy()


# DR trajectory
dr_north = np.full(
    len(df),
    np.nan
)

dr_east = np.full(
    len(df),
    np.nan
)


# ============================================================
# START DR AT OUTAGE BEGINNING
# ============================================================

start_index = np.searchsorted(
    time,
    OUTAGE_START
)

end_index = np.searchsorted(
    time,
    OUTAGE_END
)


if start_index >= len(df):

    raise ValueError(
        "Outage start is outside dataset."
    )


print(
    f"\nOutage start index: {start_index}"
)

print(
    f"Outage end index:   {end_index}"
)


# Start position from GNSS
dr_north[start_index] = north[start_index]
dr_east[start_index] = east[start_index]


# ============================================================
# DEAD RECKONING
#
# Speed:
# AI predicted speed
#
# Direction:
# calibrated smartphone heading
#
# During outage:
# NO GNSS POSITION is used.
# ============================================================

for i in range(
    start_index + 1,
    min(end_index + 1, len(df))
):

    dt = time[i] - time[i - 1]

    # Protect against timestamp anomalies
    if dt <= 0 or dt > 0.5:

        dt = 0.1


    speed_ms = (
        df["ai_speed_kmh"].iloc[i]
        / 3.6
    )


    heading_rad = np.radians(
        heading[i]
    )


    # Heading convention:
    #
    # 0°   = North
    # 90°  = East
    # 180° = South
    # 270° = West

    velocity_north = (
        speed_ms
        * np.cos(heading_rad)
    )

    velocity_east = (
        speed_ms
        * np.sin(heading_rad)
    )


    dr_north[i] = (
        dr_north[i - 1]
        + velocity_north * dt
    )

    dr_east[i] = (
        dr_east[i - 1]
        + velocity_east * dt
    )


# ============================================================
# POSITION ERROR
# ============================================================

error = np.full(
    len(df),
    np.nan
)


valid = (
    np.arange(len(df))
    >= start_index
) & (
    np.arange(len(df))
    <= end_index
)


error[valid] = np.sqrt(
    (
        dr_north[valid]
        - north[valid]
    ) ** 2
    +
    (
        dr_east[valid]
        - east[valid]
    ) ** 2
)


# ============================================================
# METRICS
# ============================================================

outage_error = error[valid]

outage_error = outage_error[
    np.isfinite(outage_error)
]


final_error = outage_error[-1]
mean_error = np.mean(outage_error)
max_error = np.max(outage_error)


# Distance traveled by GNSS during outage
gps_distance = np.sum(
    np.sqrt(
        np.diff(
            north[start_index:end_index + 1]
        ) ** 2
        +
        np.diff(
            east[start_index:end_index + 1]
        ) ** 2
    )
)


print("\n" + "=" * 70)
print("DEAD RECKONING RESULTS")
print("=" * 70)

print(
    f"Outage duration       : "
    f"{OUTAGE_DURATION:.1f} seconds"
)

print(
    f"GNSS distance traveled: "
    f"{gps_distance:.2f} m"
)

print(
    f"Mean position error   : "
    f"{mean_error:.2f} m"
)

print(
    f"Final position error  : "
    f"{final_error:.2f} m"
)

print(
    f"Maximum position error: "
    f"{max_error:.2f} m"
)


if gps_distance > 0:

    drift_percent = (
        final_error
        / gps_distance
        * 100
    )

    print(
        f"Relative drift       : "
        f"{drift_percent:.2f}%"
    )

else:

    drift_percent = np.nan


# ============================================================
# SAVE RESULTS
# ============================================================

result = pd.DataFrame({

    "time_seconds":
        time[start_index:end_index + 1],

    "gps_north_m":
        north[start_index:end_index + 1],

    "gps_east_m":
        east[start_index:end_index + 1],

    "dr_north_m":
        dr_north[start_index:end_index + 1],

    "dr_east_m":
        dr_east[start_index:end_index + 1],

    "gps_speed_kmh":
        df[GPS_SPEED_COL].iloc[
            start_index:end_index + 1
        ].values,

    "ai_speed_kmh":
        df["ai_speed_kmh"].iloc[
            start_index:end_index + 1
        ].values,

    "smartphone_heading_deg":
        heading[start_index:end_index + 1],

    "gps_heading_deg":
        gps_heading[start_index:end_index + 1],

    "position_error_m":
        error[start_index:end_index + 1]

})


result_file = os.path.join(
    PROCESSED_DIR,
    "prototype_v2_dr_results.csv"
)

result.to_csv(
    result_file,
    index=False
)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 14)
)


# ------------------------------------------------------------
# 1. TRAJECTORY
# ------------------------------------------------------------

axes[0].plot(
    east[start_index:end_index + 1],
    north[start_index:end_index + 1],
    label="GNSS Reference",
    linewidth=2
)

axes[0].plot(
    dr_east[start_index:end_index + 1],
    dr_north[start_index:end_index + 1],
    label="AI Dead Reckoning",
    linewidth=2
)

axes[0].scatter(
    east[start_index],
    north[start_index],
    s=60,
    label="Outage Start"
)

axes[0].scatter(
    east[end_index],
    north[end_index],
    s=60,
    label="GNSS End"
)

axes[0].set_title(
    "GNSS Outage: AI Dead-Reckoning Trajectory"
)

axes[0].set_xlabel(
    "East displacement (m)"
)

axes[0].set_ylabel(
    "North displacement (m)"
)

axes[0].legend()

axes[0].grid(True)


# ------------------------------------------------------------
# 2. SPEED
# ------------------------------------------------------------

axes[1].plot(
    time[start_index:end_index + 1],
    df[GPS_SPEED_COL].iloc[
        start_index:end_index + 1
    ],
    label="GNSS Speed",
    linewidth=2
)

axes[1].plot(
    time[start_index:end_index + 1],
    df["ai_speed_kmh"].iloc[
        start_index:end_index + 1
    ],
    label="AI Speed",
    linewidth=2
)

axes[1].set_title(
    "AI Speed Estimation During GNSS Outage"
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "Speed (km/h)"
)

axes[1].legend()

axes[1].grid(True)


# ------------------------------------------------------------
# 3. POSITION ERROR
# ------------------------------------------------------------

axes[2].plot(
    time[start_index:end_index + 1],
    error[start_index:end_index + 1],
    linewidth=2
)

axes[2].set_title(
    f"Dead-Reckoning Position Error "
    f"(Final = {final_error:.2f} m)"
)

axes[2].set_xlabel(
    "Time (s)"
)

axes[2].set_ylabel(
    "Position error (m)"
)

axes[2].grid(True)


plt.tight_layout()


plot_file = os.path.join(
    OUTPUT_DIR,
    "prototype_v2_dead_reckoning.png"
)

plt.savefig(
    plot_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\nFiles created:")

print(
    f"  {result_file}"
)

print(
    f"  {plot_file}"
)

print("\nPrototype V2 completed.")