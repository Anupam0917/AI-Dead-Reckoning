import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V9 STEP 1
# GNSS SPEED/COURSE -> NORTH/EAST VELOCITY TARGETS
#
# IO-VNBD:
#   Smartphone sensors ~= 10 Hz
#   GNSS information ~= 1 Hz
#
# We use GNSS speed + GNSS orientation to construct:
#
#       Vnorth
#       Veast
#
# Then interpolate these targets onto the 10 Hz timeline.
#
# GNSS is used ONLY as the supervised training target.
# ============================================================


print("=" * 70)
print("V9 NORTH/EAST VELOCITY TARGET GENERATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    "data/raw/"
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "prototype_v9_velocity_targets.csv"
)

OUTPUT_PLOT = (
    "outputs/"
    "prototype_v9_velocity_targets.png"
)

SPEED_PLOT = (
    "outputs/"
    "prototype_v9_target_speed_validation.png"
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_SPEED_KMH = 180.0


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading IO-VNBD dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252"
)

print(
    f"Rows loaded: {len(df)}"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

LAT_COL = "GPS LATITUDE (degrees)"

LON_COL = "GPS LONGITUDE (degrees)"

SPEED_COL = "GPS SPEED (Kmh)"

GPS_HEADING_COL = "GPS ORIENTATION (Â°)"

TIME_COL = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"


required_columns = [
    LAT_COL,
    LON_COL,
    SPEED_COL,
    GPS_HEADING_COL,
    TIME_COL
]


missing = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing:

    print("\nAvailable columns:")

    for col in df.columns:
        print(
            " ",
            repr(col)
        )

    raise KeyError(
        f"Missing columns: {missing}"
    )


print("\nDetected columns:")

print(
    "Latitude :",
    LAT_COL
)

print(
    "Longitude:",
    LON_COL
)

print(
    "GPS speed:",
    SPEED_COL
)

print(
    "GPS heading:",
    GPS_HEADING_COL
)

print(
    "Timestamp:",
    TIME_COL
)


# ============================================================
# TIMESTAMP
# ============================================================

print("\nParsing timestamps...")

df["timestamp"] = pd.to_datetime(
    df[TIME_COL],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)


if df["timestamp"].isna().mean() > 0.5:

    df["timestamp"] = pd.to_datetime(
        df[TIME_COL],
        errors="coerce"
    )


if df["timestamp"].isna().all():

    raise RuntimeError(
        "Timestamp parsing failed."
    )


# ============================================================
# NUMERIC DATA
# ============================================================

df["latitude"] = pd.to_numeric(
    df[LAT_COL],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df[LON_COL],
    errors="coerce"
)

df["gps_speed_kmh"] = pd.to_numeric(
    df[SPEED_COL],
    errors="coerce"
)

df["gps_heading_deg"] = pd.to_numeric(
    df[GPS_HEADING_COL],
    errors="coerce"
)


# ============================================================
# SORT BY TIME
# ============================================================

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# RELATIVE TIME
# ============================================================

df["time_s"] = (
    df["timestamp"]
    -
    df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# VALID GNSS DATA
# ============================================================

valid_gnss = (
    df["timestamp"].notna()
    &
    df["gps_speed_kmh"].notna()
    &
    df["gps_heading_deg"].notna()
)


print(
    f"\nRows with GNSS speed + heading: "
    f"{valid_gnss.sum()} / {len(df)}"
)


# ============================================================
# LIMIT IMPOSSIBLE SPEEDS
# ============================================================

df.loc[
    df["gps_speed_kmh"] < 0,
    "gps_speed_kmh"
] = np.nan


df.loc[
    df["gps_speed_kmh"] > MAX_SPEED_KMH,
    "gps_speed_kmh"
] = np.nan


# ============================================================
# NORMALIZE HEADING
# ============================================================

df["gps_heading_deg"] = (
    df["gps_heading_deg"]
    % 360.0
)


# ============================================================
# CONVERT SPEED
# ============================================================

df["gps_speed_mps"] = (
    df["gps_speed_kmh"]
    /
    3.6
)


# ============================================================
# GPS COURSE -> N/E VELOCITY
# ============================================================
#
# Heading convention:
#
#   0°   = North
#   90°  = East
#   180° = South
#   270° = West
#
# Therefore:
#
#   Vnorth = V * cos(theta)
#   Veast  = V * sin(theta)
#
# ============================================================

heading_rad = np.deg2rad(
    df["gps_heading_deg"]
)


df["vn_gnss_mps"] = (
    df["gps_speed_mps"]
    *
    np.cos(heading_rad)
)


df["ve_gnss_mps"] = (
    df["gps_speed_mps"]
    *
    np.sin(heading_rad)
)


# ============================================================
# GPS FIX TIMELINE
# ============================================================
#
# GNSS values are repeated across the 10 Hz smartphone rows.
#
# Keep one representative row whenever the GPS coordinate
# changes, OR whenever the GPS speed/course changes.
#
# ============================================================

position_changed = (
    df["latitude"].ne(
        df["latitude"].shift()
    )
    |
    df["longitude"].ne(
        df["longitude"].shift()
    )
)


speed_changed = (
    df["gps_speed_kmh"].ne(
        df["gps_speed_kmh"].shift()
    )
)


heading_changed = (
    df["gps_heading_deg"].ne(
        df["gps_heading_deg"].shift()
    )
)


gnss_fix_changed = (
    position_changed
    |
    speed_changed
    |
    heading_changed
)


gnss = df.loc[
    gnss_fix_changed,
    [
        "timestamp",
        "time_s",
        "latitude",
        "longitude",
        "gps_speed_kmh",
        "gps_heading_deg",
        "vn_gnss_mps",
        "ve_gnss_mps"
    ]
].copy()


gnss = (
    gnss
    .sort_values("time_s")
    .reset_index(drop=True)
)


print(
    f"\nDetected GNSS fix/change samples: "
    f"{len(gnss)}"
)


# ============================================================
# REMOVE INVALID TARGETS
# ============================================================

gnss = gnss[
    gnss["vn_gnss_mps"].notna()
    &
    gnss["ve_gnss_mps"].notna()
].copy()


# ============================================================
# SMOOTH TARGETS SLIGHTLY
# ============================================================

gnss["vn_gnss_mps"] = (
    gnss["vn_gnss_mps"]
    .rolling(
        window=3,
        center=True,
        min_periods=1
    )
    .median()
)


gnss["ve_gnss_mps"] = (
    gnss["ve_gnss_mps"]
    .rolling(
        window=3,
        center=True,
        min_periods=1
    )
    .median()
)


# ============================================================
# TARGET SPEED
# ============================================================

gnss["target_speed_mps"] = np.sqrt(
    gnss["vn_gnss_mps"] ** 2
    +
    gnss["ve_gnss_mps"] ** 2
)


gnss["target_speed_kmh"] = (
    gnss["target_speed_mps"]
    *
    3.6
)


# ============================================================
# CREATE 10 Hz TARGET DATASET
# ============================================================

print(
    "\nInterpolating GNSS targets "
    "onto smartphone timeline..."
)


target = df[
    [
        "timestamp",
        "time_s",
        "latitude",
        "longitude",
        "gps_speed_kmh",
        "gps_heading_deg"
    ]
].copy()


# ============================================================
# INTERPOLATION
# ============================================================

target["vn_mps"] = np.interp(
    target["time_s"],
    gnss["time_s"].values,
    gnss["vn_gnss_mps"].values
)


target["ve_mps"] = np.interp(
    target["time_s"],
    gnss["time_s"].values,
    gnss["ve_gnss_mps"].values
)


# ============================================================
# TARGET MAGNITUDE
# ============================================================

target["velocity_mps"] = np.sqrt(
    target["vn_mps"] ** 2
    +
    target["ve_mps"] ** 2
)


target["velocity_kmh"] = (
    target["velocity_mps"]
    *
    3.6
)


# ============================================================
# VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("V9 VELOCITY TARGET VALIDATION")
print("=" * 70)


print(
    f"\nGNSS target samples: "
    f"{len(gnss)}"
)


print(
    f"10 Hz target samples: "
    f"{len(target)}"
)


print(
    f"\nMean target speed: "
    f"{target['velocity_kmh'].mean():.3f} km/h"
)


print(
    f"Median target speed: "
    f"{target['velocity_kmh'].median():.3f} km/h"
)


print(
    f"Maximum target speed: "
    f"{target['velocity_kmh'].max():.3f} km/h"
)


# ============================================================
# NORTH VELOCITY
# ============================================================

print("\nNorth velocity:")

print(
    f"  Mean   : "
    f"{target['vn_mps'].mean():.4f} m/s"
)

print(
    f"  Std    : "
    f"{target['vn_mps'].std():.4f} m/s"
)

print(
    f"  Minimum: "
    f"{target['vn_mps'].min():.4f} m/s"
)

print(
    f"  Maximum: "
    f"{target['vn_mps'].max():.4f} m/s"
)


# ============================================================
# EAST VELOCITY
# ============================================================

print("\nEast velocity:")

print(
    f"  Mean   : "
    f"{target['ve_mps'].mean():.4f} m/s"
)

print(
    f"  Std    : "
    f"{target['ve_mps'].std():.4f} m/s"
)

print(
    f"  Minimum: "
    f"{target['ve_mps'].min():.4f} m/s"
)

print(
    f"  Maximum: "
    f"{target['ve_mps'].max():.4f} m/s"
)


# ============================================================
# COMPARE TARGET SPEED WITH ORIGINAL GPS SPEED
# ============================================================

valid_speed = (
    target["gps_speed_kmh"].notna()
    &
    target["velocity_kmh"].notna()
)


if valid_speed.sum() > 0:

    error = (
        target.loc[
            valid_speed,
            "velocity_kmh"
        ]
        -
        target.loc[
            valid_speed,
            "gps_speed_kmh"
        ]
    )


    print(
        "\nTarget speed vs original GPS speed:"
    )


    print(
        f"Mean difference: "
        f"{error.mean():.3f} km/h"
    )


    print(
        f"MAE: "
        f"{np.abs(error).mean():.3f} km/h"
    )


    print(
        f"Median absolute difference: "
        f"{np.median(np.abs(error)):.3f} km/h"
    )


    target_values = target.loc[
        valid_speed,
        "velocity_kmh"
    ].values


    gps_values = target.loc[
        valid_speed,
        "gps_speed_kmh"
    ].values


    if (
        np.std(target_values) > 0
        and
        np.std(gps_values) > 0
    ):

        correlation = np.corrcoef(
            target_values,
            gps_values
        )[0, 1]

    else:

        correlation = np.nan


    print(
        f"Correlation: "
        f"{correlation:.4f}"
    )


# ============================================================
# OUTPUT DATASET
# ============================================================

output = target[
    [
        "timestamp",
        "time_s",
        "latitude",
        "longitude",
        "gps_speed_kmh",
        "gps_heading_deg",
        "vn_mps",
        "ve_mps",
        "velocity_mps",
        "velocity_kmh"
    ]
].copy()


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

os.makedirs(
    "outputs",
    exist_ok=True
)


output.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# PLOT 1
# ============================================================

plt.figure(
    figsize=(12, 7)
)


plt.plot(
    output["time_s"],
    output["vn_mps"],
    label="North velocity",
    linewidth=1
)


plt.plot(
    output["time_s"],
    output["ve_mps"],
    label="East velocity",
    linewidth=1
)


plt.xlabel(
    "Time (s)"
)


plt.ylabel(
    "Velocity (m/s)"
)


plt.title(
    "V9 GNSS-Derived North/East Velocity Targets"
)


plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_PLOT,
    dpi=200
)


plt.close()


# ============================================================
# PLOT 2
# ============================================================

plt.figure(
    figsize=(12, 7)
)


plt.plot(
    output["time_s"],
    output["gps_speed_kmh"],
    label="GNSS speed",
    linewidth=1
)


plt.plot(
    output["time_s"],
    output["velocity_kmh"],
    label="N/E target speed",
    linewidth=1
)


plt.xlabel(
    "Time (s)"
)


plt.ylabel(
    "Speed (km/h)"
)


plt.title(
    "V9 GNSS Velocity Target Validation"
)


plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    SPEED_PLOT,
    dpi=200
)


plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("V9 STEP 1 COMPLETE")
print("=" * 70)


print(
    "\nFiles created:"
)


print(
    "1.",
    OUTPUT_PATH
)


print(
    "2.",
    OUTPUT_PLOT
)


print(
    "3.",
    SPEED_PLOT
)


print(
    "\nTarget definition:"
)

print(
    "GNSS speed + GNSS course -> Vnorth + Veast"
)


print(
    "\nThese targets can now be used for supervised AI training."
)


print("=" * 70)