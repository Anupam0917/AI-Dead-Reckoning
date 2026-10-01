import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SIH 2026
# Prototype V4
#
# Heading Validation
#
# Compare:
#   1. Smartphone orientation yaw
#   2. GPS orientation/course
#
# Goal:
#   Determine whether the phone yaw can be converted into
#   a useful navigation heading.
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Synchronised V abd S datasets",
    "Categorised IOVNB Dataset",
    "M (Driver B)",
    "S-M.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

OUTAGE_START = 7420.42

# Use a large period before outage for calibration
CALIBRATION_START = OUTAGE_START - 300.0
CALIBRATION_END = OUTAGE_START


# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 70)
print("SIH PROTOTYPE V4 - HEADING VALIDATION")
print("=" * 70)

df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

print(
    f"Rows: {len(df)}"
)


# ============================================================
# CLEAN COLUMNS
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_", regex=False)
)


def find_column(keyword):

    matches = [
        c for c in df.columns
        if keyword.lower() in c.lower()
    ]

    if not matches:

        raise ValueError(
            f"Column not found: {keyword}"
        )

    return matches[0]


TIME_COL = find_column(
    "DATE"
)

GPS_HEADING_COL = find_column(
    "GPS_ORIENTATION"
)

PHONE_HEADING_COL = find_column(
    "ORIENTATION_Yaw"
)

GPS_SPEED_COL = find_column(
    "GPS_SPEED"
)


print("\nColumns:")

print(
    "Timestamp:",
    TIME_COL
)

print(
    "GPS heading:",
    GPS_HEADING_COL
)

print(
    "Phone yaw:",
    PHONE_HEADING_COL
)

print(
    "GPS speed:",
    GPS_SPEED_COL
)


# ============================================================
# TIME
# ============================================================

df["timestamp"] = pd.to_datetime(
    df[TIME_COL],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)

df["time_seconds"] = (
    df["timestamp"]
    -
    df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# NUMERIC
# ============================================================

gps_heading = pd.to_numeric(
    df[GPS_HEADING_COL],
    errors="coerce"
).values

phone_heading = pd.to_numeric(
    df[PHONE_HEADING_COL],
    errors="coerce"
).values

gps_speed = pd.to_numeric(
    df[GPS_SPEED_COL],
    errors="coerce"
).values


# ============================================================
# INTERPOLATION
# ============================================================

phone_heading = (
    pd.Series(phone_heading)
    .interpolate(
        limit_direction="both"
    )
    .values
)

gps_heading = (
    pd.Series(gps_heading)
    .interpolate(
        limit_direction="both"
    )
    .values
)


# ============================================================
# CIRCULAR FUNCTIONS
# ============================================================

def circular_difference(a, b):

    return (
        (a - b + 180.0)
        % 360.0
    ) - 180.0


def circular_mean(values):

    values = np.asarray(values)

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    radians = np.radians(
        values
    )

    return np.degrees(
        np.arctan2(
            np.mean(np.sin(radians)),
            np.mean(np.cos(radians))
        )
    ) % 360.0


# ============================================================
# CALIBRATION MASK
# ============================================================

mask = (
    (df["time_seconds"].values >=
     CALIBRATION_START)
    &
    (df["time_seconds"].values <
     CALIBRATION_END)
    &
    np.isfinite(gps_heading)
    &
    np.isfinite(phone_heading)
    &
    np.isfinite(gps_speed)
    &
    (gps_speed > 3.0)
)


print(
    f"\nCalibration samples: "
    f"{mask.sum()}"
)


# ============================================================
# RAW DIFFERENCE
# ============================================================

difference = circular_difference(
    gps_heading,
    phone_heading
)


valid_difference = difference[
    mask
]


print(
    "\nRAW PHONE → GPS HEADING DIFFERENCE"
)

print(
    f"Mean   : "
    f"{np.mean(valid_difference):.3f}°"
)

print(
    f"Median : "
    f"{np.median(valid_difference):.3f}°"
)

print(
    f"Std    : "
    f"{np.std(valid_difference):.3f}°"
)

print(
    f"Min    : "
    f"{np.min(valid_difference):.3f}°"
)

print(
    f"Max    : "
    f"{np.max(valid_difference):.3f}°"
)


# ============================================================
# CIRCULAR OFFSET
# ============================================================

offset = circular_mean(
    valid_difference
)


print(
    f"\nCircular heading offset:"
)

print(
    f"{offset:.3f}°"
)


# ============================================================
# APPLY OFFSET
# ============================================================

calibrated_heading = (
    phone_heading
    + offset
) % 360.0


calibrated_error = circular_difference(
    gps_heading,
    calibrated_heading
)


valid_error = calibrated_error[
    mask
]


print(
    "\nCALIBRATED HEADING ERROR"
)

print(
    f"Mean absolute error: "
    f"{np.mean(np.abs(valid_error)):.3f}°"
)

print(
    f"Median absolute error: "
    f"{np.median(np.abs(valid_error)):.3f}°"
)

print(
    f"Std: "
    f"{np.std(valid_error):.3f}°"
)


for threshold in [
    5,
    10,
    20,
    30
]:

    percentage = (
        np.mean(
            np.abs(valid_error)
            <= threshold
        )
        * 100
    )

    print(
        f"Within ±{threshold}°: "
        f"{percentage:.2f}%"
    )


# ============================================================
# ERROR DURING OUTAGE
# ============================================================

outage_mask = (
    (df["time_seconds"].values >=
     OUTAGE_START)
    &
    (df["time_seconds"].values <=
     OUTAGE_START + 60)
    &
    np.isfinite(calibrated_error)
)


outage_error = calibrated_error[
    outage_mask
]


print(
    "\nHEADING ERROR IN 60s OUTAGE REGION"
)

print(
    f"Mean absolute error: "
    f"{np.mean(np.abs(outage_error)):.3f}°"
)

print(
    f"Median absolute error: "
    f"{np.median(np.abs(outage_error)):.3f}°"
)

print(
    f"Maximum absolute error: "
    f"{np.max(np.abs(outage_error)):.3f}°"
)


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame({

    "time_seconds":
        df["time_seconds"],

    "gps_heading_deg":
        gps_heading,

    "phone_heading_deg":
        phone_heading,

    "calibrated_heading_deg":
        calibrated_heading,

    "heading_error_deg":
        calibrated_error,

    "gps_speed_kmh":
        gps_speed

})


output_file = os.path.join(
    PROCESSED_DIR,
    "prototype_v4_heading_validation.csv"
)

result.to_csv(
    output_file,
    index=False
)


# ============================================================
# PLOT
# ============================================================

plot_mask = (
    (df["time_seconds"].values >=
     OUTAGE_START - 120)
    &
    (df["time_seconds"].values <=
     OUTAGE_START + 120)
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(13, 10)
)


# ------------------------------------------------------------
# Heading
# ------------------------------------------------------------

axes[0].plot(
    df["time_seconds"].values[plot_mask],
    gps_heading[plot_mask],
    label="GPS Heading",
    linewidth=2
)

axes[0].plot(
    df["time_seconds"].values[plot_mask],
    calibrated_heading[plot_mask],
    label="Calibrated Phone Heading",
    linewidth=2
)

axes[0].axvspan(
    OUTAGE_START,
    OUTAGE_START + 60,
    alpha=0.2,
    label="GNSS Outage"
)

axes[0].set_title(
    "Heading Comparison Around GNSS Outage"
)

axes[0].set_xlabel(
    "Time (s)"
)

axes[0].set_ylabel(
    "Heading (degrees)"
)

axes[0].legend()

axes[0].grid(True)


# ------------------------------------------------------------
# Error
# ------------------------------------------------------------

axes[1].plot(
    df["time_seconds"].values[plot_mask],
    calibrated_error[plot_mask],
    linewidth=2
)

axes[1].axhline(
    0,
    linewidth=1
)

axes[1].axvspan(
    OUTAGE_START,
    OUTAGE_START + 60,
    alpha=0.2,
    label="GNSS Outage"
)

axes[1].set_title(
    "Phone Heading Error Relative to GPS Heading"
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "Heading error (degrees)"
)

axes[1].legend()

axes[1].grid(True)


plt.tight_layout()


plot_file = os.path.join(
    OUTPUT_DIR,
    "prototype_v4_heading_validation.png"
)

plt.savefig(
    plot_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    "\nFiles created:"
)

print(
    output_file
)

print(
    plot_file
)

print(
    "\nV4 heading validation completed."
)