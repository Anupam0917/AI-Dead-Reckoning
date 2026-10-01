import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SIH 2026
# PROTOTYPE V5
#
# IMU + MAGNETOMETER HEADING ESTIMATION
#
# Gyroscope provides short-term heading change.
# Magnetometer provides absolute heading correction.
#
# GNSS heading is used ONLY for evaluation.
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
# CONFIGURATION
# ============================================================

OUTAGE_START = 7420.42
OUTAGE_END = OUTAGE_START + 60.0

EARTH_RADIUS = 6371000.0


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("SIH PROTOTYPE V5 - IMU HEADING ESTIMATION")
print("=" * 75)

df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

print(
    f"Rows: {len(df)}"
)


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


def find_column(keyword):

    matches = [
        c
        for c in df.columns
        if keyword.lower() in c.lower()
    ]

    if not matches:

        raise ValueError(
            f"Column not found: {keyword}"
        )

    return matches[0]


# ============================================================
# COLUMNS
# ============================================================

TIME_COL = find_column(
    "DATE"
)

GPS_HEADING_COL = find_column(
    "GPS_ORIENTATION"
)

GPS_SPEED_COL = find_column(
    "GPS_SPEED"
)

GYRO_YAW_COL = find_column(
    "GYROSCOPE_Yaw"
)

MAG_X_COL = find_column(
    "MAGNETIC_FIELD_X"
)

MAG_Y_COL = find_column(
    "MAGNETIC_FIELD_Y"
)

MAG_Z_COL = find_column(
    "MAGNETIC_FIELD_Z"
)


print("\nDetected columns:")

print(
    "Timestamp       :",
    TIME_COL
)

print(
    "GPS heading     :",
    GPS_HEADING_COL
)

print(
    "GPS speed       :",
    GPS_SPEED_COL
)

print(
    "Gyroscope yaw   :",
    GYRO_YAW_COL
)

print(
    "Magnetic X      :",
    MAG_X_COL
)

print(
    "Magnetic Y      :",
    MAG_Y_COL
)

print(
    "Magnetic Z      :",
    MAG_Z_COL
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


time = df["time_seconds"].values


# ============================================================
# NUMERIC DATA
# ============================================================

gps_heading = pd.to_numeric(
    df[GPS_HEADING_COL],
    errors="coerce"
).values

gps_speed = pd.to_numeric(
    df[GPS_SPEED_COL],
    errors="coerce"
).values

gyro_yaw = pd.to_numeric(
    df[GYRO_YAW_COL],
    errors="coerce"
).values

mag_x = pd.to_numeric(
    df[MAG_X_COL],
    errors="coerce"
).values

mag_y = pd.to_numeric(
    df[MAG_Y_COL],
    errors="coerce"
).values

mag_z = pd.to_numeric(
    df[MAG_Z_COL],
    errors="coerce"
).values


# ============================================================
# INTERPOLATE
# ============================================================

def interpolate(values):

    return (
        pd.Series(values)
        .interpolate(
            limit_direction="both"
        )
        .values
    )


gps_heading = interpolate(
    gps_heading
)

gps_speed = interpolate(
    gps_speed
)

gyro_yaw = interpolate(
    gyro_yaw
)

mag_x = interpolate(
    mag_x
)

mag_y = interpolate(
    mag_y
)

mag_z = interpolate(
    mag_z
)


# ============================================================
# MAGNETOMETER HEADING
#
# Basic horizontal magnetic heading.
#
# This is intentionally treated as a prototype measurement.
# Vehicle magnetic disturbances can affect it.
# ============================================================

mag_heading = np.degrees(
    np.arctan2(
        -mag_y,
        mag_x
    )
)

mag_heading = (
    mag_heading
    + 360.0
) % 360.0


# ============================================================
# CIRCULAR FUNCTIONS
# ============================================================

def wrap_angle(angle):

    return (
        angle + 180.0
    ) % 360.0 - 180.0


def circular_difference(
    a,
    b
):

    return wrap_angle(
        a - b
    )


def circular_mean(values):

    values = np.asarray(
        values
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    r = np.radians(
        values
    )

    return (
        np.degrees(
            np.arctan2(
                np.mean(
                    np.sin(r)
                ),
                np.mean(
                    np.cos(r)
                )
            )
        )
        % 360.0
    )


# ============================================================
# MAGNETOMETER OFFSET CALIBRATION
#
# Estimate a constant offset using GNSS heading before outage.
# GNSS heading is NOT used to propagate the outage.
# ============================================================

calibration_mask = (
    (time >= OUTAGE_START - 300.0)
    &
    (time < OUTAGE_START)
    &
    (gps_speed > 3.0)
    &
    np.isfinite(gps_heading)
    &
    np.isfinite(mag_heading)
)


mag_difference = circular_difference(
    gps_heading,
    mag_heading
)


offset = circular_mean(
    mag_difference[
        calibration_mask
    ]
)


if not np.isfinite(offset):

    offset = 0.0


print(
    f"\nMagnetic heading offset:"
    f" {offset:.2f} degrees"
)


mag_heading_calibrated = (
    mag_heading
    + offset
) % 360.0


# ============================================================
# INITIAL HEADING
#
# Use GNSS heading immediately before outage as initialization.
# After that, heading evolves from gyro + magnetometer.
# ============================================================

initial_mask = (
    (time >= OUTAGE_START - 5.0)
    &
    (time < OUTAGE_START)
    &
    (gps_speed > 2.0)
    &
    np.isfinite(gps_heading)
)


if np.any(initial_mask):

    initial_heading = circular_mean(
        gps_heading[
            initial_mask
        ]
    )

else:

    initial_heading = (
        mag_heading_calibrated[
            np.searchsorted(
                time,
                OUTAGE_START
            )
        ]
    )


print(
    f"Initial heading:"
    f" {initial_heading:.2f} degrees"
)


# ============================================================
# HEADING ESTIMATOR
#
# Complementary fusion:
#
# gyro:
#   fast response
#
# magnetometer:
#   slow absolute correction
# ============================================================

estimated_heading = np.full(
    len(df),
    np.nan
)

estimated_heading[0] = (
    initial_heading
)


# Correction strength
MAG_GAIN = 0.015


for i in range(
    1,
    len(df)
):

    dt = (
        time[i]
        -
        time[i - 1]
    )

    if (
        dt <= 0
        or dt > 0.5
    ):

        dt = 0.1


    # --------------------------------------------------------
    # Gyroscope integration
    # --------------------------------------------------------

    previous_heading = (
        estimated_heading[i - 1]
    )


    gyro_prediction = (
        previous_heading
        +
        np.degrees(
            gyro_yaw[i]
            * dt
        )
    )


    gyro_prediction %= 360.0


    # --------------------------------------------------------
    # Magnetometer correction
    # --------------------------------------------------------

    mag_error = circular_difference(
        mag_heading_calibrated[i],
        gyro_prediction
    )


    corrected_heading = (
        gyro_prediction
        +
        MAG_GAIN * mag_error
    )


    estimated_heading[i] = (
        corrected_heading
        % 360.0
    )


# ============================================================
# EVALUATION
# ============================================================

heading_error = circular_difference(
    gps_heading,
    estimated_heading
)


# Calibration region
calibration_error = heading_error[
    calibration_mask
]


print(
    "\n" + "=" * 75
)

print(
    "HEADING ESTIMATION RESULTS"
)

print(
    "=" * 75
)

print(
    "\nCalibration region:"
)

print(
    f"Mean absolute error : "
    f"{np.mean(np.abs(calibration_error)):.3f}°"
)

print(
    f"Median absolute error: "
    f"{np.median(np.abs(calibration_error)):.3f}°"
)


# ============================================================
# OUTAGE EVALUATION
# ============================================================

outage_mask = (
    (time >= OUTAGE_START)
    &
    (time <= OUTAGE_END)
)


outage_error = heading_error[
    outage_mask
]


print(
    "\n60-second GNSS outage:"
)

print(
    f"Mean absolute error : "
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


for threshold in [
    5,
    10,
    20,
    30
]:

    percentage = (
        np.mean(
            np.abs(outage_error)
            <= threshold
        )
        * 100
    )

    print(
        f"Within ±{threshold}°: "
        f"{percentage:.2f}%"
    )


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame({

    "timestamp":
        df["timestamp"],

    "time_seconds":
        time,

    "gps_heading_deg":
        gps_heading,

    "mag_heading_deg":
        mag_heading_calibrated,

    "estimated_heading_deg":
        estimated_heading,

    "heading_error_deg":
        heading_error,

    "gps_speed_kmh":
        gps_speed,

    "gyro_yaw_rad_s":
        gyro_yaw

})


result_file = os.path.join(
    PROCESSED_DIR,
    "prototype_v5_imu_heading.csv"
)

result.to_csv(
    result_file,
    index=False
)


# ============================================================
# PLOT
# ============================================================

plot_mask = (
    (time >= OUTAGE_START - 120.0)
    &
    (time <= OUTAGE_END + 120.0)
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 10)
)


# ------------------------------------------------------------
# HEADING
# ------------------------------------------------------------

axes[0].plot(
    time[plot_mask],
    gps_heading[plot_mask],
    label="GNSS Heading",
    linewidth=2
)

axes[0].plot(
    time[plot_mask],
    mag_heading_calibrated[plot_mask],
    label="Magnetometer",
    linewidth=1.5
)

axes[0].plot(
    time[plot_mask],
    estimated_heading[plot_mask],
    label="IMU + Magnetometer",
    linewidth=2
)

axes[0].axvspan(
    OUTAGE_START,
    OUTAGE_END,
    alpha=0.2,
    label="GNSS Outage"
)

axes[0].set_title(
    "IMU + Magnetometer Heading Estimation"
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
# ERROR
# ------------------------------------------------------------

axes[1].plot(
    time[plot_mask],
    heading_error[plot_mask],
    linewidth=2
)

axes[1].axhline(
    0,
    linewidth=1
)

axes[1].axhline(
    10,
    linestyle="--",
    linewidth=1
)

axes[1].axhline(
    -10,
    linestyle="--",
    linewidth=1
)

axes[1].axvspan(
    OUTAGE_START,
    OUTAGE_END,
    alpha=0.2,
    label="GNSS Outage"
)

axes[1].set_title(
    "Heading Estimation Error"
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "Error (degrees)"
)

axes[1].legend()

axes[1].grid(True)


plt.tight_layout()


plot_file = os.path.join(
    OUTPUT_DIR,
    "prototype_v5_imu_heading.png"
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
    result_file
)

print(
    plot_file
)

print(
    "\nV5 completed successfully."
)