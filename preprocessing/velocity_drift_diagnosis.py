import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 6.1
# VELOCITY DRIFT DIAGNOSIS
# ============================================================

print("=" * 70)
print("STEP 6.1")
print("VELOCITY DRIFT DIAGNOSIS")
print("=" * 70)

print("""
Pipeline:
IMU velocity
      ↓
GPS comparison
      ↓
velocity error
      ↓
find drift regions
      ↓
inspect NED acceleration
      ↓
diagnostic plots
""")


# ============================================================
# PATHS
# ============================================================

VELOCITY_FILE = "data/processed/imu_velocity.csv"

NED_FILE = "data/processed/ned_acceleration.csv"

RAW_FILE = (
    "data/raw/"
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def find_column(df, candidates, description):
    """
    Find a column using:
    1. Exact matching
    2. Partial matching
    """

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    # --------------------------------------------------------
    # Partial match
    # --------------------------------------------------------

    lower_columns = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        candidate_lower = candidate.lower()

        for lower_column, original_column in lower_columns.items():

            if candidate_lower in lower_column:

                return original_column

    raise ValueError(
        f"\nCould not find {description} column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# TIMESTAMP FUNCTION
# ============================================================

def prepare_timestamp(df):
    """
    Convert timestamp/date column into pandas datetime.
    """

    # --------------------------------------------------------
    # Existing timestamp column
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        print(
            "Using existing timestamp column: timestamp"
        )

        timestamp = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        return timestamp

    # --------------------------------------------------------
    # Known raw dataset date column
    # --------------------------------------------------------

    known_columns = [
        "DATE (YYYY-MO-DD HH-MI-SS_SSS)",
        "DATE_YYYY-MO-DD_HH-MI-SS_SSS"
    ]

    for column in known_columns:

        if column in df.columns:

            print(
                f"Using date column: {column}"
            )

            timestamp = pd.to_datetime(
                df[column],
                format="%Y-%m-%d %H:%M:%S:%f",
                errors="coerce"
            )

            return timestamp

    # --------------------------------------------------------
    # Flexible detection
    # --------------------------------------------------------

    for column in df.columns:

        column_name = str(column).upper()

        if "DATE" in column_name:

            print(
                f"Using detected date column: {column}"
            )

            timestamp = pd.to_datetime(
                df[column],
                format="%Y-%m-%d %H:%M:%S:%f",
                errors="coerce"
            )

            return timestamp

    raise ValueError(
        "Could not find timestamp/date column."
    )


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)


# ------------------------------------------------------------
# Velocity
# ------------------------------------------------------------

print("\nLoading velocity data...")

velocity_df = pd.read_csv(
    VELOCITY_FILE
)

print(
    "Rows    :",
    len(velocity_df)
)

print(
    "Columns :",
    len(velocity_df.columns)
)


# ------------------------------------------------------------
# NED acceleration
# ------------------------------------------------------------

print("\nLoading NED acceleration...")

ned_df = pd.read_csv(
    NED_FILE
)

print(
    "Rows    :",
    len(ned_df)
)

print(
    "Columns :",
    len(ned_df.columns)
)


# ------------------------------------------------------------
# Raw dataset
# ------------------------------------------------------------

print("\nLoading raw sensor data...")

raw_df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

print(
    "Rows    :",
    len(raw_df)
)

print(
    "Columns :",
    len(raw_df.columns)
)


# ============================================================
# DATA ALIGNMENT
# ============================================================

print()
print("=" * 70)
print("DATA ALIGNMENT")
print("=" * 70)


print(
    "Velocity rows :",
    len(velocity_df)
)

print(
    "NED rows      :",
    len(ned_df)
)

print(
    "Raw rows      :",
    len(raw_df)
)


if not (
    len(velocity_df)
    ==
    len(ned_df)
    ==
    len(raw_df)
):

    raise ValueError(
        "Dataset row counts do not match."
    )


print(
    "Row alignment : OK"
)


# ============================================================
# TIMESTAMPS
# ============================================================

print()
print("=" * 70)
print("PREPARING TIMESTAMP")
print("=" * 70)


velocity_timestamp = prepare_timestamp(
    velocity_df
)

ned_timestamp = prepare_timestamp(
    ned_df
)

raw_timestamp = prepare_timestamp(
    raw_df
)


print(
    "Velocity invalid timestamps :",
    velocity_timestamp.isna().sum()
)

print(
    "NED invalid timestamps      :",
    ned_timestamp.isna().sum()
)

print(
    "Raw invalid timestamps      :",
    raw_timestamp.isna().sum()
)


# ============================================================
# CHECK TIMESTAMP ALIGNMENT
# ============================================================

print()
print("=" * 70)
print("TIMESTAMP ALIGNMENT")
print("=" * 70)


velocity_start = velocity_timestamp.iloc[0]
velocity_end = velocity_timestamp.iloc[-1]

ned_start = ned_timestamp.iloc[0]
ned_end = ned_timestamp.iloc[-1]

raw_start = raw_timestamp.iloc[0]
raw_end = raw_timestamp.iloc[-1]


print(
    "Velocity start :",
    velocity_start
)

print(
    "Velocity end   :",
    velocity_end
)

print(
    "NED start      :",
    ned_start
)

print(
    "NED end        :",
    ned_end
)

print(
    "Raw start      :",
    raw_start
)

print(
    "Raw end        :",
    raw_end
)


# ============================================================
# FIND VELOCITY COLUMNS
# ============================================================

print()
print("=" * 70)
print("VELOCITY COLUMNS")
print("=" * 70)


velocity_north_col = find_column(
    velocity_df,
    [
        "velocity_north_raw_m_s",
        "velocity_north",
        "vel_north",
        "north_velocity"
    ],
    "north velocity"
)


velocity_east_col = find_column(
    velocity_df,
    [
        "velocity_east_raw_m_s",
        "velocity_east",
        "vel_east",
        "east_velocity"
    ],
    "east velocity"
)


velocity_down_col = find_column(
    velocity_df,
    [
        "velocity_down_raw_m_s",
        "velocity_down",
        "vel_down",
        "down_velocity"
    ],
    "down velocity"
)


print(
    "North :",
    velocity_north_col
)

print(
    "East  :",
    velocity_east_col
)

print(
    "Down  :",
    velocity_down_col
)


# ============================================================
# FIND GPS SPEED
# ============================================================

print()
print("=" * 70)
print("GPS SPEED")
print("=" * 70)


gps_speed_col = find_column(
    raw_df,
    [
        "GPS SPEED (Kmh)",
        "GPS_SPEED_Kmh"
    ],
    "GPS speed"
)


print(
    "GPS speed :",
    gps_speed_col
)


# ============================================================
# FIND NED ACCELERATION
# ============================================================

print()
print("=" * 70)
print("NED ACCELERATION")
print("=" * 70)


accel_north_col = find_column(
    ned_df,
    [
        "accel_north"
    ],
    "north acceleration"
)


accel_east_col = find_column(
    ned_df,
    [
        "accel_east"
    ],
    "east acceleration"
)


accel_down_col = find_column(
    ned_df,
    [
        "accel_down"
    ],
    "down acceleration"
)


print(
    "North :",
    accel_north_col
)

print(
    "East  :",
    accel_east_col
)

print(
    "Down  :",
    accel_down_col
)


# ============================================================
# EXTRACT VELOCITY
# ============================================================

vn = pd.to_numeric(
    velocity_df[velocity_north_col],
    errors="coerce"
).to_numpy()

ve = pd.to_numeric(
    velocity_df[velocity_east_col],
    errors="coerce"
).to_numpy()

vd = pd.to_numeric(
    velocity_df[velocity_down_col],
    errors="coerce"
).to_numpy()


# ============================================================
# EXTRACT ACCELERATION
# ============================================================

an = pd.to_numeric(
    ned_df[accel_north_col],
    errors="coerce"
).to_numpy()

ae = pd.to_numeric(
    ned_df[accel_east_col],
    errors="coerce"
).to_numpy()

ad = pd.to_numeric(
    ned_df[accel_down_col],
    errors="coerce"
).to_numpy()


# ============================================================
# EXTRACT GPS SPEED
# ============================================================

gps_speed_kmh = pd.to_numeric(
    raw_df[gps_speed_col],
    errors="coerce"
).to_numpy()


# ============================================================
# VALIDATION OF ARRAYS
# ============================================================

print()
print("=" * 70)
print("ARRAY VALIDATION")
print("=" * 70)


print(
    "Velocity North shape :",
    vn.shape
)

print(
    "Velocity East shape  :",
    ve.shape
)

print(
    "Velocity Down shape  :",
    vd.shape
)

print(
    "GPS speed shape      :",
    gps_speed_kmh.shape
)

print(
    "NED acceleration     :",
    an.shape
)


invalid_velocity = (
    ~np.isfinite(vn)
    |
    ~np.isfinite(ve)
    |
    ~np.isfinite(vd)
)

invalid_acceleration = (
    ~np.isfinite(an)
    |
    ~np.isfinite(ae)
    |
    ~np.isfinite(ad)
)

invalid_gps = (
    ~np.isfinite(gps_speed_kmh)
)


print(
    "Invalid velocity rows     :",
    np.sum(invalid_velocity)
)

print(
    "Invalid acceleration rows :",
    np.sum(invalid_acceleration)
)

print(
    "Invalid GPS rows          :",
    np.sum(invalid_gps)
)


# ============================================================
# SPEED CALCULATION
# ============================================================

print()
print("=" * 70)
print("CALCULATING SPEED")
print("=" * 70)


imu_speed_ms = np.sqrt(
    vn ** 2
    +
    ve ** 2
    +
    vd ** 2
)


imu_speed_kmh = (
    imu_speed_ms * 3.6
)


gps_speed_ms = (
    gps_speed_kmh / 3.6
)


# Keep GPS speed in km/h for reporting
gps_speed_kmh = gps_speed_kmh.copy()


# ============================================================
# VELOCITY ERROR
# ============================================================

velocity_error_ms = (
    imu_speed_ms
    -
    gps_speed_ms
)


absolute_error_ms = np.abs(
    velocity_error_ms
)


# ============================================================
# ACCELERATION MAGNITUDE
# ============================================================

accel_magnitude = np.sqrt(
    an ** 2
    +
    ae ** 2
    +
    ad ** 2
)


# ============================================================
# VELOCITY STATISTICS
# ============================================================

print()
print("=" * 70)
print("VELOCITY STATISTICS")
print("=" * 70)


print(
    f"GPS mean speed       : "
    f"{np.nanmean(gps_speed_ms):.4f} m/s"
)

print(
    f"IMU mean speed       : "
    f"{np.nanmean(imu_speed_ms):.4f} m/s"
)

print(
    f"GPS max speed        : "
    f"{np.nanmax(gps_speed_ms):.4f} m/s"
)

print(
    f"IMU max speed        : "
    f"{np.nanmax(imu_speed_ms):.4f} m/s"
)

print(
    f"Velocity MAE         : "
    f"{np.nanmean(absolute_error_ms):.4f} m/s"
)

print(
    f"Velocity RMSE        : "
    f"{np.sqrt(np.nanmean(velocity_error_ms ** 2)):.4f} m/s"
)


# ============================================================
# MAXIMUM DRIFT
# ============================================================

print()
print("=" * 70)
print("MAXIMUM DRIFT")
print("=" * 70)


max_error_index = np.nanargmax(
    absolute_error_ms
)


print(
    "Maximum velocity error :",
    f"{absolute_error_ms[max_error_index]:.4f} m/s"
)

print(
    "Maximum velocity error :",
    f"{absolute_error_ms[max_error_index] * 3.6:.4f} km/h"
)

print(
    "Row                    :",
    max_error_index
)

print(
    "Timestamp              :",
    velocity_timestamp.iloc[max_error_index]
)

print(
    "GPS speed              :",
    f"{gps_speed_kmh[max_error_index]:.4f} km/h"
)

print(
    "IMU speed              :",
    f"{imu_speed_kmh[max_error_index]:.4f} km/h"
)


# ============================================================
# FIRST MAJOR DRIFT
# ============================================================

print()
print("=" * 70)
print("FIRST MAJOR DRIFT")
print("=" * 70)


error_thresholds = [
    10,
    50,
    100,
    500
]


for threshold in error_thresholds:

    indices = np.where(
        absolute_error_ms > threshold
    )[0]

    if len(indices) > 0:

        first_index = indices[0]

        print()
        print(
            f"Error > {threshold:4d} m/s "
            f"first occurs at row {first_index}"
        )

        print(
            "  Time:",
            velocity_timestamp.iloc[first_index]
        )

        print(
            "  GPS :",
            f"{gps_speed_kmh[first_index]:.2f} km/h"
        )

        print(
            "  IMU :",
            f"{imu_speed_kmh[first_index]:.2f} km/h"
        )

        print(
            "  Error:",
            f"{absolute_error_ms[first_index]:.2f} m/s"
        )

    else:

        print(
            f"Error > {threshold:4d} m/s : never"
        )


# ============================================================
# TOP DRIFT POINTS
# ============================================================

print()
print("=" * 70)
print("TOP DRIFT POINTS")
print("=" * 70)


top_n = 20


top_indices = np.argsort(
    absolute_error_ms
)[-top_n:][::-1]


print(
    f"{'Rank':<6}"
    f"{'Row':<10}"
    f"{'Time':<25}"
    f"{'GPS km/h':<14}"
    f"{'IMU km/h':<14}"
    f"{'Error m/s':<14}"
)


for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank:<6}"
        f"{index:<10}"
        f"{str(velocity_timestamp.iloc[index]):<25}"
        f"{gps_speed_kmh[index]:<14.2f}"
        f"{imu_speed_kmh[index]:<14.2f}"
        f"{absolute_error_ms[index]:<14.2f}"
    )


# ============================================================
# NED ACCELERATION STATISTICS
# ============================================================

print()
print("=" * 70)
print("NED ACCELERATION STATISTICS")
print("=" * 70)


print(
    f"North mean : "
    f"{np.nanmean(an):.6f} m/s²"
)

print(
    f"East mean  : "
    f"{np.nanmean(ae):.6f} m/s²"
)

print(
    f"Down mean  : "
    f"{np.nanmean(ad):.6f} m/s²"
)


print(
    f"North std  : "
    f"{np.nanstd(an):.6f} m/s²"
)

print(
    f"East std   : "
    f"{np.nanstd(ae):.6f} m/s²"
)

print(
    f"Down std   : "
    f"{np.nanstd(ad):.6f} m/s²"
)


print(
    f"Acceleration magnitude mean : "
    f"{np.nanmean(accel_magnitude):.6f} m/s²"
)

print(
    f"Acceleration magnitude max  : "
    f"{np.nanmax(accel_magnitude):.6f} m/s²"
)


# ============================================================
# TIME AXIS
# ============================================================

timestamps_seconds = (
    velocity_timestamp
    -
    velocity_timestamp.iloc[0]
).dt.total_seconds().to_numpy()


time_hours = (
    timestamps_seconds / 3600.0
)


# ============================================================
# WORST DRIFT REGION
# ============================================================

print()
print("=" * 70)
print("WORST DRIFT REGION")
print("=" * 70)


window_seconds = 30.0


worst_time = timestamps_seconds[
    max_error_index
]


region_start_time = max(
    0,
    worst_time -
    window_seconds / 2
)


region_end_time = (
    worst_time +
    window_seconds / 2
)


region_mask = (
    (timestamps_seconds >= region_start_time)
    &
    (timestamps_seconds <= region_end_time)
)


region_indices = np.where(
    region_mask
)[0]


if len(region_indices) > 0:

    region_start = region_indices[0]

    region_end = region_indices[-1]


    print(
        "Region start:",
        velocity_timestamp.iloc[region_start]
    )

    print(
        "Region end  :",
        velocity_timestamp.iloc[region_end]
    )

    print(
        "Duration    :",
        f"{timestamps_seconds[region_end] - timestamps_seconds[region_start]:.2f} s"
    )

    print(
        "GPS mean    :",
        f"{np.nanmean(gps_speed_kmh[region_indices]):.2f} km/h"
    )

    print(
        "IMU mean    :",
        f"{np.nanmean(imu_speed_kmh[region_indices]):.2f} km/h"
    )

    print(
        "Error mean  :",
        f"{np.nanmean(absolute_error_ms[region_indices]):.2f} m/s"
    )

    print(
        "Accel mean  :",
        f"{np.nanmean(accel_magnitude[region_indices]):.4f} m/s²"
    )

    print(
        "Accel max   :",
        f"{np.nanmax(accel_magnitude[region_indices]):.4f} m/s²"
    )


# ============================================================
# SAVE DIAGNOSTIC DATA
# ============================================================

diagnostic_df = pd.DataFrame({

    "timestamp":
        velocity_timestamp,

    "gps_speed_kmh":
        gps_speed_kmh,

    "gps_speed_ms":
        gps_speed_ms,

    "imu_velocity_north":
        vn,

    "imu_velocity_east":
        ve,

    "imu_velocity_down":
        vd,

    "imu_speed_ms":
        imu_speed_ms,

    "imu_speed_kmh":
        imu_speed_kmh,

    "velocity_error_ms":
        velocity_error_ms,

    "absolute_velocity_error_ms":
        absolute_error_ms,

    "accel_north":
        an,

    "accel_east":
        ae,

    "accel_down":
        ad,

    "accel_magnitude":
        accel_magnitude
})


diagnostic_file = (
    "data/processed/"
    "velocity_drift_diagnosis.csv"
)


diagnostic_df.to_csv(
    diagnostic_file,
    index=False
)


print()
print(
    "Diagnostic data saved to:"
)

print(
    diagnostic_file
)


# ============================================================
# PLOT 1
# GPS VS IMU SPEED
# ============================================================

plt.figure(
    figsize=(14, 6)
)


plt.plot(
    time_hours,
    gps_speed_kmh,
    label="GPS Speed"
)


plt.plot(
    time_hours,
    imu_speed_kmh,
    label="IMU Speed",
    alpha=0.7
)


plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "Speed (km/h)"
)


plt.title(
    "GPS Speed vs Raw IMU Integrated Speed"
)


plt.legend()

plt.grid(
    alpha=0.3
)


plt.tight_layout()


plot1 = (
    OUTPUT_DIR +
    "/velocity_drift_overview.png"
)


plt.savefig(
    plot1,
    dpi=150
)


plt.close()


# ============================================================
# PLOT 2
# VELOCITY ERROR
# ============================================================

plt.figure(
    figsize=(14, 6)
)


plt.plot(
    time_hours,
    absolute_error_ms
)


plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "Absolute Velocity Error (m/s)"
)


plt.title(
    "IMU Velocity Drift"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plot2 = (
    OUTPUT_DIR +
    "/velocity_error_over_time.png"
)


plt.savefig(
    plot2,
    dpi=150
)


plt.close()


# ============================================================
# PLOT 3
# NED ACCELERATION
# ============================================================

plt.figure(
    figsize=(14, 7)
)


plt.plot(
    time_hours,
    an,
    label="North"
)

plt.plot(
    time_hours,
    ae,
    label="East"
)

plt.plot(
    time_hours,
    ad,
    label="Down"
)


plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "Acceleration (m/s²)"
)


plt.title(
    "NED Acceleration"
)


plt.legend()

plt.grid(
    alpha=0.3
)


plt.tight_layout()


plot3 = (
    OUTPUT_DIR +
    "/ned_acceleration_over_time.png"
)


plt.savefig(
    plot3,
    dpi=150
)


plt.close()


# ============================================================
# PLOT 4
# WORST DRIFT REGION
# ============================================================

if len(region_indices) > 0:

    region_time = (
        timestamps_seconds[
            region_indices
        ]
        -
        timestamps_seconds[
            region_indices[0]
        ]
    )


    plt.figure(
        figsize=(14, 8)
    )


    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    plt.subplot(
        2,
        1,
        1
    )


    plt.plot(
        region_time,
        gps_speed_kmh[
            region_indices
        ],
        label="GPS Speed"
    )


    plt.plot(
        region_time,
        imu_speed_kmh[
            region_indices
        ],
        label="IMU Speed"
    )


    plt.ylabel(
        "Speed (km/h)"
    )


    plt.title(
        "Worst Velocity Drift Region"
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )


    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    plt.subplot(
        2,
        1,
        2
    )


    plt.plot(
        region_time,
        an[
            region_indices
        ],
        label="North"
    )


    plt.plot(
        region_time,
        ae[
            region_indices
        ],
        label="East"
    )


    plt.plot(
        region_time,
        ad[
            region_indices
        ],
        label="Down"
    )


    plt.xlabel(
        "Time from region start (s)"
    )


    plt.ylabel(
        "Acceleration (m/s²)"
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plot4 = (
        OUTPUT_DIR +
        "/worst_velocity_drift_region.png"
    )


    plt.savefig(
        plot4,
        dpi=150
    )


    plt.close()


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("STEP 6.1 COMPLETE")
print("=" * 70)


print()
print("Generated files:")

print(
    "1.",
    diagnostic_file
)

print(
    "2.",
    plot1
)

print(
    "3.",
    plot2
)

print(
    "4.",
    plot3
)


if len(region_indices) > 0:

    print(
        "5.",
        plot4
    )


print()
print("Next:")
print(
    "Analyze the first major velocity drift before modifying "
    "the integration algorithm."
)

print()
print("=" * 70)
print("DO NOT START POSITION INTEGRATION YET")
print("=" * 70)