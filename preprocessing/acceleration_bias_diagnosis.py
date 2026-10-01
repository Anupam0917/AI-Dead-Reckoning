import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 6.2
# ACCELERATION BIAS DIAGNOSIS
# ============================================================

print("=" * 70)
print("STEP 6.2")
print("ACCELERATION BIAS DIAGNOSIS")
print("=" * 70)

print("""
Pipeline:

NED acceleration
      ↓
stationary detection
      ↓
stationary acceleration statistics
      ↓
bias estimation
      ↓
moving vs stationary comparison
      ↓
segment analysis
      ↓
diagnostic plots
""")


# ============================================================
# PATHS
# ============================================================

NED_FILE = (
    "data/processed/"
    "ned_acceleration.csv"
)

VELOCITY_FILE = (
    "data/processed/"
    "imu_velocity.csv"
)

RAW_FILE = (
    "data/raw/"
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

OUTPUT_DIR = "outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HELPER
# ============================================================

def find_column(
    df,
    candidates,
    description
):

    # Exact match
    for candidate in candidates:

        if candidate in df.columns:

            return candidate

    # Partial match
    lower_columns = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        candidate_lower = (
            candidate.lower()
        )

        for (
            lower_column,
            original_column
        ) in lower_columns.items():

            if candidate_lower in lower_column:

                return original_column

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# TIMESTAMP
# ============================================================

def prepare_timestamp(df):

    if "timestamp" in df.columns:

        print(
            "Using existing timestamp column"
        )

        return pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    known_columns = [
        "DATE (YYYY-MO-DD HH-MI-SS_SSS)",
        "DATE_YYYY-MO-DD_HH-MI-SS_SSS"
    ]

    for column in known_columns:

        if column in df.columns:

            print(
                "Using date column:",
                column
            )

            return pd.to_datetime(
                df[column],
                format="%Y-%m-%d %H:%M:%S:%f",
                errors="coerce"
            )

    for column in df.columns:

        name = str(column).upper()

        if "DATE" in name:

            print(
                "Using detected date column:",
                column
            )

            return pd.to_datetime(
                df[column],
                format="%Y-%m-%d %H:%M:%S:%f",
                errors="coerce"
            )

    raise ValueError(
        "Could not find timestamp column."
    )


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)


print("\nLoading NED acceleration...")

ned_df = pd.read_csv(
    NED_FILE
)

print(
    "Rows:",
    len(ned_df)
)


print("\nLoading velocity...")

velocity_df = pd.read_csv(
    VELOCITY_FILE
)

print(
    "Rows:",
    len(velocity_df)
)


print("\nLoading raw dataset...")

raw_df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

print(
    "Rows:",
    len(raw_df)
)


# ============================================================
# ALIGNMENT
# ============================================================

print()
print("=" * 70)
print("DATA ALIGNMENT")
print("=" * 70)


if not (
    len(ned_df)
    ==
    len(velocity_df)
    ==
    len(raw_df)
):

    raise ValueError(
        "Dataset lengths do not match."
    )


print(
    "Rows:",
    len(ned_df)
)

print(
    "Alignment: OK"
)


# ============================================================
# TIMESTAMP
# ============================================================

print()
print("=" * 70)
print("TIMESTAMP")
print("=" * 70)


timestamp = prepare_timestamp(
    ned_df
)


print(
    "Invalid:",
    timestamp.isna().sum()
)


time_seconds = (
    timestamp -
    timestamp.iloc[0]
).dt.total_seconds().to_numpy()


# ============================================================
# NED ACCELERATION
# ============================================================

north_col = find_column(
    ned_df,
    ["accel_north"],
    "north acceleration"
)

east_col = find_column(
    ned_df,
    ["accel_east"],
    "east acceleration"
)

down_col = find_column(
    ned_df,
    ["accel_down"],
    "down acceleration"
)


north = pd.to_numeric(
    ned_df[north_col],
    errors="coerce"
).to_numpy()

east = pd.to_numeric(
    ned_df[east_col],
    errors="coerce"
).to_numpy()

down = pd.to_numeric(
    ned_df[down_col],
    errors="coerce"
).to_numpy()


# ============================================================
# GPS SPEED
# ============================================================

gps_speed_col = find_column(
    raw_df,
    [
        "GPS SPEED (Kmh)",
        "GPS_SPEED_Kmh"
    ],
    "GPS speed"
)


gps_speed = pd.to_numeric(
    raw_df[gps_speed_col],
    errors="coerce"
).to_numpy()


# ============================================================
# ACCELERATION MAGNITUDE
# ============================================================

accel_magnitude = np.sqrt(
    north ** 2
    +
    east ** 2
    +
    down ** 2
)


# ============================================================
# STATIONARY DETECTION
# ============================================================

print()
print("=" * 70)
print("STATIONARY DETECTION")
print("=" * 70)


# GPS stationary threshold
GPS_STATIONARY_KMH = 0.5

# NED acceleration threshold
ACCEL_STATIONARY_MS2 = 1.5


gps_stationary = (
    gps_speed <=
    GPS_STATIONARY_KMH
)


accel_stationary = (
    accel_magnitude <=
    ACCEL_STATIONARY_MS2
)


stationary = (
    gps_stationary
    &
    accel_stationary
)


moving = ~stationary


print(
    "GPS stationary:",
    np.sum(gps_stationary)
)

print(
    "Low acceleration:",
    np.sum(accel_stationary)
)

print(
    "Combined stationary:",
    np.sum(stationary)
)

print(
    "Stationary percentage:",
    f"{100 * np.mean(stationary):.2f}%"
)


# ============================================================
# STATIONARY ACCELERATION STATISTICS
# ============================================================

print()
print("=" * 70)
print("STATIONARY ACCELERATION")
print("=" * 70)


stationary_north = north[
    stationary
]

stationary_east = east[
    stationary
]

stationary_down = down[
    stationary
]

stationary_magnitude = (
    accel_magnitude[
        stationary
    ]
)


print(
    f"North mean : "
    f"{np.mean(stationary_north):.6f} m/s²"
)

print(
    f"East mean  : "
    f"{np.mean(stationary_east):.6f} m/s²"
)

print(
    f"Down mean  : "
    f"{np.mean(stationary_down):.6f} m/s²"
)


print(
    f"North std  : "
    f"{np.std(stationary_north):.6f} m/s²"
)

print(
    f"East std   : "
    f"{np.std(stationary_east):.6f} m/s²"
)

print(
    f"Down std   : "
    f"{np.std(stationary_down):.6f} m/s²"
)


print(
    f"Mean magnitude : "
    f"{np.mean(stationary_magnitude):.6f} m/s²"
)

print(
    f"Max magnitude  : "
    f"{np.max(stationary_magnitude):.6f} m/s²"
)


# ============================================================
# ESTIMATE BIAS
# ============================================================

bias_north = np.mean(
    stationary_north
)

bias_east = np.mean(
    stationary_east
)

bias_down = np.mean(
    stationary_down
)


bias_vector = np.array([
    bias_north,
    bias_east,
    bias_down
])


bias_magnitude = np.linalg.norm(
    bias_vector
)


print()
print("=" * 70)
print("ESTIMATED NED ACCELERATION BIAS")
print("=" * 70)


print(
    f"North bias : "
    f"{bias_north:.8f} m/s²"
)

print(
    f"East bias  : "
    f"{bias_east:.8f} m/s²"
)

print(
    f"Down bias  : "
    f"{bias_down:.8f} m/s²"
)

print(
    f"Bias magnitude : "
    f"{bias_magnitude:.8f} m/s²"
)


# ============================================================
# BIAS CORRECTED ACCELERATION
# ============================================================

north_corrected = (
    north -
    bias_north
)

east_corrected = (
    east -
    bias_east
)

down_corrected = (
    down -
    bias_down
)


corrected_magnitude = np.sqrt(
    north_corrected ** 2
    +
    east_corrected ** 2
    +
    down_corrected ** 2
)


# ============================================================
# BEFORE / AFTER STATIONARY
# ============================================================

print()
print("=" * 70)
print("BIAS CORRECTION VALIDATION")
print("=" * 70)


print(
    "BEFORE"
)

print(
    "North:",
    f"{np.mean(stationary_north):.8f}"
)

print(
    "East :",
    f"{np.mean(stationary_east):.8f}"
)

print(
    "Down :",
    f"{np.mean(stationary_down):.8f}"
)


print()
print(
    "AFTER"
)


print(
    "North:",
    f"{np.mean(north_corrected[stationary]):.8f}"
)

print(
    "East :",
    f"{np.mean(east_corrected[stationary]):.8f}"
)

print(
    "Down :",
    f"{np.mean(down_corrected[stationary]):.8f}"
)


# ============================================================
# MOVING VS STATIONARY
# ============================================================

print()
print("=" * 70)
print("MOVING VS STATIONARY")
print("=" * 70)


print(
    "Stationary acceleration:"
)

print(
    "Mean:",
    f"{np.mean(stationary_magnitude):.6f}"
)

print(
    "Median:",
    f"{np.median(stationary_magnitude):.6f}"
)

print(
    "Maximum:",
    f"{np.max(stationary_magnitude):.6f}"
)


moving_magnitude = (
    accel_magnitude[
        moving
    ]
)


print()
print(
    "Moving acceleration:"
)

print(
    "Mean:",
    f"{np.mean(moving_magnitude):.6f}"
)

print(
    "Median:",
    f"{np.median(moving_magnitude):.6f}"
)

print(
    "Maximum:",
    f"{np.max(moving_magnitude):.6f}"
)


# ============================================================
# TIME-SEGMENT ANALYSIS
# ============================================================

print()
print("=" * 70)
print("TIME SEGMENT ANALYSIS")
print("=" * 70)


segment_duration = 600.0

total_duration = (
    time_seconds[-1]
)


number_of_segments = int(
    np.ceil(
        total_duration /
        segment_duration
    )
)


segment_rows = []


for segment in range(
    number_of_segments
):

    start_time = (
        segment *
        segment_duration
    )

    end_time = (
        start_time +
        segment_duration
    )


    mask = (
        (time_seconds >= start_time)
        &
        (time_seconds < end_time)
    )


    if not np.any(mask):

        continue


    stationary_mask = (
        mask &
        stationary
    )


    segment_rows.append({

        "segment":
            segment + 1,

        "start_seconds":
            start_time,

        "end_seconds":
            end_time,

        "north_mean":
            np.mean(north[mask]),

        "east_mean":
            np.mean(east[mask]),

        "down_mean":
            np.mean(down[mask]),

        "north_stationary_mean":
            (
                np.mean(
                    north[stationary_mask]
                )
                if np.any(stationary_mask)
                else np.nan
            ),

        "east_stationary_mean":
            (
                np.mean(
                    east[stationary_mask]
                )
                if np.any(stationary_mask)
                else np.nan
            ),

        "down_stationary_mean":
            (
                np.mean(
                    down[stationary_mask]
                )
                if np.any(stationary_mask)
                else np.nan
            ),

        "stationary_samples":
            np.sum(stationary_mask),

        "total_samples":
            np.sum(mask)

    })


segment_df = pd.DataFrame(
    segment_rows
)


print(
    segment_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE SEGMENT ANALYSIS
# ============================================================

segment_file = (
    "data/processed/"
    "acceleration_bias_segments.csv"
)


segment_df.to_csv(
    segment_file,
    index=False
)


# ============================================================
# SAVE BIAS CORRECTED DATA
# ============================================================

bias_df = pd.DataFrame({

    "timestamp":
        timestamp,

    "accel_north":
        north,

    "accel_east":
        east,

    "accel_down":
        down,

    "accel_north_bias_corrected":
        north_corrected,

    "accel_east_bias_corrected":
        east_corrected,

    "accel_down_bias_corrected":
        down_corrected,

    "accel_magnitude":
        accel_magnitude,

    "accel_magnitude_bias_corrected":
        corrected_magnitude,

    "gps_speed_kmh":
        gps_speed,

    "stationary":
        stationary

})


bias_file = (
    "data/processed/"
    "ned_acceleration_bias_corrected.csv"
)


bias_df.to_csv(
    bias_file,
    index=False
)


# ============================================================
# PLOT 1
# STATIONARY ACCELERATION
# ============================================================

print()
print("=" * 70)
print("GENERATING PLOTS")
print("=" * 70)


time_hours = (
    time_seconds /
    3600.0
)


plt.figure(
    figsize=(14, 7)
)


plt.plot(
    time_hours,
    north,
    label="North"
)

plt.plot(
    time_hours,
    east,
    label="East"
)

plt.plot(
    time_hours,
    down,
    label="Down"
)


plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "Acceleration (m/s²)"
)


plt.title(
    "NED Acceleration Over Time"
)


plt.legend()

plt.grid(
    alpha=0.3
)


plt.tight_layout()


plot1 = (
    OUTPUT_DIR +
    "/acceleration_bias_overview.png"
)


plt.savefig(
    plot1,
    dpi=150
)

plt.close()


# ============================================================
# PLOT 2
# STATIONARY SAMPLES
# ============================================================

plt.figure(
    figsize=(14, 7)
)


plt.plot(
    time_hours,
    north,
    label="North",
    alpha=0.5
)

plt.plot(
    time_hours,
    east,
    label="East",
    alpha=0.5
)

plt.plot(
    time_hours,
    down,
    label="Down",
    alpha=0.5
)


stationary_times = (
    time_hours[
        stationary
    ]
)


plt.scatter(
    stationary_times,
    north[
        stationary
    ],
    s=2,
    label="Stationary"
)


plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "Acceleration (m/s²)"
)


plt.title(
    "Stationary Samples Used for Bias Estimation"
)


plt.legend()

plt.grid(
    alpha=0.3
)


plt.tight_layout()


plot2 = (
    OUTPUT_DIR +
    "/stationary_acceleration_bias.png"
)


plt.savefig(
    plot2,
    dpi=150
)

plt.close()


# ============================================================
# PLOT 3
# BEFORE VS AFTER BIAS
# ============================================================

plt.figure(
    figsize=(14, 7)
)


plt.plot(
    time_hours,
    north,
    label="North before"
)

plt.plot(
    time_hours,
    north_corrected,
    label="North after",
    alpha=0.7
)


plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "North acceleration (m/s²)"
)


plt.title(
    "North Acceleration Bias Correction"
)


plt.legend()

plt.grid(
    alpha=0.3
)


plt.tight_layout()


plot3 = (
    OUTPUT_DIR +
    "/north_acceleration_bias_correction.png"
)


plt.savefig(
    plot3,
    dpi=150
)

plt.close()


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("STEP 6.2 COMPLETE")
print("=" * 70)


print()
print("Estimated bias:")

print(
    f"North : {bias_north:.8f} m/s²"
)

print(
    f"East  : {bias_east:.8f} m/s²"
)

print(
    f"Down  : {bias_down:.8f} m/s²"
)


print()
print(
    "Saved:"
)

print(
    bias_file
)

print(
    segment_file
)

print(
    plot1
)

print(
    plot2
)

print(
    plot3
)


print()
print(
    "IMPORTANT:"
)

print(
    "This bias-corrected acceleration is diagnostic only."
)

print(
    "Do NOT replace the original NED acceleration yet."
)

print(
    "Do NOT start position integration yet."
)

print()
print("=" * 70)