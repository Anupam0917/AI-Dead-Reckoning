import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 6.3
# STRICT STATIONARY DETECTION
# ============================================================

print("=" * 70)
print("STEP 6.3")
print("STRICT STATIONARY DETECTION")
print("=" * 70)

print("""
Pipeline:

Raw gyro
    +
NED acceleration
    +
GPS speed
    ↓
sensor magnitude calculation
    ↓
strict stationary conditions
    ↓
continuous-window filtering
    ↓
high-confidence stationary windows
    ↓
bias estimation
    ↓
bias stability analysis
""")


# ============================================================
# PATHS
# ============================================================

NED_FILE = (
    "data/processed/"
    "ned_acceleration.csv"
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
# PARAMETERS
# ============================================================

# GPS speed threshold
GPS_SPEED_THRESHOLD_KMH = 0.5

# Gyroscope threshold
# rad/s
GYRO_THRESHOLD_RAD_S = 0.10

# NED linear acceleration threshold
# m/s²
ACCEL_THRESHOLD_MS2 = 0.50

# Minimum continuous stationary duration
# seconds
MIN_STATIONARY_DURATION = 3.0

# Maximum allowed gap inside a stationary run
MAX_GAP_SECONDS = 0.20


# ============================================================
# HELPER FUNCTION
# ============================================================

def find_column(
    df,
    candidates,
    description
):

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
        f"\nCould not find {description} column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# TIMESTAMP FUNCTION
# ============================================================

def prepare_timestamp(df):

    # --------------------------------------------------------
    # Existing timestamp
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        print(
            "Using existing timestamp column"
        )

        return pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Known raw date columns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Flexible search
    # --------------------------------------------------------

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
# NED
# ------------------------------------------------------------

print()
print("Loading NED acceleration...")

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
# RAW
# ------------------------------------------------------------

print()
print("Loading raw sensor data...")

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


if len(ned_df) != len(raw_df):

    raise ValueError(
        "NED and raw dataset row counts do not match."
    )


print(
    "NED rows :",
    len(ned_df)
)

print(
    "Raw rows :",
    len(raw_df)
)

print(
    "Alignment : OK"
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


invalid_timestamp = (
    timestamp.isna()
)


print(
    "Invalid timestamps:",
    invalid_timestamp.sum()
)


if invalid_timestamp.any():

    raise ValueError(
        "Invalid timestamps detected."
    )


# Convert timestamp to seconds
time_seconds = (
    timestamp -
    timestamp.iloc[0]
).dt.total_seconds().to_numpy()


# ============================================================
# DT
# ============================================================

dt = np.diff(
    time_seconds,
    prepend=time_seconds[0]
)


# First sample
if len(dt) > 1:

    dt[0] = np.median(
        dt[1:]
    )

else:

    dt[0] = 0.1


print(
    "Median dt:",
    f"{np.median(dt):.4f} s"
)

print(
    "Minimum dt:",
    f"{np.min(dt):.4f} s"
)

print(
    "Maximum dt:",
    f"{np.max(dt):.4f} s"
)


# ============================================================
# FIND NED ACCELERATION
# ============================================================

print()
print("=" * 70)
print("NED ACCELERATION")
print("=" * 70)


north_col = find_column(
    ned_df,
    [
        "accel_north"
    ],
    "north acceleration"
)

east_col = find_column(
    ned_df,
    [
        "accel_east"
    ],
    "east acceleration"
)

down_col = find_column(
    ned_df,
    [
        "accel_down"
    ],
    "down acceleration"
)


print(
    "North :",
    north_col
)

print(
    "East  :",
    east_col
)

print(
    "Down  :",
    down_col
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
# FIND GYROSCOPE
# ============================================================

print()
print("=" * 70)
print("GYROSCOPE")
print("=" * 70)


gyro_x_col = find_column(
    raw_df,
    [
        "GYROSCOPE_Roll_rad_s",
        "GYROSCOPE Roll (rad/s)"
    ],
    "gyro X / roll"
)

gyro_y_col = find_column(
    raw_df,
    [
        "GYROSCOPE_Pitch_rad_s",
        "GYROSCOPE Pitch (rad/s)"
    ],
    "gyro Y / pitch"
)

gyro_z_col = find_column(
    raw_df,
    [
        "GYROSCOPE_Yaw_rad_s",
        "GYROSCOPE Yaw (rad/s)"
    ],
    "gyro Z / yaw"
)


print(
    "Gyro X :",
    gyro_x_col
)

print(
    "Gyro Y :",
    gyro_y_col
)

print(
    "Gyro Z :",
    gyro_z_col
)


gyro_x = pd.to_numeric(
    raw_df[gyro_x_col],
    errors="coerce"
).to_numpy()

gyro_y = pd.to_numeric(
    raw_df[gyro_y_col],
    errors="coerce"
).to_numpy()

gyro_z = pd.to_numeric(
    raw_df[gyro_z_col],
    errors="coerce"
).to_numpy()


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


gps_speed = pd.to_numeric(
    raw_df[gps_speed_col],
    errors="coerce"
).to_numpy()


# ============================================================
# SENSOR MAGNITUDES
# ============================================================

print()
print("=" * 70)
print("SENSOR MAGNITUDES")
print("=" * 70)


gyro_magnitude = np.sqrt(
    gyro_x ** 2
    +
    gyro_y ** 2
    +
    gyro_z ** 2
)


accel_magnitude = np.sqrt(
    north ** 2
    +
    east ** 2
    +
    down ** 2
)


print(
    "Gyro magnitude mean:",
    f"{np.nanmean(gyro_magnitude):.6f} rad/s"
)

print(
    "Gyro magnitude median:",
    f"{np.nanmedian(gyro_magnitude):.6f} rad/s"
)

print(
    "Gyro magnitude max:",
    f"{np.nanmax(gyro_magnitude):.6f} rad/s"
)


print(
    "Acceleration magnitude mean:",
    f"{np.nanmean(accel_magnitude):.6f} m/s²"
)

print(
    "Acceleration magnitude median:",
    f"{np.nanmedian(accel_magnitude):.6f} m/s²"
)

print(
    "Acceleration magnitude max:",
    f"{np.nanmax(accel_magnitude):.6f} m/s²"
)


# ============================================================
# STRICT CONDITIONS
# ============================================================

print()
print("=" * 70)
print("STRICT STATIONARY CONDITIONS")
print("=" * 70)


gps_condition = (
    gps_speed <=
    GPS_SPEED_THRESHOLD_KMH
)


gyro_condition = (
    gyro_magnitude <=
    GYRO_THRESHOLD_RAD_S
)


accel_condition = (
    accel_magnitude <=
    ACCEL_THRESHOLD_MS2
)


print(
    "GPS threshold     :",
    f"<= {GPS_SPEED_THRESHOLD_KMH} km/h"
)

print(
    "Gyro threshold    :",
    f"<= {GYRO_THRESHOLD_RAD_S} rad/s"
)

print(
    "Acceleration      :",
    f"<= {ACCEL_THRESHOLD_MS2} m/s²"
)


print()
print(
    "GPS condition samples :",
    np.sum(gps_condition)
)

print(
    "Gyro condition samples:",
    np.sum(gyro_condition)
)

print(
    "Accel condition       :",
    np.sum(accel_condition)
)


# ============================================================
# COMBINED CONDITION
# ============================================================

candidate_stationary = (
    gps_condition
    &
    gyro_condition
    &
    accel_condition
)


print()
print(
    "Combined candidate samples:",
    np.sum(candidate_stationary)
)


# ============================================================
# CONTINUOUS WINDOW DETECTION
# ============================================================

print()
print("=" * 70)
print("CONTINUOUS STATIONARY WINDOWS")
print("=" * 70)


# ------------------------------------------------------------
# Find runs of True values
# ------------------------------------------------------------

candidate = (
    candidate_stationary
)


change = np.diff(
    candidate.astype(np.int8)
)


start_indices = (
    np.where(change == 1)[0]
    + 1
)

end_indices = (
    np.where(change == -1)[0]
    + 1
)


# Handle first sample
if candidate[0]:

    start_indices = np.insert(
        start_indices,
        0,
        0
    )


# Handle last sample
if candidate[-1]:

    end_indices = np.append(
        end_indices,
        len(candidate)
    )


stationary_windows = []


for start, end in zip(
    start_indices,
    end_indices
):

    if end <= start:

        continue


    duration = (
        time_seconds[end - 1]
        -
        time_seconds[start]
    )


    # --------------------------------------------------------
    # Require minimum duration
    # --------------------------------------------------------

    if duration < MIN_STATIONARY_DURATION:

        continue


    stationary_windows.append({

        "start_index":
            start,

        "end_index":
            end - 1,

        "start_time_s":
            time_seconds[start],

        "end_time_s":
            time_seconds[end - 1],

        "duration_s":
            duration,

        "samples":
            end - start

    })


print(
    "Minimum duration:",
    f"{MIN_STATIONARY_DURATION:.1f} s"
)


print(
    "Detected windows:",
    len(stationary_windows)
)


# ============================================================
# PRINT WINDOWS
# ============================================================

print()

if len(stationary_windows) == 0:

    print(
        "NO HIGH-CONFIDENCE STATIONARY WINDOWS FOUND."
    )

else:

    print(
        f"{'Window':<8}"
        f"{'Start(s)':<14}"
        f"{'End(s)':<14}"
        f"{'Duration(s)':<14}"
        f"{'Samples':<10}"
    )

    print(
        "-" * 60
    )


    for number, window in enumerate(
        stationary_windows,
        start=1
    ):

        print(
            f"{number:<8}"
            f"{window['start_time_s']:<14.2f}"
            f"{window['end_time_s']:<14.2f}"
            f"{window['duration_s']:<14.2f}"
            f"{window['samples']:<10}"
        )


# ============================================================
# BUILD FINAL STATIONARY MASK
# ============================================================

strict_stationary = np.zeros(
    len(candidate_stationary),
    dtype=bool
)


for window in stationary_windows:

    start = window[
        "start_index"
    ]

    end = window[
        "end_index"
    ]


    strict_stationary[
        start:end + 1
    ] = True


print()
print(
    "Strict stationary samples:",
    np.sum(strict_stationary)
)

print(
    "Strict stationary percentage:",
    f"{100 * np.mean(strict_stationary):.2f}%"
)


# ============================================================
# BIAS ANALYSIS PER WINDOW
# ============================================================

print()
print("=" * 70)
print("STATIONARY BIAS PER WINDOW")
print("=" * 70)


bias_rows = []


for number, window in enumerate(
    stationary_windows,
    start=1
):

    start = window[
        "start_index"
    ]

    end = window[
        "end_index"
    ] + 1


    n = north[start:end]
    e = east[start:end]
    d = down[start:end]


    gx = gyro_x[start:end]
    gy = gyro_y[start:end]
    gz = gyro_z[start:end]


    bias_rows.append({

        "window":
            number,

        "start_time_s":
            window["start_time_s"],

        "end_time_s":
            window["end_time_s"],

        "duration_s":
            window["duration_s"],

        "samples":
            window["samples"],

        "north_bias":
            np.mean(n),

        "east_bias":
            np.mean(e),

        "down_bias":
            np.mean(d),

        "north_std":
            np.std(n),

        "east_std":
            np.std(e),

        "down_std":
            np.std(d),

        "gyro_x_mean":
            np.mean(gx),

        "gyro_y_mean":
            np.mean(gy),

        "gyro_z_mean":
            np.mean(gz),

        "gyro_magnitude_mean":
            np.mean(
                gyro_magnitude[start:end]
            ),

        "accel_magnitude_mean":
            np.mean(
                accel_magnitude[start:end]
            )

    })


bias_df = pd.DataFrame(
    bias_rows
)


if len(bias_df) > 0:

    print(
        bias_df.to_string(
            index=False
        )
    )

else:

    print(
        "No valid stationary windows."
    )


# ============================================================
# GLOBAL STRICT BIAS
# ============================================================

print()
print("=" * 70)
print("GLOBAL STRICT STATIONARY BIAS")
print("=" * 70)


if np.any(strict_stationary):

    strict_north = north[
        strict_stationary
    ]

    strict_east = east[
        strict_stationary
    ]

    strict_down = down[
        strict_stationary
    ]


    strict_north_bias = np.mean(
        strict_north
    )

    strict_east_bias = np.mean(
        strict_east
    )

    strict_down_bias = np.mean(
        strict_down
    )


    strict_bias_vector = np.array([
        strict_north_bias,
        strict_east_bias,
        strict_down_bias
    ])


    strict_bias_magnitude = np.linalg.norm(
        strict_bias_vector
    )


    print(
        f"North bias : "
        f"{strict_north_bias:.8f} m/s²"
    )

    print(
        f"East bias  : "
        f"{strict_east_bias:.8f} m/s²"
    )

    print(
        f"Down bias  : "
        f"{strict_down_bias:.8f} m/s²"
    )

    print(
        f"Bias magnitude : "
        f"{strict_bias_magnitude:.8f} m/s²"
    )


else:

    strict_north_bias = np.nan
    strict_east_bias = np.nan
    strict_down_bias = np.nan
    strict_bias_magnitude = np.nan


# ============================================================
# BIAS STABILITY
# ============================================================

print()
print("=" * 70)
print("BIAS STABILITY")
print("=" * 70)


if len(bias_df) >= 2:

    print(
        "North window-to-window std:",
        f"{bias_df['north_bias'].std():.8f} m/s²"
    )

    print(
        "East window-to-window std :",
        f"{bias_df['east_bias'].std():.8f} m/s²"
    )

    print(
        "Down window-to-window std :",
        f"{bias_df['down_bias'].std():.8f} m/s²"
    )

    print()

    print(
        "North bias range:",
        f"{bias_df['north_bias'].min():.6f}",
        "to",
        f"{bias_df['north_bias'].max():.6f}"
    )

    print(
        "East bias range :",
        f"{bias_df['east_bias'].min():.6f}",
        "to",
        f"{bias_df['east_bias'].max():.6f}"
    )

    print(
        "Down bias range :",
        f"{bias_df['down_bias'].min():.6f}",
        "to",
        f"{bias_df['down_bias'].max():.6f}"
    )

else:

    print(
        "Not enough windows for stability analysis."
    )


# ============================================================
# SAVE STATIONARY WINDOW DATA
# ============================================================

window_df = pd.DataFrame(
    stationary_windows
)


window_file = (
    "data/processed/"
    "strict_stationary_windows.csv"
)


window_df.to_csv(
    window_file,
    index=False
)


# ============================================================
# SAVE BIAS DATA
# ============================================================

bias_file = (
    "data/processed/"
    "strict_stationary_bias.csv"
)


bias_df.to_csv(
    bias_file,
    index=False
)


# ============================================================
# SAVE SAMPLE-LEVEL DATA
# ============================================================

sample_df = pd.DataFrame({

    "timestamp":
        timestamp,

    "time_seconds":
        time_seconds,

    "gps_speed_kmh":
        gps_speed,

    "gyro_x":
        gyro_x,

    "gyro_y":
        gyro_y,

    "gyro_z":
        gyro_z,

    "gyro_magnitude":
        gyro_magnitude,

    "accel_north":
        north,

    "accel_east":
        east,

    "accel_down":
        down,

    "accel_magnitude":
        accel_magnitude,

    "gps_stationary":
        gps_condition,

    "gyro_stationary":
        gyro_condition,

    "accel_stationary":
        accel_condition,

    "candidate_stationary":
        candidate_stationary,

    "strict_stationary":
        strict_stationary

})


sample_file = (
    "data/processed/"
    "strict_stationary_samples.csv"
)


sample_df.to_csv(
    sample_file,
    index=False
)


# ============================================================
# PLOT 1
# SENSOR CONDITIONS
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
    figsize=(14, 8)
)


plt.subplot(
    3,
    1,
    1
)

plt.plot(
    time_hours,
    gps_speed
)

plt.axhline(
    GPS_SPEED_THRESHOLD_KMH,
    linestyle="--"
)

plt.ylabel(
    "GPS Speed\n(km/h)"
)

plt.grid(
    alpha=0.3
)


plt.subplot(
    3,
    1,
    2
)

plt.plot(
    time_hours,
    gyro_magnitude
)

plt.axhline(
    GYRO_THRESHOLD_RAD_S,
    linestyle="--"
)

plt.ylabel(
    "Gyro\n(rad/s)"
)

plt.grid(
    alpha=0.3
)


plt.subplot(
    3,
    1,
    3
)

plt.plot(
    time_hours,
    accel_magnitude
)

plt.axhline(
    ACCEL_THRESHOLD_MS2,
    linestyle="--"
)

plt.ylabel(
    "Acceleration\n(m/s²)"
)

plt.xlabel(
    "Time (hours)"
)

plt.grid(
    alpha=0.3
)


plt.suptitle(
    "Strict Stationary Detection Conditions"
)

plt.tight_layout()


plot1 = (
    OUTPUT_DIR +
    "/strict_stationary_conditions.png"
)


plt.savefig(
    plot1,
    dpi=150
)

plt.close()


# ============================================================
# PLOT 2
# STATIONARY WINDOWS
# ============================================================

plt.figure(
    figsize=(14, 5)
)


plt.plot(
    time_hours,
    accel_magnitude,
    alpha=0.5,
    label="Acceleration magnitude"
)


stationary_times = time_hours[
    strict_stationary
]


stationary_accel = accel_magnitude[
    strict_stationary
]


plt.scatter(
    stationary_times,
    stationary_accel,
    s=3,
    label="Strict stationary"
)


plt.axhline(
    ACCEL_THRESHOLD_MS2,
    linestyle="--",
    label="Threshold"
)


plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "Acceleration magnitude (m/s²)"
)


plt.title(
    "High-Confidence Stationary Samples"
)


plt.legend()

plt.grid(
    alpha=0.3
)


plt.tight_layout()


plot2 = (
    OUTPUT_DIR +
    "/strict_stationary_windows.png"
)


plt.savefig(
    plot2,
    dpi=150
)

plt.close()


# ============================================================
# PLOT 3
# WINDOW BIAS
# ============================================================

if len(bias_df) > 0:

    plt.figure(
        figsize=(12, 6)
    )


    plt.plot(
        bias_df["window"],
        bias_df["north_bias"],
        marker="o",
        label="North"
    )


    plt.plot(
        bias_df["window"],
        bias_df["east_bias"],
        marker="o",
        label="East"
    )


    plt.plot(
        bias_df["window"],
        bias_df["down_bias"],
        marker="o",
        label="Down"
    )


    plt.axhline(
        0,
        linestyle="--"
    )


    plt.xlabel(
        "Stationary Window"
    )

    plt.ylabel(
        "Mean acceleration (m/s²)"
    )


    plt.title(
        "Stationary NED Bias Across Windows"
    )


    plt.legend()

    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    plot3 = (
        OUTPUT_DIR +
        "/stationary_bias_stability.png"
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
print("STEP 6.3 COMPLETE")
print("=" * 70)


print()
print("Saved files:")

print(
    "1.",
    window_file
)

print(
    "2.",
    bias_file
)

print(
    "3.",
    sample_file
)

print(
    "4.",
    plot1
)

print(
    "5.",
    plot2
)

if len(bias_df) > 0:

    print(
        "6.",
        plot3
    )


print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "Do NOT modify the main NED acceleration yet."
)

print(
    "Do NOT rerun velocity integration yet."
)

print(
    "Use the stationary-window results to determine"
)

print(
    "whether acceleration bias is stable."
)

print("=" * 70)