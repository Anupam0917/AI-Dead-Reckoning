import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V9.6
# CONFIDENCE-AWARE AI DEAD RECKONING
#
# Pipeline:
#
# Raw IMU
#    |
#    +--> Motion / stationary detection
#    |
#    +--> AI velocity: Vnorth + Veast
#    |
#    +--> AI confidence
#    |
#    +--> Adaptive velocity fusion
#    |
#    +--> Dead reckoning
#    |
#    +--> GNSS correction when available
#    |
#    +--> Stop correction
#
# IMPORTANT:
# This version does NOT integrate raw IMU acceleration into
# position. Raw IMU is used for motion/reliability estimation.
# ============================================================


# ============================================================
# PATHS
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

AI_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "prototype_v9_ai_velocity_results.csv"
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

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

EARTH_RADIUS = 6371000.0

# Stationary detector
STATIONARY_WINDOW = 20

STATIONARY_ACCEL_THRESHOLD = 0.35
STATIONARY_GYRO_THRESHOLD = 0.08
STATIONARY_GRAVITY_THRESHOLD = 0.30

# Speed below this value is considered approximately stopped
STOP_SPEED_THRESHOLD = 0.50

# Confidence limits
MIN_CONFIDENCE = 0.15
MAX_CONFIDENCE = 1.00

# Maximum accepted timestep
MAX_DT = 0.5


# ============================================================
# LOAD RAW DATA
# ============================================================

print("\n" + "=" * 70)
print("V9.6 CONFIDENCE-AWARE AI DEAD RECKONING")
print("=" * 70)

print("\nLoading raw dataset...")

if not os.path.exists(RAW_FILE):

    raise FileNotFoundError(
        f"Raw dataset not found:\n{RAW_FILE}"
    )

df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print(
    f"Raw rows: {len(df)}"
)


# ============================================================
# TIMESTAMP
# ============================================================

TIME_COLUMN = (
    "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
)

if TIME_COLUMN not in df.columns:

    raise ValueError(
        f"Missing timestamp column: {TIME_COLUMN}"
    )

df["timestamp"] = pd.to_datetime(
    df[TIME_COLUMN],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)

if df["timestamp"].isna().any():

    bad_count = df["timestamp"].isna().sum()

    raise ValueError(
        f"Timestamp parsing failed for "
        f"{bad_count} rows."
    )

df["time_s"] = (
    df["timestamp"]
    - df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# LOAD AI VELOCITY
# ============================================================

print("\nLoading AI velocity...")

if not os.path.exists(AI_FILE):

    raise FileNotFoundError(
        f"AI velocity file not found:\n{AI_FILE}"
    )

ai = pd.read_csv(
    AI_FILE
)

if "timestamp" not in ai.columns:

    raise ValueError(
        "AI file does not contain timestamp column."
    )

ai["timestamp"] = pd.to_datetime(
    ai["timestamp"],
    errors="coerce"
)

required_ai = [
    "timestamp",
    "vnorth_ai_mps",
    "veast_ai_mps"
]

for column in required_ai:

    if column not in ai.columns:

        raise ValueError(
            f"Missing AI column: {column}"
        )

print(
    f"AI rows: {len(ai)}"
)


# ============================================================
# MERGE AI VELOCITY WITH RAW DATA
# ============================================================

ai_small = ai[
    [
        "timestamp",
        "vnorth_ai_mps",
        "veast_ai_mps"
    ]
].copy()

ai_small = (
    ai_small
    .sort_values("timestamp")
    .drop_duplicates(
        subset="timestamp"
    )
)

df = (
    df
    .sort_values("timestamp")
)

df = pd.merge_asof(

    df,

    ai_small,

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        milliseconds=60
    )

)

print(
    f"Rows after AI merge: {len(df)}"
)


# ============================================================
# SENSOR COLUMN DEFINITIONS
# ============================================================

ACC_X = "ACCELEROMETER X (m/s²)"
ACC_Y = "ACCELEROMETER Y (m/s²)"
ACC_Z = "ACCELEROMETER Z (m/s²)"

GRAV_X = "GRAVITY X (m/s²)"
GRAV_Y = "GRAVITY Y (m/s²)"
GRAV_Z = "GRAVITY Z (m/s²)"

GYRO_ROLL = "GYROSCOPE Roll (rad/s)"
GYRO_PITCH = "GYROSCOPE Pitch (rad/s)"
GYRO_YAW = "GYROSCOPE Yaw (rad/s)"

GPS_LAT = "GPS LATITUDE (degrees)"
GPS_LON = "GPS LONGITUDE (degrees)"


required_sensor_columns = [

    ACC_X,
    ACC_Y,
    ACC_Z,

    GRAV_X,
    GRAV_Y,
    GRAV_Z,

    GYRO_ROLL,
    GYRO_PITCH,
    GYRO_YAW,

    GPS_LAT,
    GPS_LON

]


for column in required_sensor_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required sensor column: {column}"
        )


# ============================================================
# MAGNETOMETER COLUMN DETECTION
# ============================================================

def find_magnetometer_columns(dataframe):

    found = {}

    for column in dataframe.columns:

        normalized = (
            str(column)
            .upper()
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
            .replace("_", "")
        )

        if (
            "MAGNETICFIELDX" in normalized
            and "MAGNETICFIELDY" not in normalized
            and "MAGNETICFIELDZ" not in normalized
        ):

            found["x"] = column

        elif (
            "MAGNETICFIELDY" in normalized
            and "MAGNETICFIELDX" not in normalized
            and "MAGNETICFIELDZ" not in normalized
        ):

            found["y"] = column

        elif (
            "MAGNETICFIELDZ" in normalized
            and "MAGNETICFIELDX" not in normalized
            and "MAGNETICFIELDY" not in normalized
        ):

            found["z"] = column

    if len(found) != 3:

        print(
            "\nMagnetometer columns found:"
        )

        for column in dataframe.columns:

            if "MAGNETIC" in str(column).upper():

                print(
                    repr(column)
                )

        raise ValueError(
            "Could not identify X/Y/Z magnetometer columns."
        )

    return (
        found["x"],
        found["y"],
        found["z"]
    )


(
    MAG_X,
    MAG_Y,
    MAG_Z
) = find_magnetometer_columns(
    df
)

print(
    "\nDetected magnetometer columns:"
)

print(
    "X:",
    repr(MAG_X)
)

print(
    "Y:",
    repr(MAG_Y)
)

print(
    "Z:",
    repr(MAG_Z)
)


# ============================================================
# EXTRACT SENSOR ARRAYS
# ============================================================

ax = df[ACC_X].to_numpy(
    dtype=float
)

ay = df[ACC_Y].to_numpy(
    dtype=float
)

az = df[ACC_Z].to_numpy(
    dtype=float
)

gravity_x = df[GRAV_X].to_numpy(
    dtype=float
)

gravity_y = df[GRAV_Y].to_numpy(
    dtype=float
)

gravity_z = df[GRAV_Z].to_numpy(
    dtype=float
)

gx = df[GYRO_ROLL].to_numpy(
    dtype=float
)

gy = df[GYRO_PITCH].to_numpy(
    dtype=float
)

gz = df[GYRO_YAW].to_numpy(
    dtype=float
)

mx = df[MAG_X].to_numpy(
    dtype=float
)

my = df[MAG_Y].to_numpy(
    dtype=float
)

mz = df[MAG_Z].to_numpy(
    dtype=float
)

lat = df[GPS_LAT].to_numpy(
    dtype=float
)

lon = df[GPS_LON].to_numpy(
    dtype=float
)

time_s = df[
    "time_s"
].to_numpy(
    dtype=float
)

ai_vn = df[
    "vnorth_ai_mps"
].to_numpy(
    dtype=float
)

ai_ve = df[
    "veast_ai_mps"
].to_numpy(
    dtype=float
)


# ============================================================
# VALID AI MASK
# ============================================================

ai_valid = (

    np.isfinite(ai_vn)

    &

    np.isfinite(ai_ve)

)


print(
    f"\nValid AI velocity rows: "
    f"{ai_valid.sum()}"
)


# ============================================================
# SENSOR MAGNITUDES
# ============================================================

accel_magnitude = np.sqrt(

    ax ** 2
    + ay ** 2
    + az ** 2

)

gravity_magnitude = np.sqrt(

    gravity_x ** 2
    + gravity_y ** 2
    + gravity_z ** 2

)

gyro_magnitude = np.sqrt(

    gx ** 2
    + gy ** 2
    + gz ** 2

)

linear_accel_magnitude = np.sqrt(

    (
        ax - gravity_x
    ) ** 2

    +

    (
        ay - gravity_y
    ) ** 2

    +

    (
        az - gravity_z
    ) ** 2

)


# ============================================================
# REPLACE INVALID VALUES
# ============================================================

linear_accel_magnitude = np.nan_to_num(

    linear_accel_magnitude,

    nan=999.0,
    posinf=999.0,
    neginf=999.0

)

gyro_magnitude = np.nan_to_num(

    gyro_magnitude,

    nan=999.0,
    posinf=999.0,
    neginf=999.0

)

gravity_difference = np.abs(

    accel_magnitude
    - gravity_magnitude

)

gravity_difference = np.nan_to_num(

    gravity_difference,

    nan=999.0,
    posinf=999.0,
    neginf=999.0

)


# ============================================================
# ROLLING MOTION STATISTICS
# ============================================================

print(
    "\nCalculating motion statistics..."
)

linear_accel_series = pd.Series(
    linear_accel_magnitude
)

gyro_series = pd.Series(
    gyro_magnitude
)

gravity_difference_series = pd.Series(
    gravity_difference
)


accel_mean = (

    linear_accel_series

    .rolling(
        STATIONARY_WINDOW,
        center=True,
        min_periods=1
    )

    .mean()

    .to_numpy()

)


gyro_mean = (

    gyro_series

    .rolling(
        STATIONARY_WINDOW,
        center=True,
        min_periods=1
    )

    .mean()

    .to_numpy()

)


gravity_difference_mean = (

    gravity_difference_series

    .rolling(
        STATIONARY_WINDOW,
        center=True,
        min_periods=1
    )

    .mean()

    .to_numpy()

)


# ============================================================
# STATIONARY DETECTION
#
# IMPORTANT:
# Parentheses are required around every NumPy comparison
# before using the bitwise '&' operator.
# ============================================================

stationary = (

    (accel_mean < STATIONARY_ACCEL_THRESHOLD)

    &

    (gyro_mean < STATIONARY_GYRO_THRESHOLD)

    &

    (
        gravity_difference_mean
        < STATIONARY_GRAVITY_THRESHOLD
    )

)


print(
    f"Stationary samples: "
    f"{stationary.sum()} "
    f"({100.0 * stationary.mean():.2f}%)"
)


# ============================================================
# AI SPEED
# ============================================================

ai_speed = np.sqrt(

    ai_vn ** 2
    + ai_ve ** 2

)

ai_speed[
    ~ai_valid
] = np.nan


# ============================================================
# CONFIDENCE ESTIMATION
#
# Confidence combines:
#
# 1. AI availability
# 2. acceleration activity
# 3. angular activity
# 4. stationary detection
#
# This is a transparent prototype confidence score.
# ============================================================

confidence = np.full(

    len(df),

    MIN_CONFIDENCE,

    dtype=float

)


# ------------------------------------------------------------
# Base confidence
# ------------------------------------------------------------

confidence[
    ai_valid
] = 0.70


# ------------------------------------------------------------
# Acceleration stability
# ------------------------------------------------------------

accel_factor = np.exp(

    -linear_accel_magnitude / 8.0

)


# ------------------------------------------------------------
# Gyroscope stability
# ------------------------------------------------------------

gyro_factor = np.exp(

    -gyro_magnitude / 4.0

)


# ------------------------------------------------------------
# Combined sensor factor
# ------------------------------------------------------------

sensor_factor = (

    0.5 * accel_factor

    +

    0.5 * gyro_factor

)


valid_indices = ai_valid


confidence[
    valid_indices
] += (

    0.25
    * sensor_factor[valid_indices]

)


# ------------------------------------------------------------
# Stationary confidence
# ------------------------------------------------------------

stationary_ai = (

    stationary
    & ai_valid

)

confidence[
    stationary_ai
] = 1.0


# ------------------------------------------------------------
# Clamp confidence
# ------------------------------------------------------------

confidence = np.clip(

    confidence,

    MIN_CONFIDENCE,

    MAX_CONFIDENCE

)


# ============================================================
# GNSS LOCAL N/E COORDINATES
# ============================================================

origin_lat = lat[0]
origin_lon = lon[0]

origin_lat_rad = np.radians(
    origin_lat
)

lat_rad = np.radians(
    lat
)

lon_rad = np.radians(
    lon
)

gnss_n = (

    lat_rad
    - origin_lat_rad

) * EARTH_RADIUS


gnss_e = (

    lon_rad
    - np.radians(origin_lon)

) * EARTH_RADIUS * np.cos(
    origin_lat_rad
)


# ============================================================
# GNSS AVAILABILITY
# ============================================================

gnss_available = np.ones(

    len(df),

    dtype=bool

)

outage_mask = (

    (time_s >= OUTAGE_START)

    &

    (time_s <= OUTAGE_END)

)

gnss_available[
    outage_mask
] = False


# ============================================================
# NAVIGATION ARRAYS
# ============================================================

fused_vn = np.zeros(
    len(df)
)

fused_ve = np.zeros(
    len(df)
)

fused_speed = np.zeros(
    len(df)
)

nav_n = np.zeros(
    len(df)
)

nav_e = np.zeros(
    len(df)
)


# ============================================================
# INITIALIZATION
# ============================================================

if ai_valid[0]:

    fused_vn[0] = ai_vn[0]
    fused_ve[0] = ai_ve[0]

else:

    fused_vn[0] = 0.0
    fused_ve[0] = 0.0


nav_n[0] = gnss_n[0]
nav_e[0] = gnss_e[0]


# ============================================================
# DEAD RECKONING LOOP
# ============================================================

print(
    "\nRunning confidence-aware dead reckoning..."
)


previous_time = time_s[0]


for i in range(
    1,
    len(df)
):

    # ========================================================
    # TIMESTEP
    # ========================================================

    dt = (

        time_s[i]
        - previous_time

    )

    previous_time = time_s[i]


    if (
        dt <= 0
        or dt > MAX_DT
    ):

        dt = 0.1


    # ========================================================
    # AI VELOCITY FUSION
    # ========================================================

    if ai_valid[i]:

        c = confidence[i]

        fused_vn[i] = (

            c * ai_vn[i]

            +

            (1.0 - c)
            * fused_vn[i - 1]

        )

        fused_ve[i] = (

            c * ai_ve[i]

            +

            (1.0 - c)
            * fused_ve[i - 1]

        )

    else:

        fused_vn[i] = (
            fused_vn[i - 1]
        )

        fused_ve[i] = (
            fused_ve[i - 1]
        )


    # ========================================================
    # STOP CORRECTION
    # ========================================================

    if stationary[i]:

        fused_vn[i] *= 0.20

        fused_ve[i] *= 0.20


    # ========================================================
    # VELOCITY LIMIT
    #
    # Prevent accidental AI spikes from producing huge
    # position jumps.
    # ========================================================

    current_speed = np.sqrt(

        fused_vn[i] ** 2
        + fused_ve[i] ** 2

    )

    if current_speed > 20.0:

        scale = (
            20.0
            / current_speed
        )

        fused_vn[i] *= scale
        fused_ve[i] *= scale


    fused_speed[i] = np.sqrt(

        fused_vn[i] ** 2
        + fused_ve[i] ** 2

    )


    # ========================================================
    # POSITION PROPAGATION
    # ========================================================

    nav_n[i] = (

        nav_n[i - 1]

        +

        fused_vn[i] * dt

    )

    nav_e[i] = (

        nav_e[i - 1]

        +

        fused_ve[i] * dt

    )


    # ========================================================
    # GNSS CORRECTION
    # ========================================================

    if gnss_available[i]:

        position_gain = 0.80

        nav_n[i] = (

            (1.0 - position_gain)
            * nav_n[i]

            +

            position_gain
            * gnss_n[i]

        )

        nav_e[i] = (

            (1.0 - position_gain)
            * nav_e[i]

            +

            position_gain
            * gnss_e[i]

        )


        # ----------------------------------------------------
        # Optional velocity correction from GNSS displacement
        # ----------------------------------------------------

        if dt > 0:

            gnss_vn = (

                gnss_n[i]
                - gnss_n[i - 1]

            ) / dt

            gnss_ve = (

                gnss_e[i]
                - gnss_e[i - 1]

            ) / dt


            # Ignore implausible coordinate-jump velocities.

            if (

                abs(gnss_vn) < 15.0

                and

                abs(gnss_ve) < 15.0

            ):

                velocity_gain = 0.25

                fused_vn[i] = (

                    (1.0 - velocity_gain)
                    * fused_vn[i]

                    +

                    velocity_gain
                    * gnss_vn

                )

                fused_ve[i] = (

                    (1.0 - velocity_gain)
                    * fused_ve[i]

                    +

                    velocity_gain
                    * gnss_ve

                )


# ============================================================
# FINAL FUSED SPEED
# ============================================================

fused_speed = np.sqrt(

    fused_vn ** 2
    + fused_ve ** 2

)


# ============================================================
# POSITION ERROR
# ============================================================

position_error = np.sqrt(

    (
        nav_n
        - gnss_n
    ) ** 2

    +

    (
        nav_e
        - gnss_e
    ) ** 2

)


# ============================================================
# OUTAGE DATA
# ============================================================

outage_errors = (
    position_error[
        outage_mask
    ]
)

outage_times = (
    time_s[
        outage_mask
    ]
)


if len(outage_errors) == 0:

    raise RuntimeError(
        "No samples found inside outage window."
    )


# ============================================================
# METRICS
# ============================================================

mean_error = np.mean(
    outage_errors
)

median_error = np.median(
    outage_errors
)

final_error = outage_errors[-1]

max_error = np.max(
    outage_errors
)


# ============================================================
# GNSS REFERENCE DISTANCE
# ============================================================

outage_gnss_n = (
    gnss_n[
        outage_mask
    ]
)

outage_gnss_e = (
    gnss_e[
        outage_mask
    ]
)

gnss_distance = np.sqrt(

    (
        outage_gnss_n[-1]
        - outage_gnss_n[0]
    ) ** 2

    +

    (
        outage_gnss_e[-1]
        - outage_gnss_e[0]
    ) ** 2

)


if gnss_distance > 0:

    relative_drift = (

        final_error
        / gnss_distance
        * 100.0

    )

else:

    relative_drift = np.nan


# ============================================================
# SAVE COMPLETE RESULTS
# ============================================================

results = pd.DataFrame({

    "timestamp":
        df["timestamp"],

    "time_s":
        time_s,

    "gnss_available":
        gnss_available,

    "gnss_n":
        gnss_n,

    "gnss_e":
        gnss_e,

    "nav_n":
        nav_n,

    "nav_e":
        nav_e,

    "ai_vn":
        ai_vn,

    "ai_ve":
        ai_ve,

    "fused_vn":
        fused_vn,

    "fused_ve":
        fused_ve,

    "fused_speed_mps":
        fused_speed,

    "fused_speed_kmh":
        fused_speed * 3.6,

    "confidence":
        confidence,

    "stationary":
        stationary,

    "linear_accel_magnitude":
        linear_accel_magnitude,

    "gyro_magnitude":
        gyro_magnitude,

    "position_error":
        position_error

})


result_file = os.path.join(

    PROCESSED_DIR,

    "prototype_v9_6_confidence_dr_results.csv"

)

results.to_csv(

    result_file,

    index=False

)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({

    "metric": [

        "outage_start_s",
        "outage_end_s",

        "outage_samples",

        "gnss_distance_m",

        "mean_position_error_m",

        "median_position_error_m",

        "final_position_error_m",

        "maximum_position_error_m",

        "relative_final_drift_percent",

        "mean_ai_confidence",

        "stationary_fraction_percent"

    ],

    "value": [

        OUTAGE_START,
        OUTAGE_END,

        outage_mask.sum(),

        gnss_distance,

        mean_error,

        median_error,

        final_error,

        max_error,

        relative_drift,

        np.mean(
            confidence[
                outage_mask
            ]
        ),

        100.0 * np.mean(
            stationary[
                outage_mask
            ]
        )

    ]

})


summary_file = os.path.join(

    PROCESSED_DIR,

    "prototype_v9_6_confidence_dr_summary.csv"

)

summary.to_csv(

    summary_file,

    index=False

)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "V9.6 RESULTS"
)

print(
    "=" * 70
)

print(
    f"GNSS outage: "
    f"{OUTAGE_START:.0f}s -> "
    f"{OUTAGE_END:.0f}s"
)

print(
    f"Outage samples: "
    f"{outage_mask.sum()}"
)

print(
    f"GNSS reference distance: "
    f"{gnss_distance:.3f} m"
)

print(
    f"Mean position error: "
    f"{mean_error:.3f} m"
)

print(
    f"Median position error: "
    f"{median_error:.3f} m"
)

print(
    f"Final outage error: "
    f"{final_error:.3f} m"
)

print(
    f"Maximum position error: "
    f"{max_error:.3f} m"
)

print(
    f"Relative final drift: "
    f"{relative_drift:.3f}%"
)

print(
    f"Mean AI confidence: "
    f"{np.mean(confidence[outage_mask]):.3f}"
)

print(
    f"Stationary fraction: "
    f"{100.0 * np.mean(stationary[outage_mask]):.2f}%"
)


# ============================================================
# PLOT 1: TRAJECTORY
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.plot(

    gnss_e,
    gnss_n,

    label="GNSS reference"

)

plt.plot(

    nav_e,
    nav_n,

    label="V9.6 confidence-aware DR"

)

plt.xlabel(
    "East (m)"
)

plt.ylabel(
    "North (m)"
)

plt.title(
    "V9.6 Confidence-Aware Dead Reckoning"
)

plt.legend()

plt.grid(
    True
)

plt.axis(
    "equal"
)

plt.tight_layout()


trajectory_file = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_6_confidence_dr_trajectory.png"

)

plt.savefig(

    trajectory_file,

    dpi=200

)

plt.close()


# ============================================================
# PLOT 2: POSITION ERROR
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(

    time_s,
    position_error,

    label="Position error"

)

plt.axvspan(

    OUTAGE_START,
    OUTAGE_END,

    alpha=0.2,

    label="GNSS outage"

)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Position error (m)"
)

plt.title(
    "V9.6 Position Error"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


error_file = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_6_confidence_dr_error.png"

)

plt.savefig(

    error_file,

    dpi=200

)

plt.close()


# ============================================================
# PLOT 3: AI CONFIDENCE
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(

    time_s,
    confidence,

    label="AI confidence"

)

plt.axvspan(

    OUTAGE_START,
    OUTAGE_END,

    alpha=0.2,

    label="GNSS outage"

)

plt.ylim(
    0,
    1.05
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Confidence"
)

plt.title(
    "V9.6 AI Reliability / Confidence"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


confidence_file = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_6_confidence.png"

)

plt.savefig(

    confidence_file,

    dpi=200

)

plt.close()


# ============================================================
# PLOT 4: VELOCITY
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(

    time_s,
    ai_vn,

    label="AI North velocity"

)

plt.plot(

    time_s,
    fused_vn,

    label="Fused North velocity"

)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Velocity (m/s)"
)

plt.title(
    "V9.6 North Velocity Fusion"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


velocity_file = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_6_velocity.png"

)

plt.savefig(

    velocity_file,

    dpi=200

)

plt.close()


# ============================================================
# PLOT 5: STATIONARY DETECTION
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(

    time_s,
    stationary.astype(int),

    label="Stationary"

)

plt.axvspan(

    OUTAGE_START,
    OUTAGE_END,

    alpha=0.2,

    label="GNSS outage"

)

plt.yticks(
    [0, 1],
    ["Moving", "Stationary"]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Motion state"
)

plt.title(
    "V9.6 Stationary Detection"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


stationary_file = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_6_stationary_detection.png"

)

plt.savefig(

    stationary_file,

    dpi=200

)

plt.close()


# ============================================================
# DONE
# ============================================================

print(
    "\nSaved:"
)

print(
    result_file
)

print(
    summary_file
)

print(
    trajectory_file
)

print(
    error_file
)

print(
    confidence_file
)

print(
    velocity_file
)

print(
    stationary_file
)

print(
    "\nV9.6 complete."
)

print(
    "Do not tune parameters yet."
)

print(
    "Send the complete terminal output."
)