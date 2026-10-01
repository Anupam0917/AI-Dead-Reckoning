import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_FILE = os.path.join(
    BASE,
    "data",
    "raw",
    "Synchronised V abd S datasets",
    "Categorised IOVNB Dataset",
    "M (Driver B)",
    "S-M.csv"
)

TARGET_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v9_velocity_targets.csv"
)

AI_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_1_full_ai_velocity.csv"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_13_iekf_style.csv"
)


# ============================================================
# SETTINGS
# ============================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

# Initial covariance
INITIAL_POSITION_STD = 2.0
INITIAL_VELOCITY_STD = 1.0
INITIAL_YAW_STD = np.deg2rad(20.0)
INITIAL_GYRO_BIAS_STD = 0.02

# Process noise
POSITION_PROCESS_NOISE = 0.05
VELOCITY_PROCESS_NOISE = 0.8
YAW_PROCESS_NOISE = np.deg2rad(2.0)
GYRO_BIAS_PROCESS_NOISE = 0.0005

# GNSS position measurement noise
GNSS_POSITION_STD = 3.0

# AI velocity measurement base noise
AI_VELOCITY_STD = 0.8

# AI velocity gets more trust when confidence is high
MIN_AI_STD = 0.35
MAX_AI_STD = 2.0

# Acceleration propagation weight
ACCELERATION_SCALE = 0.35

# Maximum usable dt
MAX_DT = 0.5


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 80)
print("V10.13 IEKF-STYLE ERROR-STATE FUSION")
print("=" * 80)

print("\nDevice:", device)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_angle(angle):

    return (
        angle + np.pi
    ) % (
        2 * np.pi
    ) - np.pi


def rotation_matrix_2d(yaw):

    c = np.cos(yaw)
    s = np.sin(yaw)

    return np.array([
        [c, -s],
        [s,  c]
    ])


def safe_inverse(matrix):

    try:

        return np.linalg.inv(matrix)

    except np.linalg.LinAlgError:

        return np.linalg.pinv(matrix)


# ============================================================
# 1. LOAD AI VELOCITY
# ============================================================

print("\n[1] Loading AI velocity predictions...")

ai = pd.read_csv(
    AI_FILE
)

ai["timestamp"] = pd.to_datetime(
    ai["timestamp"]
)

print(
    "AI rows:",
    len(ai)
)


# ============================================================
# 2. LOAD TARGETS
# ============================================================

print("\n[2] Loading velocity targets...")

target = pd.read_csv(
    TARGET_FILE
)

target["timestamp"] = pd.to_datetime(
    target["timestamp"]
)

target = target.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# 3. LOAD RAW SENSOR DATA
# ============================================================

print("\n[3] Loading raw sensor data...")

raw = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

raw.columns = raw.columns.str.strip()

raw["timestamp"] = pd.to_datetime(
    raw[
        "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
    ],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)


# ============================================================
# 4. FIND MAGNETOMETER
# ============================================================

mag_x = None
mag_y = None
mag_z = None

for column in raw.columns:

    upper = column.upper()

    if "MAGNETIC FIELD" in upper:

        if " X " in upper:
            mag_x = column

        elif " Y " in upper:
            mag_y = column

        elif " Z " in upper:
            mag_z = column


if (
    mag_x is None
    or mag_y is None
    or mag_z is None
):

    raise ValueError(
        "Magnetometer columns not found."
    )


print(
    "Mag X:",
    mag_x
)

print(
    "Mag Y:",
    mag_y
)

print(
    "Mag Z:",
    mag_z
)


# ============================================================
# 5. NUMERIC CONVERSION
# ============================================================

def numeric(column):

    return pd.to_numeric(
        raw[column],
        errors="coerce"
    )


# ============================================================
# 6. SENSOR DATA
# ============================================================

sensor = pd.DataFrame({

    "timestamp":
        raw["timestamp"],

    "gyro_yaw":
        numeric(
            "GYROSCOPE Yaw (rad/s)"
        ),

    "gyro_pitch":
        numeric(
            "GYROSCOPE Pitch (rad/s)"
        ),

    "gyro_roll":
        numeric(
            "GYROSCOPE Roll (rad/s)"
        ),

    "acc_x":
        numeric(
            "ACCELEROMETER X (m/s²)"
        ),

    "acc_y":
        numeric(
            "ACCELEROMETER Y (m/s²)"
        ),

    "acc_z":
        numeric(
            "ACCELEROMETER Z (m/s²)"
        ),

    "grav_x":
        numeric(
            "GRAVITY X (m/s²)"
        ),

    "grav_y":
        numeric(
            "GRAVITY Y (m/s²)"
        ),

    "grav_z":
        numeric(
            "GRAVITY Z (m/s²)"
        ),

    "mag_x":
        numeric(mag_x),

    "mag_y":
        numeric(mag_y),

    "mag_z":
        numeric(mag_z)

})


# ============================================================
# 7. CREATE GNSS REFERENCE
# ============================================================

print("\n[4] Creating GNSS reference...")


gps = pd.DataFrame({

    "timestamp":
        raw["timestamp"],

    "latitude":
        numeric(
            "GPS LATITUDE (degrees)"
        ),

    "longitude":
        numeric(
            "GPS LONGITUDE (degrees)"
        ),

    "gps_speed_kmh":
        numeric(
            "GPS SPEED (Kmh)"
        ),

    "gps_accuracy":
        numeric(
            "GPS ACCURACY (m)"
        )

})


gps = gps.dropna(
    subset=[
        "latitude",
        "longitude"
    ]
)


# ============================================================
# 8. MERGE ALL DATA
# ============================================================

print("\n[5] Synchronizing data...")


data = pd.merge_asof(

    target[
        [
            "timestamp",
            "time_s",
            "vn_mps",
            "ve_mps"
        ]
    ].sort_values("timestamp"),

    ai.sort_values(
        "timestamp"
    ),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        "50ms"
    )
)


data = pd.merge_asof(

    data.sort_values("timestamp"),

    sensor.sort_values(
        "timestamp"
    ),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        "50ms"
    )
)


# GNSS is sparse, therefore use nearest
# with a larger matching tolerance

data = pd.merge_asof(

    data.sort_values("timestamp"),

    gps.sort_values(
        "timestamp"
    ),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        "600ms"
    )
)


# ============================================================
# 9. IDENTIFY AI COLUMNS
# ============================================================

required_ai = [
    "vn_ai_mps",
    "ve_ai_mps"
]

for column in required_ai:

    if column not in data.columns:

        raise ValueError(
            f"Missing AI column: {column}"
        )


# ============================================================
# 10. GNSS LOCAL COORDINATES
# ============================================================

origin_lat = data[
    "latitude"
].dropna().iloc[0]

origin_lon = data[
    "longitude"
].dropna().iloc[0]


EARTH_RADIUS = 6371000.0


def latlon_to_local(
    lat,
    lon
):

    lat_rad = np.deg2rad(
        lat
    )

    lon_rad = np.deg2rad(
        lon
    )

    lat0_rad = np.deg2rad(
        origin_lat
    )

    lon0_rad = np.deg2rad(
        origin_lon
    )

    north = (
        lat_rad - lat0_rad
    ) * EARTH_RADIUS

    east = (
        lon_rad - lon0_rad
    ) * EARTH_RADIUS * np.cos(
        lat0_rad
    )

    return north, east


data[
    "gps_north"
], data[
    "gps_east"
] = latlon_to_local(
    data["latitude"].values,
    data["longitude"].values
)


# ============================================================
# 11. SORT AND CLEAN
# ============================================================

data = data.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


sensor_columns = [
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",
    "acc_x",
    "acc_y",
    "acc_z",
    "grav_x",
    "grav_y",
    "grav_z",
    "mag_x",
    "mag_y",
    "mag_z"
]


data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


data[sensor_columns] = (
    data[sensor_columns]
    .interpolate(
        limit_direction="both"
    )
)


data = data.dropna(
    subset=[
        "vn_ai_mps",
        "ve_ai_mps",
        "time_s"
    ]
).reset_index(
    drop=True
)


print(
    "Final usable rows:",
    len(data)
)


# ============================================================
# 12. INITIAL YAW ESTIMATION
# ============================================================

print("\n[6] Estimating initial heading...")


# Use AI velocity direction over a short
# pre-outage interval.

calibration = data[
    (data["time_s"] >= 8300)
    &
    (data["time_s"] < 8400)
].copy()


speed = np.sqrt(

    calibration["vn_ai_mps"] ** 2
    +
    calibration["ve_ai_mps"] ** 2

)


moving = calibration[
    speed > 2.0
]


if len(moving) > 20:

    desired_heading = np.arctan2(
        moving["ve_ai_mps"].values,
        moving["vn_ai_mps"].values
    )

    # Estimate gyro-integrated yaw
    # relative to first sample.

    gyro_time = moving[
        "time_s"
    ].values

    gyro_values = moving[
        "gyro_yaw"
    ].values

    gyro_yaw_relative = np.zeros(
        len(gyro_values)
    )

    for i in range(
        1,
        len(gyro_values)
    ):

        dtime = np.clip(
            gyro_time[i]
            -
            gyro_time[i - 1],
            0.001,
            MAX_DT
        )

        gyro_yaw_relative[i] = (
            gyro_yaw_relative[i - 1]
            +
            gyro_values[i - 1]
            * dtime
        )

    angle_difference = np.angle(
        np.exp(
            1j
            *
            (
                desired_heading
                -
                gyro_yaw_relative
            )
        )
    )

    initial_yaw = np.angle(
        np.mean(
            np.exp(
                1j
                *
                angle_difference
            )
        )
    )

else:

    initial_yaw = 0.0


print(
    "Initial yaw:",
    round(
        np.rad2deg(initial_yaw),
        3
    ),
    "degrees"
)


# ============================================================
# 13. FILTER STATE
# ============================================================

# State:
#
# x =
# [ north,
#   east,
#   vn,
#   ve,
#   yaw,
#   gyro_bias ]
#
# This is an error-state/EKF-style
# implementation.

x = np.zeros(
    6,
    dtype=float
)


x[0] = data[
    "gps_north"
].dropna().iloc[0]

x[1] = data[
    "gps_east"
].dropna().iloc[0]

x[2] = data[
    "vn_ai_mps"
].iloc[0]

x[3] = data[
    "ve_ai_mps"
].iloc[0]

x[4] = initial_yaw

x[5] = 0.0


# ============================================================
# 14. COVARIANCE MATRIX
# ============================================================

P = np.diag([

    INITIAL_POSITION_STD ** 2,

    INITIAL_POSITION_STD ** 2,

    INITIAL_VELOCITY_STD ** 2,

    INITIAL_VELOCITY_STD ** 2,

    INITIAL_YAW_STD ** 2,

    INITIAL_GYRO_BIAS_STD ** 2

])


# ============================================================
# 15. OUTPUT STORAGE
# ============================================================

results = []


# ============================================================
# 16. MAIN FILTER LOOP
# ============================================================

print("\n[7] Running fusion filter...")


previous_time = None


for index, row in data.iterrows():

    current_time = float(
        row["time_s"]
    )


    # --------------------------------------------------------
    # DT
    # --------------------------------------------------------

    if previous_time is None:

        dt = 0.1

    else:

        dt = np.clip(
            current_time
            -
            previous_time,
            0.001,
            MAX_DT
        )


    previous_time = current_time


    # --------------------------------------------------------
    # SENSOR VALUES
    # --------------------------------------------------------

    gyro_z = float(
        row["gyro_yaw"]
    )


    acc_x = float(
        row["acc_x"]
    )

    acc_y = float(
        row["acc_y"]
    )

    acc_z = float(
        row["acc_z"]
    )


    grav_x = float(
        row["grav_x"]
    )

    grav_y = float(
        row["grav_y"]
    )

    grav_z = float(
        row["grav_z"]
    )


    # --------------------------------------------------------
    # LINEAR ACCELERATION
    # --------------------------------------------------------

    linear_acc = np.array([

        acc_x - grav_x,

        acc_y - grav_y,

        acc_z - grav_z

    ])


    # Only horizontal magnitude is used
    # for the 2D propagation.

    body_acc = np.array([

        linear_acc[0],

        linear_acc[1]

    ])


    # --------------------------------------------------------
    # YAW PROPAGATION
    # --------------------------------------------------------

    yaw = x[4]

    gyro_bias = x[5]

    corrected_gyro = (
        gyro_z
        -
        gyro_bias
    )


    yaw = wrap_angle(
        yaw
        +
        corrected_gyro
        *
        dt
    )


    # --------------------------------------------------------
    # BODY -> NAV ROTATION
    # --------------------------------------------------------

    R = rotation_matrix_2d(
        yaw
    )


    nav_acc = (
        R
        @
        body_acc
    )


    nav_acc *= ACCELERATION_SCALE


    # --------------------------------------------------------
    # STATE PREDICTION
    # --------------------------------------------------------

    old_vn = x[2]

    old_ve = x[3]


    x[0] = (
        x[0]
        +
        old_vn * dt
        +
        0.5
        *
        nav_acc[0]
        *
        dt
        *
        dt
    )


    x[1] = (
        x[1]
        +
        old_ve * dt
        +
        0.5
        *
        nav_acc[1]
        *
        dt
        *
        dt
    )


    x[2] = (
        old_vn
        +
        nav_acc[0]
        *
        dt
    )


    x[3] = (
        old_ve
        +
        nav_acc[1]
        *
        dt
    )


    x[4] = yaw


    # --------------------------------------------------------
    # PROCESS COVARIANCE
    # --------------------------------------------------------

    Q = np.diag([

        POSITION_PROCESS_NOISE
        *
        dt,

        POSITION_PROCESS_NOISE
        *
        dt,

        VELOCITY_PROCESS_NOISE
        *
        dt,

        VELOCITY_PROCESS_NOISE
        *
        dt,

        YAW_PROCESS_NOISE
        *
        dt,

        GYRO_BIAS_PROCESS_NOISE
        *
        dt

    ])


    P = (
        P
        +
        Q
    )


    # --------------------------------------------------------
    # AI VELOCITY MEASUREMENT
    # --------------------------------------------------------

    ai_vn = float(
        row["vn_ai_mps"]
    )

    ai_ve = float(
        row["ve_ai_mps"]
    )


    ai_speed = np.sqrt(
        ai_vn ** 2
        +
        ai_ve ** 2
    )


    # Higher-speed AI estimates are
    # generally more informative for
    # heading correction.

    if ai_speed > 2.0:

        ai_heading = np.arctan2(
            ai_ve,
            ai_vn
        )

        heading_error = wrap_angle(
            ai_heading
            -
            x[4]
        )

    else:

        heading_error = 0.0


    # --------------------------------------------------------
    # AI CONFIDENCE
    # --------------------------------------------------------

    # Confidence decreases slightly
    # at very low speed and increases
    # during stable motion.

    if ai_speed < 0.5:

        confidence = 0.55

    elif ai_speed < 2.0:

        confidence = 0.70

    else:

        confidence = 0.85


    ai_std = (

        AI_VELOCITY_STD
        /
        np.sqrt(confidence)

    )


    ai_std = np.clip(
        ai_std,
        MIN_AI_STD,
        MAX_AI_STD
    )


    # --------------------------------------------------------
    # VELOCITY UPDATE
    # --------------------------------------------------------

    H = np.array([

        [0, 0, 1, 0, 0, 0],

        [0, 0, 0, 1, 0, 0]

    ], dtype=float)


    z = np.array([

        ai_vn,
        ai_ve

    ])


    predicted = np.array([

        x[2],
        x[3]

    ])


    innovation = (
        z
        -
        predicted
    )


    R_ai = np.diag([

        ai_std ** 2,
        ai_std ** 2

    ])


    S = (
        H
        @ P
        @ H.T
        +
        R_ai
    )


    K = (
        P
        @ H.T
        @ safe_inverse(S)
    )


    x = (
        x
        +
        K
        @ innovation
    )


    P = (
        np.eye(6)
        -
        K @ H
    ) @ P


    # --------------------------------------------------------
    # AI HEADING CORRECTION
    # --------------------------------------------------------

    if ai_speed > 2.0:

        heading_variance = (
            np.deg2rad(12.0)
            /
            np.sqrt(confidence)
        ) ** 2


        yaw_error = wrap_angle(
            heading_error
        )


        yaw_gain = (

            P[4, 4]
            /
            (
                P[4, 4]
                +
                heading_variance
            )

        )


        x[4] = wrap_angle(

            x[4]
            +
            yaw_gain
            *
            yaw_error

        )


        P[4, 4] *= (
            1
            -
            yaw_gain
        )


    # --------------------------------------------------------
    # GNSS POSITION UPDATE
    # --------------------------------------------------------

    gnss_available = (

        current_time
        <
        OUTAGE_START

        or

        current_time
        >
        OUTAGE_END

    )


    gps_n = row[
        "gps_north"
    ]

    gps_e = row[
        "gps_east"
    ]


    if (

        gnss_available
        and
        not pd.isna(gps_n)
        and
        not pd.isna(gps_e)

    ):

        H_gps = np.array([

            [1, 0, 0, 0, 0, 0],

            [0, 1, 0, 0, 0, 0]

        ], dtype=float)


        z_gps = np.array([

            float(gps_n),
            float(gps_e)

        ])


        predicted_gps = np.array([

            x[0],
            x[1]

        ])


        gps_innovation = (
            z_gps
            -
            predicted_gps
        )


        R_gps = np.diag([

            GNSS_POSITION_STD ** 2,

            GNSS_POSITION_STD ** 2

        ])


        S_gps = (

            H_gps
            @ P
            @ H_gps.T
            +
            R_gps

        )


        K_gps = (

            P
            @ H_gps.T
            @ safe_inverse(S_gps)

        )


        x = (

            x
            +
            K_gps
            @ gps_innovation

        )


        P = (

            np.eye(6)
            -
            K_gps @ H_gps

        ) @ P


    # --------------------------------------------------------
    # REFERENCE ERROR
    # --------------------------------------------------------

    reference_n = row[
        "gps_north"
    ]

    reference_e = row[
        "gps_east"
    ]


    if (
        not pd.isna(reference_n)
        and
        not pd.isna(reference_e)
    ):

        position_error = np.sqrt(

            (
                x[0]
                -
                float(reference_n)
            ) ** 2

            +

            (
                x[1]
                -
                float(reference_e)
            ) ** 2

        )

    else:

        position_error = np.nan


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results.append({

        "timestamp":
            row["timestamp"],

        "time_s":
            current_time,

        "gnss_available":
            gnss_available,

        "north":
            x[0],

        "east":
            x[1],

        "vn":
            x[2],

        "ve":
            x[3],

        "yaw_deg":
            np.rad2deg(x[4]),

        "gyro_bias":
            x[5],

        "ai_vn":
            ai_vn,

        "ai_ve":
            ai_ve,

        "ai_speed":
            ai_speed,

        "ai_confidence":
            confidence,

        "position_error_m":
            position_error

    })


# ============================================================
# 17. RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    results
)


# ============================================================
# 18. OUTAGE EVALUATION
# ============================================================

outage_results = results[
    (
        results["time_s"]
        >= OUTAGE_START
    )
    &
    (
        results["time_s"]
        <= OUTAGE_END
    )
].copy()


if len(outage_results) == 0:

    raise ValueError(
        "No outage results generated."
    )


# ============================================================
# 19. OUTAGE REFERENCE
# ============================================================

# Use the target-derived velocity reference
# consistently with our common benchmark.

outage_reference = data[
    (
        data["time_s"]
        >= OUTAGE_START
    )
    &
    (
        data["time_s"]
        <= OUTAGE_END
    )
].copy()


outage_reference = (
    outage_reference
    .reset_index(drop=True)
)


ref_vn = outage_reference[
    "vn_mps"
].values


ref_ve = outage_reference[
    "ve_mps"
].values


ref_time = outage_reference[
    "time_s"
].values


dt_ref = np.diff(
    ref_time
)


dt_ref = np.clip(
    dt_ref,
    0.001,
    MAX_DT
)


reference_n = np.zeros(
    len(ref_time)
)

reference_e = np.zeros(
    len(ref_time)
)


for i in range(
    1,
    len(ref_time)
):

    reference_n[i] = (

        reference_n[i - 1]

        +

        ref_vn[i - 1]
        *
        dt_ref[i - 1]

    )


    reference_e[i] = (

        reference_e[i - 1]

        +

        ref_ve[i - 1]
        *
        dt_ref[i - 1]

    )


# ============================================================
# 20. MATCH FILTER TRAJECTORY TO REFERENCE
# ============================================================

filter_n = np.interp(
    ref_time,
    outage_results["time_s"],
    outage_results["north"]
)


filter_e = np.interp(
    ref_time,
    outage_results["time_s"],
    outage_results["east"]
)


# Normalize both trajectories to
# the beginning of the outage.

filter_n = (
    filter_n
    -
    filter_n[0]
)

filter_e = (
    filter_e
    -
    filter_e[0]
)


# ============================================================
# 21. POSITION ERROR
# ============================================================

position_error = np.sqrt(

    (
        filter_n
        -
        reference_n
    ) ** 2

    +

    (
        filter_e
        -
        reference_e
    ) ** 2

)


mean_error = np.mean(
    position_error
)

final_error = position_error[-1]

max_error = np.max(
    position_error
)


# ============================================================
# 22. DISTANCE
# ============================================================

reference_distance = np.sum(

    np.sqrt(

        np.diff(reference_n) ** 2
        +
        np.diff(reference_e) ** 2

    )

)


filter_distance = np.sum(

    np.sqrt(

        np.diff(filter_n) ** 2
        +
        np.diff(filter_e) ** 2

    )

)


drift_percent = (

    abs(
        filter_distance
        -
        reference_distance
    )

    /

    reference_distance

    *

    100

)


# ============================================================
# 23. OUTAGE VELOCITY ERROR
# ============================================================

filter_vn = np.interp(
    ref_time,
    outage_results["time_s"],
    outage_results["vn"]
)

filter_ve = np.interp(
    ref_time,
    outage_results["time_s"],
    outage_results["ve"]
)


vn_mae = np.mean(
    np.abs(
        filter_vn
        -
        ref_vn
    )
)


ve_mae = np.mean(
    np.abs(
        filter_ve
        -
        ref_ve
    )
)


# ============================================================
# 24. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("V10.13 OUTAGE RESULTS")
print("=" * 80)

print(
    "\nOutage:",
    OUTAGE_START,
    "to",
    OUTAGE_END,
    "seconds"
)

print(
    "\nNorth velocity MAE:",
    round(
        vn_mae,
        4
    ),
    "m/s"
)

print(
    "East velocity MAE:",
    round(
        ve_mae,
        4
    ),
    "m/s"
)

print(
    "\nMean position error:",
    round(
        mean_error,
        3
    ),
    "m"
)

print(
    "Final position error:",
    round(
        final_error,
        3
    ),
    "m"
)

print(
    "Maximum position error:",
    round(
        max_error,
        3
    ),
    "m"
)

print(
    "Reference distance:",
    round(
        reference_distance,
        3
    ),
    "m"
)

print(
    "Filter distance:",
    round(
        filter_distance,
        3
    ),
    "m"
)

print(
    "Drift:",
    round(
        drift_percent,
        3
    ),
    "%"
)


# ============================================================
# 25. SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(
    OUTPUT_FILE
)


print("\n")
print("=" * 80)
print("V10.13 COMPLETE")
print("=" * 80)