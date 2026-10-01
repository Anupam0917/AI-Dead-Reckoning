"""
V10 - AI + IMU Invariant-Style EKF Prototype

State:
    [north, east,
     vn, ve,
     bax, bay, baz,
     bgx, bgy, bgz]

Prediction:
    Raw IMU -> orientation -> navigation acceleration
            -> velocity -> position

Measurements:
    1. GNSS position when available
    2. AI North/East velocity pseudo-measurement

During GNSS outage:
    IMU propagation + AI velocity measurement

Important:
This is an engineering prototype of an invariant-style
error-state EKF. It is NOT claimed to reproduce every
mathematical detail of a production SE_2(3) IEKF.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    ROOT
    / "data"
    / "raw"
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
    / "M (Driver B)"
    / "S-M.csv"
)

AI_FILE = (
    ROOT
    / "data"
    / "processed"
    / "prototype_v9_8_uncertainty_ai_results.csv"
)

REFERENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "canonical_gnss_reference.csv"
)

BIAS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "prototype_v9_4_imu_bias.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)

OUTPUTS = (
    ROOT
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUTS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

# State:
#
# 0  north
# 1  east
# 2  vn
# 3  ve
# 4  bax
# 5  bay
# 6  baz
# 7  bgx
# 8  bgy
# 9  bgz

STATE_SIZE = 10


# Process noise.
#
# These are deliberately conservative starting values.
# They should be tuned using validation data, not chosen
# to force a particular benchmark result.

POS_PROCESS_NOISE = 0.01

VEL_PROCESS_NOISE = 0.50

ACC_BIAS_PROCESS_NOISE = 0.005

GYRO_BIAS_PROCESS_NOISE = 0.001


# GNSS measurement noise.

GNSS_POSITION_STD = 5.0


# AI velocity measurement floor.

AI_VELOCITY_STD_FLOOR = 0.50


# Initial covariance.

INITIAL_POSITION_STD = 3.0

INITIAL_VELOCITY_STD = 2.0

INITIAL_ACCEL_BIAS_STD = 0.20

INITIAL_GYRO_BIAS_STD = 0.05


# ============================================================
# HELPERS
# ============================================================

def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi].
    """

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def rotation_matrix_from_rpy(
    roll,
    pitch,
    yaw,
):
    """
    Body -> navigation rotation.

    Rotation convention:
        Rz(yaw) Ry(pitch) Rx(roll)
    """

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    R = np.array(
        [
            [
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
            ],
            [
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
            ],
            [
                -sp,
                cp * sr,
                cp * cr,
            ],
        ]
    )

    return R


def normalize_vector(v):
    norm = np.linalg.norm(v)

    if norm < 1e-9:
        return v

    return v / norm


def orientation_from_accel_mag(
    ax,
    ay,
    az,
    mx,
    my,
    mz,
    previous_yaw,
):
    """
    Estimate roll/pitch from gravity and yaw from magnetometer.

    This is intentionally simple and robust for the prototype.

    Accelerometer is used as gravity direction.
    Magnetometer supplies heading.

    Returns:
        roll
        pitch
        yaw
    """

    accel = np.array(
        [ax, ay, az],
        dtype=float,
    )

    gravity_norm = np.linalg.norm(
        accel
    )

    if gravity_norm < 1e-6:
        return (
            0.0,
            0.0,
            previous_yaw,
        )

    axn, ayn, azn = (
        accel / gravity_norm
    )

    roll = np.arctan2(
        ayn,
        azn,
    )

    pitch = np.arctan2(
        -axn,
        np.sqrt(
            ayn * ayn
            +
            azn * azn
        ),
    )

    # Tilt compensated magnetometer.

    mx_comp = (
        mx * np.cos(pitch)
        +
        mz * np.sin(pitch)
    )

    my_comp = (
        mx * np.sin(roll)
        * np.sin(pitch)
        +
        my * np.cos(roll)
        -
        mz * np.sin(roll)
        * np.cos(pitch)
    )

    if (
        abs(mx_comp) < 1e-9
        and
        abs(my_comp) < 1e-9
    ):
        yaw = previous_yaw

    else:
        yaw = np.arctan2(
            -my_comp,
            mx_comp,
        )

    return (
        roll,
        pitch,
        wrap_angle(yaw),
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("V10 AI + IMU IEKF PROTOTYPE")
print("=" * 80)

print("\nLoading raw dataset...")

df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252",
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print(
    "Raw rows:",
    len(df)
)


# ============================================================
# TIMESTAMP
# ============================================================

TIME_COL = (
    "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
)

df["timestamp"] = pd.to_datetime(
    df[TIME_COL],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce",
)


df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

df["time_s"] = (
    (
        df["timestamp"]
        -
        df["timestamp"].iloc[0]
    )
    .dt.total_seconds()
)


# ============================================================
# SENSOR COLUMNS
# ============================================================

ACC_X = "ACCELEROMETER X (m/s²)"
ACC_Y = "ACCELEROMETER Y (m/s²)"
ACC_Z = "ACCELEROMETER Z (m/s²)"

GRAV_X = "GRAVITY X (m/s²)"
GRAV_Y = "GRAVITY Y (m/s²)"
GRAV_Z = "GRAVITY Z (m/s²)"

GYRO_YAW = "GYROSCOPE Yaw (rad/s)"
GYRO_PITCH = "GYROSCOPE Pitch (rad/s)"
GYRO_ROLL = "GYROSCOPE Roll (rad/s)"


for c in [
    ACC_X,
    ACC_Y,
    ACC_Z,
    GRAV_X,
    GRAV_Y,
    GRAV_Z,
    GYRO_YAW,
    GYRO_PITCH,
    GYRO_ROLL,
]:
    df[c] = safe_numeric(
        df[c]
    )


# ============================================================
# MAGNETOMETER DETECTION
# ============================================================

mag_columns = {}

for axis in [
    "X",
    "Y",
    "Z",
]:

    matches = [
        c
        for c in df.columns
        if (
            "MAGNETIC FIELD" in c.upper()
            and f" {axis}" in c
        )
    ]

    if not matches:
        raise ValueError(
            f"Cannot find magnetometer {axis}"
        )

    mag_columns[axis] = matches[0]

    df[
        mag_columns[axis]
    ] = safe_numeric(
        df[
            mag_columns[axis]
        ]
    )


print(
    "\nMagnetometer:"
)

for axis, column in mag_columns.items():
    print(
        f"{axis}: {column}"
    )


# ============================================================
# LOAD AI PREDICTIONS
# ============================================================

print(
    "\nLoading V9.8-B AI velocity..."
)

ai = pd.read_csv(
    AI_FILE
)

ai["timestamp"] = pd.to_datetime(
    ai["timestamp"],
    errors="coerce",
)

ai = ai.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

print(
    "AI rows:",
    len(ai)
)


# ============================================================
# LOAD CANONICAL GNSS REFERENCE
# ============================================================

print(
    "\nLoading canonical GNSS reference..."
)

reference = pd.read_csv(
    REFERENCE_FILE
)

reference["timestamp"] = pd.to_datetime(
    reference["timestamp"],
    errors="coerce",
)

reference = reference.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

reference = reference[
    [
        "timestamp",
        "time_s",
        "north_m",
        "east_m",
    ]
].copy()


# ============================================================
# MERGE
# ============================================================

print(
    "\nMerging data..."
)

df = pd.merge_asof(
    df,
    ai[
        [
            "timestamp",
            "vn_ai_mps",
            "ve_ai_mps",
            "vn_std_mps",
            "ve_std_mps",
        ]
    ],
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(
        milliseconds=60
    ),
)


df = pd.merge_asof(
    df.sort_values(
        "timestamp"
    ),
    reference,
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(
        milliseconds=60
    ),
    suffixes=(
        "",
        "_reference",
    ),
)


df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


print(
    "Merged rows:",
    len(df)
)


# ============================================================
# VALIDITY
# ============================================================

required_columns = [
    "vn_ai_mps",
    "ve_ai_mps",
    "vn_std_mps",
    "ve_std_mps",
    "north_m",
    "east_m",
]


df = df.dropna(
    subset=required_columns
).reset_index(
    drop=True
)


print(
    "Valid rows:",
    len(df)
)


# ============================================================
# INITIAL BIASES
# ============================================================

initial_bax = 0.0
initial_bay = 0.0
initial_baz = 0.0

initial_bgx = 0.0
initial_bgy = 0.0
initial_bgz = 0.0


if BIAS_FILE.exists():

    print(
        "\nLoading V9.4 stationary bias..."
    )

    bias_df = pd.read_csv(
        BIAS_FILE
    )

    # Expected format:
    #
    # sensor,bias,noise_std

    for _, row in bias_df.iterrows():

        sensor = str(
            row["sensor"]
        ).lower()

        bias = float(
            row["bias"]
        )

        if sensor in [
            "accel_x",
            "accelerometer_x",
        ]:
            initial_bax = bias

        elif sensor in [
            "accel_y",
            "accelerometer_y",
        ]:
            initial_bay = bias

        elif sensor in [
            "accel_z",
            "accelerometer_z",
        ]:
            initial_baz = bias

        elif sensor in [
            "gyro_x",
            "gyroscope_x",
        ]:
            initial_bgx = bias

        elif sensor in [
            "gyro_y",
            "gyroscope_y",
        ]:
            initial_bgy = bias

        elif sensor in [
            "gyro_z",
            "gyroscope_z",
        ]:
            initial_bgz = bias


print(
    "\nInitial accelerometer bias:",
    initial_bax,
    initial_bay,
    initial_baz,
)

print(
    "Initial gyroscope bias:",
    initial_bgx,
    initial_bgy,
    initial_bgz,
)


# ============================================================
# STATE INITIALIZATION
# ============================================================

first = df.iloc[0]

x = np.zeros(
    STATE_SIZE,
    dtype=float,
)

x[0] = float(
    first["north_m"]
)

x[1] = float(
    first["east_m"]
)

x[2] = float(
    first["vn_ai_mps"]
)

x[3] = float(
    first["ve_ai_mps"]
)

x[4] = initial_bax
x[5] = initial_bay
x[6] = initial_baz

x[7] = initial_bgx
x[8] = initial_bgy
x[9] = initial_bgz


# ============================================================
# COVARIANCE
# ============================================================

P = np.diag(
    [
        INITIAL_POSITION_STD ** 2,
        INITIAL_POSITION_STD ** 2,

        INITIAL_VELOCITY_STD ** 2,
        INITIAL_VELOCITY_STD ** 2,

        INITIAL_ACCEL_BIAS_STD ** 2,
        INITIAL_ACCEL_BIAS_STD ** 2,
        INITIAL_ACCEL_BIAS_STD ** 2,

        INITIAL_GYRO_BIAS_STD ** 2,
        INITIAL_GYRO_BIAS_STD ** 2,
        INITIAL_GYRO_BIAS_STD ** 2,
    ]
)


# ============================================================
# ORIENTATION
# ============================================================

roll = 0.0
pitch = 0.0
yaw = 0.0


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# MAIN FILTER LOOP
# ============================================================

print(
    "\nRunning IEKF..."
)

previous_time = float(
    df["time_s"].iloc[0]
)


for index, row in df.iterrows():

    current_time = float(
        row["time_s"]
    )

    dt = (
        current_time
        -
        previous_time
    )

    previous_time = current_time


    # --------------------------------------------------------
    # Protect against timestamp anomalies
    # --------------------------------------------------------

    if (
        dt <= 0.0
        or dt > 0.5
    ):
        dt = 0.1


    # --------------------------------------------------------
    # RAW IMU
    # --------------------------------------------------------

    ax = float(
        row[ACC_X]
    )

    ay = float(
        row[ACC_Y]
    )

    az = float(
        row[ACC_Z]
    )

    gx = float(
        row[GYRO_YAW]
    )

    gy = float(
        row[GYRO_PITCH]
    )

    gz = float(
        row[GYRO_ROLL]
    )

    mx = float(
        row[mag_columns["X"]]
    )

    my = float(
        row[mag_columns["Y"]]
    )

    mz = float(
        row[mag_columns["Z"]]
    )


    # --------------------------------------------------------
    # ORIENTATION
    # --------------------------------------------------------

    (
        roll,
        pitch,
        yaw_mag,
    ) = orientation_from_accel_mag(
        ax,
        ay,
        az,
        mx,
        my,
        mz,
        yaw,
    )


    # Complementary yaw update.

    gyro_yaw_rate = (
        gz
        -
        x[9]
    )

    yaw_gyro = wrap_angle(
        yaw
        +
        gyro_yaw_rate
        *
        dt
    )


    # Mostly gyro for short-term heading,
    # magnetometer for slow correction.

    yaw = wrap_angle(
        0.98 * yaw_gyro
        +
        0.02 * yaw_mag
    )


    # --------------------------------------------------------
    # ROTATION
    # --------------------------------------------------------

    R_bn = rotation_matrix_from_rpy(
        roll,
        pitch,
        yaw,
    )


    # --------------------------------------------------------
    # ACCELEROMETER BIAS
    # --------------------------------------------------------

    corrected_accel = np.array(
        [
            ax - x[4],
            ay - x[5],
            az - x[6],
        ]
    )


    # --------------------------------------------------------
    # BODY -> NAVIGATION
    # --------------------------------------------------------

    accel_nav = (
        R_bn
        @
        corrected_accel
    )


    # --------------------------------------------------------
    # GRAVITY
    # --------------------------------------------------------

    # Dataset accelerometer is approximately measuring
    # specific force. Add gravity back into navigation
    # acceleration according to the chosen convention.

    accel_nav[2] -= 9.80665


    # Only horizontal navigation is maintained here.

    accel_n = accel_nav[0]

    accel_e = accel_nav[1]


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    old_vn = x[2]
    old_ve = x[3]


    x[0] += (
        old_vn * dt
        +
        0.5 * accel_n * dt * dt
    )

    x[1] += (
        old_ve * dt
        +
        0.5 * accel_e * dt * dt
    )


    x[2] += (
        accel_n * dt
    )

    x[3] += (
        accel_e * dt
    )


    # --------------------------------------------------------
    # COVARIANCE PREDICTION
    # --------------------------------------------------------

    F = np.eye(
        STATE_SIZE
    )

    F[0, 2] = dt

    F[1, 3] = dt


    Q = np.zeros(
        (
            STATE_SIZE,
            STATE_SIZE,
        )
    )


    Q[0, 0] = (
        POS_PROCESS_NOISE
        *
        dt
    ) ** 2

    Q[1, 1] = (
        POS_PROCESS_NOISE
        *
        dt
    ) ** 2


    Q[2, 2] = (
        VEL_PROCESS_NOISE
        *
        dt
    ) ** 2

    Q[3, 3] = (
        VEL_PROCESS_NOISE
        *
        dt
    ) ** 2


    Q[4, 4] = (
        ACC_BIAS_PROCESS_NOISE
        *
        dt
    ) ** 2

    Q[5, 5] = (
        ACC_BIAS_PROCESS_NOISE
        *
        dt
    ) ** 2

    Q[6, 6] = (
        ACC_BIAS_PROCESS_NOISE
        *
        dt
    ) ** 2


    Q[7, 7] = (
        GYRO_BIAS_PROCESS_NOISE
        *
        dt
    ) ** 2

    Q[8, 8] = (
        GYRO_BIAS_PROCESS_NOISE
        *
        dt
    ) ** 2

    Q[9, 9] = (
        GYRO_BIAS_PROCESS_NOISE
        *
        dt
    ) ** 2


    P = (
        F
        @
        P
        @
        F.T
        +
        Q
    )


    # ========================================================
    # GNSS MEASUREMENT
    # ========================================================

    gnss_available = not (
        OUTAGE_START
        <= current_time
        <= OUTAGE_END
    )


    if gnss_available:

        z = np.array(
            [
                float(
                    row["north_m"]
                ),
                float(
                    row["east_m"]
                ),
            ]
        )


        h = np.array(
            [
                x[0],
                x[1],
            ]
        )


        innovation = (
            z - h
        )


        H = np.zeros(
            (
                2,
                STATE_SIZE,
            )
        )

        H[0, 0] = 1.0
        H[1, 1] = 1.0


        R_gnss = np.diag(
            [
                GNSS_POSITION_STD ** 2,
                GNSS_POSITION_STD ** 2,
            ]
        )


        S = (
            H
            @
            P
            @
            H.T
            +
            R_gnss
        )


        K = (
            P
            @
            H.T
            @
            np.linalg.inv(S)
        )


        x = (
            x
            +
            K
            @
            innovation
        )


        I = np.eye(
            STATE_SIZE
        )


        P = (
            I
            -
            K @ H
        ) @ P


    # ========================================================
    # AI VELOCITY MEASUREMENT
    # ========================================================

    ai_vn = float(
        row["vn_ai_mps"]
    )

    ai_ve = float(
        row["ve_ai_mps"]
    )


    ai_std_n = max(
        float(
            row["vn_std_mps"]
        ),
        AI_VELOCITY_STD_FLOOR,
    )

    ai_std_e = max(
        float(
            row["ve_std_mps"]
        ),
        AI_VELOCITY_STD_FLOOR,
    )


    # --------------------------------------------------------
    # During GNSS outage, trust AI velocity more.
    #
    # When GNSS is available, AI acts as a weaker
    # pseudo-measurement.
    # --------------------------------------------------------

    if gnss_available:

        ai_gain_scale = 2.0

    else:

        ai_gain_scale = 1.0


    R_ai = np.diag(
        [
            (
                ai_std_n
                *
                ai_gain_scale
            ) ** 2,

            (
                ai_std_e
                *
                ai_gain_scale
            ) ** 2,
        ]
    )


    z_ai = np.array(
        [
            ai_vn,
            ai_ve,
        ]
    )


    h_ai = np.array(
        [
            x[2],
            x[3],
        ]
    )


    innovation_ai = (
        z_ai
        -
        h_ai
    )


    H_ai = np.zeros(
        (
            2,
            STATE_SIZE,
        )
    )


    H_ai[0, 2] = 1.0
    H_ai[1, 3] = 1.0


    S_ai = (
        H_ai
        @
        P
        @
        H_ai.T
        +
        R_ai
    )


    K_ai = (
        P
        @
        H_ai.T
        @
        np.linalg.inv(S_ai)
    )


    x = (
        x
        +
        K_ai
        @
        innovation_ai
    )


    I = np.eye(
        STATE_SIZE
    )


    P = (
        I
        -
        K_ai @ H_ai
    ) @ P


    # --------------------------------------------------------
    # Numerical covariance stabilization
    # --------------------------------------------------------

    P = (
        P
        +
        P.T
    ) / 2.0


    # --------------------------------------------------------
    # REFERENCE ERROR
    # --------------------------------------------------------

    ref_n = float(
        row["north_m"]
    )

    ref_e = float(
        row["east_m"]
    )


    position_error = np.sqrt(
        (
            x[0]
            -
            ref_n
        ) ** 2
        +
        (
            x[1]
            -
            ref_e
        ) ** 2
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results.append(
        {
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

            "accel_bias_x":
                x[4],

            "accel_bias_y":
                x[5],

            "accel_bias_z":
                x[6],

            "gyro_bias_x":
                x[7],

            "gyro_bias_y":
                x[8],

            "gyro_bias_z":
                x[9],

            "ai_vn":
                ai_vn,

            "ai_ve":
                ai_ve,

            "ai_std_n":
                ai_std_n,

            "ai_std_e":
                ai_std_e,

            "roll_deg":
                np.degrees(roll),

            "pitch_deg":
                np.degrees(pitch),

            "yaw_deg":
                np.degrees(yaw),

            "reference_north":
                ref_n,

            "reference_east":
                ref_e,

            "position_error_m":
                position_error,

            "accel_n":
                accel_n,

            "accel_e":
                accel_e,
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# OUTAGE RESULTS
# ============================================================

outage = results_df[
    (
        results_df["time_s"]
        >= OUTAGE_START
    )
    &
    (
        results_df["time_s"]
        <= OUTAGE_END
    )
].copy()


if len(outage) == 0:

    raise RuntimeError(
        "No samples found inside outage window."
    )


mean_error = float(
    outage[
        "position_error_m"
    ].mean()
)

median_error = float(
    outage[
        "position_error_m"
    ].median()
)

final_error = float(
    outage[
        "position_error_m"
    ].iloc[-1]
)

max_error = float(
    outage[
        "position_error_m"
    ].max()
)


# ============================================================
# REFERENCE DISTANCE
# ============================================================

dn = np.diff(
    outage[
        "reference_north"
    ].to_numpy()
)

de = np.diff(
    outage[
        "reference_east"
    ].to_numpy()
)

reference_distance = float(
    np.sum(
        np.sqrt(
            dn * dn
            +
            de * de
        )
    )
)


if reference_distance > 0:

    drift_percent = (
        final_error
        /
        reference_distance
        *
        100.0
    )

else:

    drift_percent = np.nan


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("V10 IEKF RESULTS")
print("=" * 80)

print(
    f"\nGNSS outage:"
    f" {OUTAGE_START}s -> {OUTAGE_END}s"
)

print(
    f"Samples: {len(outage)}"
)

print(
    f"\nReference distance:"
    f" {reference_distance:.3f} m"
)

print(
    f"Mean position error:"
    f" {mean_error:.3f} m"
)

print(
    f"Median position error:"
    f" {median_error:.3f} m"
)

print(
    f"Final position error:"
    f" {final_error:.3f} m"
)

print(
    f"Maximum position error:"
    f" {max_error:.3f} m"
)

print(
    f"Relative final drift:"
    f" {drift_percent:.3f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    OUTPUT_DIR
    /
    "prototype_v10_iekf_results.csv"
)

results_df.to_csv(
    results_file,
    index=False,
)


summary = pd.DataFrame(
    [
        {
            "outage_start_s":
                OUTAGE_START,

            "outage_end_s":
                OUTAGE_END,

            "samples":
                len(outage),

            "reference_distance_m":
                reference_distance,

            "mean_error_m":
                mean_error,

            "median_error_m":
                median_error,

            "final_error_m":
                final_error,

            "max_error_m":
                max_error,

            "drift_percent":
                drift_percent,
        }
    ]
)


summary_file = (
    OUTPUT_DIR
    /
    "prototype_v10_iekf_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# TRAJECTORY PLOT
# ============================================================

plt.figure(
    figsize=(12, 8)
)

plt.plot(
    outage[
        "reference_east"
    ],
    outage[
        "reference_north"
    ],
    label="GNSS reference",
)

plt.plot(
    outage[
        "east"
    ],
    outage[
        "north"
    ],
    label="V10 IEKF",
)

plt.xlabel(
    "East (m)"
)

plt.ylabel(
    "North (m)"
)

plt.title(
    "V10 AI + IMU IEKF During GNSS Outage"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.axis(
    "equal"
)

plt.tight_layout()

trajectory_file = (
    OUTPUTS
    /
    "prototype_v10_iekf_trajectory.png"
)

plt.savefig(
    trajectory_file,
    dpi=200,
)

plt.close()


# ============================================================
# ERROR PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    outage[
        "time_s"
    ],
    outage[
        "position_error_m"
    ],
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Position error (m)"
)

plt.title(
    "V10 IEKF Position Error During GNSS Outage"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

error_file = (
    OUTPUTS
    /
    "prototype_v10_iekf_error.png"
)

plt.savefig(
    error_file,
    dpi=200,
)

plt.close()


# ============================================================
# VELOCITY PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    outage[
        "time_s"
    ],
    outage[
        "vn"
    ],
    label="IEKF North velocity",
)

plt.plot(
    outage[
        "time_s"
    ],
    outage[
        "ai_vn"
    ],
    label="AI North velocity",
    alpha=0.7,
)

plt.plot(
    outage[
        "time_s"
    ],
    outage[
        "ve"
    ],
    label="IEKF East velocity",
)

plt.plot(
    outage[
        "time_s"
    ],
    outage[
        "ai_ve"
    ],
    label="AI East velocity",
    alpha=0.7,
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Velocity (m/s)"
)

plt.title(
    "V10 AI Velocity vs IEKF Velocity"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

velocity_file = (
    OUTPUTS
    /
    "prototype_v10_iekf_velocity.png"
)

plt.savefig(
    velocity_file,
    dpi=200,
)

plt.close()


# ============================================================
# BIAS PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    outage[
        "time_s"
    ],
    outage[
        "accel_bias_x"
    ],
    label="Accel X bias",
)

plt.plot(
    outage[
        "time_s"
    ],
    outage[
        "accel_bias_y"
    ],
    label="Accel Y bias",
)

plt.plot(
    outage[
        "time_s"
    ],
    outage[
        "accel_bias_z"
    ],
    label="Accel Z bias",
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Bias"
)

plt.title(
    "V10 Estimated Accelerometer Bias"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

bias_file = (
    OUTPUTS
    /
    "prototype_v10_iekf_bias.png"
)

plt.savefig(
    bias_file,
    dpi=200,
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("V10 IEKF COMPLETE")
print("=" * 80)

print(
    "\nResults:"
)

print(
    results_file
)

print(
    "\nSummary:"
)

print(
    summary_file
)

print(
    "\nTrajectory:"
)

print(
    trajectory_file
)

print(
    "\nError:"
)

print(
    error_file
)

print(
    "\nVelocity:"
)

print(
    velocity_file
)

print(
    "\nBias:"
)

print(
    bias_file
)