import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V9.5
# PHYSICALLY MEANINGFUL IMU + AI + GNSS EKF
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# FILE PATHS
# ============================================================

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

BIAS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "prototype_v9_imu_bias.csv"
)

TARGET_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "prototype_v9_velocity_targets.csv"
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
# GNSS OUTAGE
# ============================================================

OUTAGE_START = 8500.0
OUTAGE_DURATION = 60.0
OUTAGE_END = OUTAGE_START + OUTAGE_DURATION


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    df = pd.read_csv(
        path,
        encoding="cp1252"
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# PARSE RAW TIMESTAMP
# ============================================================

def parse_time(df):

    time_col = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"

    df["timestamp"] = pd.to_datetime(
        df[time_col],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    if df["timestamp"].isna().any():

        bad = df["timestamp"].isna().sum()

        raise ValueError(
            f"{bad} timestamps could not be parsed."
        )

    df["time_s"] = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    return df


# ============================================================
# LAT/LON -> LOCAL NORTH/EAST
# ============================================================

def latlon_to_local(
    lat,
    lon,
    lat0,
    lon0
):

    R = 6371000.0

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    lat0_rad = np.radians(lat0)

    north = (
        lat_rad - lat0_rad
    ) * R

    east = (
        lon_rad
        - np.radians(lon0)
    ) * R * np.cos(lat0_rad)

    return north, east


# ============================================================
# ANGLE WRAP
# ============================================================

def wrap_angle(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


# ============================================================
# ROTATION MATRIX
# ============================================================

def rotation_matrix(
    roll,
    pitch,
    yaw
):

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr]
    ])

    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp]
    ])

    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return Rz @ Ry @ Rx


# ============================================================
# EKF UPDATE
# ============================================================

def ekf_update(
    x,
    P,
    z,
    h,
    H,
    R
):

    innovation = z - h

    S = (
        H
        @ P
        @ H.T
        + R
    )

    try:

        K = (
            P
            @ H.T
            @ np.linalg.inv(S)
        )

    except np.linalg.LinAlgError:

        K = (
            P
            @ H.T
            @ np.linalg.pinv(S)
        )

    x = (
        x
        + K @ innovation
    )

    I = np.eye(
        len(x)
    )

    # Joseph covariance update
    P = (
        (I - K @ H)
        @ P
        @ (I - K @ H).T
        + K @ R @ K.T
    )

    return x, P


# ============================================================
# 1. RAW DATA
# ============================================================

print("\n" + "=" * 70)
print("V9.5 IMU + AI + GNSS EKF")
print("=" * 70)

print("\nLoading raw dataset...")

raw = load_csv(
    RAW_FILE
)

raw = parse_time(
    raw
)

print(
    "Raw rows:",
    len(raw)
)


# ============================================================
# 2. AI VELOCITY OUTPUT
# ============================================================

print(
    "\nLoading AI velocity predictions..."
)

ai = pd.read_csv(
    AI_FILE
)

ai["timestamp"] = pd.to_datetime(
    ai["timestamp"]
)


required_ai = [
    "timestamp",
    "vnorth_ai_mps",
    "veast_ai_mps"
]

for col in required_ai:

    if col not in ai.columns:

        raise ValueError(
            f"AI file missing column: {col}"
        )


# Actual V9.2 names -> internal names
ai = ai.rename(
    columns={
        "vnorth_ai_mps":
            "ai_vn_mps",

        "veast_ai_mps":
            "ai_ve_mps"
    }
)


ai = (
    ai
    .sort_values("timestamp")
    .reset_index(drop=True)
)


print(
    "AI rows:",
    len(ai)
)

print(
    "AI columns:",
    ai.columns.tolist()
)


# ============================================================
# 3. VELOCITY TARGETS
# ============================================================

print(
    "\nLoading velocity targets..."
)

targets = pd.read_csv(
    TARGET_FILE
)

targets["timestamp"] = pd.to_datetime(
    targets["timestamp"]
)


required_targets = [
    "timestamp",
    "vn_mps",
    "ve_mps"
]

for col in required_targets:

    if col not in targets.columns:

        raise ValueError(
            f"Target file missing column: {col}"
        )


targets = (
    targets
    .sort_values("timestamp")
    .reset_index(drop=True)
)


print(
    "Target rows:",
    len(targets)
)


# ============================================================
# 4. IMU BIAS INITIALIZATION
# ============================================================

print(
    "\nLoading IMU bias initialization..."
)

bias_df = pd.read_csv(
    BIAS_FILE
)


required_bias = [
    "sensor",
    "bias",
    "noise_std"
]

for col in required_bias:

    if col not in bias_df.columns:

        raise ValueError(
            f"Bias file missing column: {col}"
        )


# Actual file format:
#
# sensor             bias       noise_std
# accelerometer_x    ...
# accelerometer_y    ...
# accelerometer_z    ...
# gyroscope_x        ...
# gyroscope_y        ...
# gyroscope_z        ...


bias_dict = dict(
    zip(
        bias_df["sensor"],
        bias_df["bias"]
    )
)


noise_dict = dict(
    zip(
        bias_df["sensor"],
        bias_df["noise_std"]
    )
)


print(
    "\nInitial IMU bias:"
)

for sensor in bias_dict:

    print(
        f"  {sensor}: "
        f"{bias_dict[sensor]:.6f}"
    )


# ============================================================
# 5. RAW SENSOR COLUMNS
# ============================================================

required_raw = [

    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",

    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)",

    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",

    "GPS LATITUDE (degrees)",
    "GPS LONGITUDE (degrees)"
]


for col in required_raw:

    if col not in raw.columns:

        raise ValueError(
            f"Raw dataset missing column: {col}"
        )


# ============================================================
# 6. MERGE AI WITH RAW DATA
# ============================================================

print(
    "\nMerging AI velocity predictions..."
)


data = pd.merge_asof(

    raw.sort_values(
        "timestamp"
    ),

    ai[
        [
            "timestamp",
            "ai_vn_mps",
            "ai_ve_mps"
        ]
    ].sort_values(
        "timestamp"
    ),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        milliseconds=80
    )
)


# Raw dataset is the master clock
data["time_s"] = (
    data["timestamp"]
    - data["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# 7. MERGE VELOCITY TARGETS
# ============================================================

print(
    "Merging velocity targets..."
)


data = pd.merge_asof(

    data.sort_values(
        "timestamp"
    ),

    targets[
        [
            "timestamp",
            "vn_mps",
            "ve_mps"
        ]
    ].sort_values(
        "timestamp"
    ),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        milliseconds=80
    )
)


# ============================================================
# 8. REMOVE MISSING VALUES
# ============================================================

before = len(data)


data = data.dropna(
    subset=[
        "ai_vn_mps",
        "ai_ve_mps",
        "vn_mps",
        "ve_mps"
    ]
).reset_index(
    drop=True
)


after = len(data)


print(
    f"Rows after merge: {after}"
)

print(
    f"Rows removed: {before - after}"
)


# ============================================================
# 9. LOCAL GPS COORDINATES
# ============================================================

lat0 = data[
    "GPS LATITUDE (degrees)"
].iloc[0]

lon0 = data[
    "GPS LONGITUDE (degrees)"
].iloc[0]


north, east = latlon_to_local(

    data[
        "GPS LATITUDE (degrees)"
    ].values,

    data[
        "GPS LONGITUDE (degrees)"
    ].values,

    lat0,
    lon0

)


data["gps_north"] = north
data["gps_east"] = east


# ============================================================
# 10. INITIAL ORIENTATION FROM GRAVITY
# ============================================================

gx0 = data[
    "GRAVITY X (m/s²)"
].iloc[0]

gy0 = data[
    "GRAVITY Y (m/s²)"
].iloc[0]

gz0 = data[
    "GRAVITY Z (m/s²)"
].iloc[0]


roll = np.arctan2(
    gy0,
    gz0
)


pitch = np.arctan2(

    -gx0,

    np.sqrt(
        gy0 ** 2
        + gz0 ** 2
    )

)


# Yaw starts at zero.
# A later V9.x stage will improve heading initialization.
yaw = 0.0


print(
    "\nInitial orientation:"
)

print(
    f"Roll : "
    f"{np.degrees(roll):.3f}°"
)

print(
    f"Pitch: "
    f"{np.degrees(pitch):.3f}°"
)

print(
    f"Yaw  : "
    f"{np.degrees(yaw):.3f}°"
)


# ============================================================
# 11. INITIAL BIAS VALUES
# ============================================================

ba = np.array([

    bias_dict.get(
        "accelerometer_x",
        0.0
    ),

    bias_dict.get(
        "accelerometer_y",
        0.0
    ),

    bias_dict.get(
        "accelerometer_z",
        0.0
    )

])


bg = np.array([

    bias_dict.get(
        "gyroscope_x",
        0.0
    ),

    bias_dict.get(
        "gyroscope_y",
        0.0
    ),

    bias_dict.get(
        "gyroscope_z",
        0.0
    )

])


print(
    "\nAccelerometer bias:"
)

print(
    ba
)


print(
    "\nGyroscope bias:"
)

print(
    bg
)


# ============================================================
# 12. SENSOR NOISE
# ============================================================

accel_noise = np.array([

    noise_dict.get(
        "accelerometer_x",
        0.17
    ),

    noise_dict.get(
        "accelerometer_y",
        0.13
    ),

    noise_dict.get(
        "accelerometer_z",
        0.10
    )

])


gyro_noise = np.array([

    noise_dict.get(
        "gyroscope_x",
        0.01
    ),

    noise_dict.get(
        "gyroscope_y",
        0.01
    ),

    noise_dict.get(
        "gyroscope_z",
        0.01
    )

])


print(
    "\nAccelerometer noise:"
)

print(
    accel_noise
)


print(
    "\nGyroscope noise:"
)

print(
    gyro_noise
)


# ============================================================
# 13. EKF STATE
# ============================================================

# State:
#
# x[0] = North position
# x[1] = East position
#
# x[2] = North velocity
# x[3] = East velocity
#
# x[4] = Accelerometer bias X
# x[5] = Accelerometer bias Y
# x[6] = Accelerometer bias Z
#
# x[7] = Gyroscope bias X
# x[8] = Gyroscope bias Y
# x[9] = Gyroscope bias Z


x = np.zeros(
    10
)


# Initial position
x[0] = data[
    "gps_north"
].iloc[0]

x[1] = data[
    "gps_east"
].iloc[0]


# Initial velocity from GNSS-derived target
x[2] = data[
    "vn_mps"
].iloc[0]

x[3] = data[
    "ve_mps"
].iloc[0]


# Initial biases
x[4:7] = ba
x[7:10] = bg


# ============================================================
# 14. INITIAL COVARIANCE
# ============================================================

P = np.diag([

    1.0,
    1.0,

    1.0,
    1.0,

    0.05 ** 2,
    0.05 ** 2,
    0.05 ** 2,

    0.01 ** 2,
    0.01 ** 2,
    0.01 ** 2

])


# ============================================================
# 15. MEASUREMENT NOISE
# ============================================================

AI_VEL_STD = 1.0

GNSS_POS_STD = 5.0


# ============================================================
# 16. RESULTS STORAGE
# ============================================================

results = []


prev_time = data[
    "time_s"
].iloc[0]


# ============================================================
# 17. MAIN PROPAGATION LOOP
# ============================================================

print(
    "\nRunning V9.5 IMU + AI + GNSS EKF..."
)


for i in range(
    len(data)
):

    row = data.iloc[i]


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    current_time = row[
        "time_s"
    ]


    dt = (
        current_time
        - prev_time
    )


    # Protect against abnormal timestamp gaps
    if (
        dt <= 0
        or dt > 0.5
    ):

        dt = 0.1


    prev_time = current_time


    # --------------------------------------------------------
    # ACCELEROMETER
    # --------------------------------------------------------

    acc_body = np.array([

        row[
            "ACCELEROMETER X (m/s²)"
        ],

        row[
            "ACCELEROMETER Y (m/s²)"
        ],

        row[
            "ACCELEROMETER Z (m/s²)"
        ]

    ])


    # --------------------------------------------------------
    # GYROSCOPE
    # --------------------------------------------------------

    gyro_body = np.array([

        row[
            "GYROSCOPE Roll (rad/s)"
        ],

        row[
            "GYROSCOPE Pitch (rad/s)"
        ],

        row[
            "GYROSCOPE Yaw (rad/s)"
        ]

    ])


    # --------------------------------------------------------
    # CURRENT EKF BIASES
    # --------------------------------------------------------

    ba_est = x[
        4:7
    ]

    bg_est = x[
        7:10
    ]


    # --------------------------------------------------------
    # BIAS CORRECTION
    # --------------------------------------------------------

    acc_corrected = (
        acc_body
        - ba_est
    )


    gyro_corrected = (
        gyro_body
        - bg_est
    )


    # --------------------------------------------------------
    # ORIENTATION PROPAGATION
    # --------------------------------------------------------

    roll += (
        gyro_corrected[0]
        * dt
    )

    pitch += (
        gyro_corrected[1]
        * dt
    )

    yaw += (
        gyro_corrected[2]
        * dt
    )


    yaw = wrap_angle(
        yaw
    )


    # --------------------------------------------------------
    # ROTATE BODY ACCELERATION
    # INTO NAVIGATION FRAME
    # --------------------------------------------------------

    R_bn = rotation_matrix(

        roll,
        pitch,
        yaw

    )


    acc_nav = (
        R_bn
        @ acc_corrected
    )


    # --------------------------------------------------------
    # GRAVITY COMPENSATION
    # --------------------------------------------------------

    gravity = np.array([

        0.0,
        0.0,
        9.80665

    ])


    linear_acc_nav = (
        acc_nav
        - gravity
    )


    # --------------------------------------------------------
    # NORTH / EAST ACCELERATION
    # --------------------------------------------------------

    a_n = linear_acc_nav[0]

    a_e = linear_acc_nav[1]


    # --------------------------------------------------------
    # POSITION PROPAGATION
    # --------------------------------------------------------

    x[0] += (

        x[2] * dt

        + 0.5
        * a_n
        * dt ** 2

    )


    x[1] += (

        x[3] * dt

        + 0.5
        * a_e
        * dt ** 2

    )


    # --------------------------------------------------------
    # VELOCITY PROPAGATION
    # --------------------------------------------------------

    x[2] += (
        a_n * dt
    )

    x[3] += (
        a_e * dt
    )


    # ========================================================
    # COVARIANCE PROPAGATION
    # ========================================================

    F = np.eye(
        10
    )


    F[0, 2] = dt
    F[1, 3] = dt


    # --------------------------------------------------------
    # PROCESS NOISE
    # --------------------------------------------------------

    accel_var = np.mean(
        accel_noise ** 2
    )


    gyro_var = np.mean(
        gyro_noise ** 2
    )


    Q = np.zeros(
        (10, 10)
    )


    Q[0, 0] = (
        0.25
        * accel_var
        * dt ** 4
    )


    Q[1, 1] = (
        0.25
        * accel_var
        * dt ** 4
    )


    Q[2, 2] = (
        accel_var
        * dt ** 2
    )


    Q[3, 3] = (
        accel_var
        * dt ** 2
    )


    # Accelerometer bias random walk
    Q[4, 4] = (
        accel_var
        * 1e-4
        * dt
    )

    Q[5, 5] = (
        accel_var
        * 1e-4
        * dt
    )

    Q[6, 6] = (
        accel_var
        * 1e-4
        * dt
    )


    # Gyroscope bias random walk
    Q[7, 7] = (
        gyro_var
        * 1e-4
        * dt
    )

    Q[8, 8] = (
        gyro_var
        * 1e-4
        * dt
    )

    Q[9, 9] = (
        gyro_var
        * 1e-4
        * dt
    )


    # --------------------------------------------------------
    # EKF COVARIANCE PROPAGATION
    # --------------------------------------------------------

    P = (

        F
        @ P
        @ F.T

        + Q

    )


    # ========================================================
    # AI VELOCITY UPDATE
    # ========================================================

    z_ai = np.array([

        row[
            "ai_vn_mps"
        ],

        row[
            "ai_ve_mps"
        ]

    ])


    h_ai = np.array([

        x[2],
        x[3]

    ])


    H_ai = np.zeros(
        (2, 10)
    )


    H_ai[0, 2] = 1.0
    H_ai[1, 3] = 1.0


    R_ai = np.diag([

        AI_VEL_STD ** 2,
        AI_VEL_STD ** 2

    ])


    x, P = ekf_update(

        x,
        P,

        z_ai,
        h_ai,

        H_ai,
        R_ai

    )


    # ========================================================
    # GNSS POSITION UPDATE
    # ========================================================

    gnss_available = not (

        OUTAGE_START
        <= current_time
        <= OUTAGE_END

    )


    if gnss_available:

        z_gps = np.array([

            row[
                "gps_north"
            ],

            row[
                "gps_east"
            ]

        ])


        h_gps = np.array([

            x[0],
            x[1]

        ])


        H_gps = np.zeros(
            (2, 10)
        )


        H_gps[0, 0] = 1.0
        H_gps[1, 1] = 1.0


        R_gps = np.diag([

            GNSS_POS_STD ** 2,
            GNSS_POS_STD ** 2

        ])


        x, P = ekf_update(

            x,
            P,

            z_gps,
            h_gps,

            H_gps,
            R_gps

        )


    # ========================================================
    # POSITION ERROR
    # ========================================================

    position_error = np.sqrt(

        (
            x[0]
            - row["gps_north"]
        ) ** 2

        +

        (
            x[1]
            - row["gps_east"]
        ) ** 2

    )


    # ========================================================
    # SAVE SAMPLE
    # ========================================================

    results.append({

        "timestamp":
            row["timestamp"],

        "time_s":
            current_time,

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

        "gps_north":
            row["gps_north"],

        "gps_east":
            row["gps_east"],

        "position_error_m":
            position_error,

        "gnss_available":
            gnss_available,

        "accel_north":
            a_n,

        "accel_east":
            a_e,

        "roll_deg":
            np.degrees(roll),

        "pitch_deg":
            np.degrees(pitch),

        "yaw_deg":
            np.degrees(yaw)

    })


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    results
)


result_file = os.path.join(

    PROCESSED_DIR,

    "prototype_v9_5_imu_ekf_results.csv"

)


results.to_csv(

    result_file,

    index=False

)


# ============================================================
# 19. OUTAGE EVALUATION
# ============================================================

outage = results[

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


if len(outage) == 0:

    raise RuntimeError(
        "No samples found inside GNSS outage."
    )


mean_error = outage[
    "position_error_m"
].mean()


final_error = outage[
    "position_error_m"
].iloc[-1]


max_error = outage[
    "position_error_m"
].max()


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "V9.5 IMU + AI + GNSS EKF RESULTS"
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
    f"{len(outage)}"
)

print(
    f"Mean position error: "
    f"{mean_error:.3f} m"
)

print(
    f"Final outage error: "
    f"{final_error:.3f} m"
)

print(
    f"Maximum outage error: "
    f"{max_error:.3f} m"
)


# ============================================================
# FINAL BIASES
# ============================================================

print(
    "\nFinal accelerometer bias:"
)

print(

    results[
        [
            "accel_bias_x",
            "accel_bias_y",
            "accel_bias_z"
        ]
    ].iloc[-1].values

)


print(
    "\nFinal gyroscope bias:"
)

print(

    results[
        [
            "gyro_bias_x",
            "gyro_bias_y",
            "gyro_bias_z"
        ]
    ].iloc[-1].values

)


# ============================================================
# 21. TRAJECTORY PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)


plt.plot(

    results["gps_east"],

    results["gps_north"],

    label="GNSS reference"

)


plt.plot(

    results["east"],

    results["north"],

    label="V9.5 IMU + AI EKF"

)


# Highlight the actual outage trajectory
outage_plot = results[

    (
        results["time_s"]
        >= OUTAGE_START
    )

    &

    (
        results["time_s"]
        <= OUTAGE_END
    )

]


if len(outage_plot) > 0:

    plt.plot(

        outage_plot["east"],

        outage_plot["north"],

        linewidth=3,

        label="During GNSS outage"

    )


plt.xlabel(
    "East (m)"
)

plt.ylabel(
    "North (m)"
)

plt.title(
    "V9.5 IMU + AI + GNSS EKF Trajectory"
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

    "prototype_v9_5_imu_ekf_trajectory.png"

)


plt.savefig(

    trajectory_file,

    dpi=200

)

plt.close()


# ============================================================
# 22. ERROR PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(

    results["time_s"],

    results["position_error_m"],

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
    "V9.5 Position Error"
)

plt.legend()

plt.grid(
    True

)

plt.tight_layout()


error_file = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_5_imu_ekf_error.png"

)


plt.savefig(

    error_file,

    dpi=200

)

plt.close()


# ============================================================
# 23. ACCELEROMETER BIAS PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(

    results["time_s"],

    results["accel_bias_x"],

    label="Accelerometer X"

)

plt.plot(

    results["time_s"],

    results["accel_bias_y"],

    label="Accelerometer Y"

)

plt.plot(

    results["time_s"],

    results["accel_bias_z"],

    label="Accelerometer Z"

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
    "Bias (m/s²)"
)

plt.title(
    "V9.5 Accelerometer Bias"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


accel_bias_file = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_5_accel_bias.png"

)


plt.savefig(

    accel_bias_file,

    dpi=200

)

plt.close()


# ============================================================
# 24. GYROSCOPE BIAS PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(

    results["time_s"],

    results["gyro_bias_x"],

    label="Gyroscope X"

)

plt.plot(

    results["time_s"],

    results["gyro_bias_y"],

    label="Gyroscope Y"

)

plt.plot(

    results["time_s"],

    results["gyro_bias_z"],

    label="Gyroscope Z"

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
    "Bias (rad/s)"
)

plt.title(
    "V9.5 Gyroscope Bias"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


gyro_bias_file = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_5_gyro_bias.png"

)


plt.savefig(

    gyro_bias_file,

    dpi=200

)

plt.close()


# ============================================================
# 25. FINISHED
# ============================================================

print(
    "\nSaved:"
)

print(
    result_file
)

print(
    trajectory_file
)

print(
    error_file
)

print(
    accel_bias_file
)

print(
    gyro_bias_file
)


print(
    "\nV9.5 complete."
)