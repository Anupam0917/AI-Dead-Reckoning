import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V9.5.2
# PHYSICAL IMU PROPAGATION + AI VELOCITY + GNSS EKF
#
# State:
#   [N, E, Vn, Ve, bax, bay, baz, bgx, bgy, bgz]
#
# Inputs:
#   Raw accelerometer
#   Raw gyroscope
#   Magnetometer
#   AI North/East velocity
#   GNSS position
#
# Test:
#   GNSS outage: 8500 -> 8560 seconds
#
# Important:
#   This is a prototype EKF, not the final IEKF.
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

BIAS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "prototype_v9_imu_bias.csv"
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

GRAVITY = 9.80665

MAX_DT = 0.5

# AI velocity measurement noise
AI_VELOCITY_STD = 0.8

# GNSS position measurement noise
GNSS_POSITION_STD = 3.0

# IMU acceleration process noise
ACCEL_PROCESS_STD = 0.5

# Bias random walk
ACCEL_BIAS_RW = 0.005
GYRO_BIAS_RW = 0.001

# Initial state covariance
INITIAL_POSITION_STD = 2.0
INITIAL_VELOCITY_STD = 1.0
INITIAL_ACCEL_BIAS_STD = 0.10
INITIAL_GYRO_BIAS_STD = 0.02


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def wrap_angle(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def angle_difference(target, current):

    return wrap_angle(
        target - current
    )


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


def gravity_roll_pitch(
    ax,
    ay,
    az
):

    roll = np.arctan2(
        ay,
        az
    )

    pitch = np.arctan2(
        -ax,
        np.sqrt(
            ay ** 2
            + az ** 2
        )
    )

    return roll, pitch


def magnetometer_heading(
    mx,
    my,
    mz,
    roll,
    pitch
):

    mx_level = (
        mx * np.cos(pitch)
        + mz * np.sin(pitch)
    )

    my_level = (
        mx * np.sin(roll) * np.sin(pitch)
        + my * np.cos(roll)
        - mz * np.sin(roll) * np.cos(pitch)
    )

    return wrap_angle(
        np.arctan2(
            -my_level,
            mx_level
        )
    )


# ============================================================
# FIND MAGNETOMETER COLUMNS
# ============================================================

def find_magnetometer_columns(df):

    found = {}

    for column in df.columns:

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
            "\nAvailable magnetometer columns:"
        )

        for column in df.columns:

            if "MAGNETIC" in str(column).upper():

                print(
                    repr(column)
                )

        raise ValueError(
            "Could not identify magnetometer columns."
        )

    return (
        found["x"],
        found["y"],
        found["z"]
    )


# ============================================================
# LOAD RAW DATA
# ============================================================

print("\n" + "=" * 70)
print("V9.5.2 IMU + AI + GNSS EKF")
print("=" * 70)

print(
    "\nLoading raw dataset..."
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


# ============================================================
# TIMESTAMP
# ============================================================

TIME_COLUMN = (
    "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
)

df["timestamp"] = pd.to_datetime(
    df[TIME_COLUMN],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)

if df["timestamp"].isna().any():

    raise ValueError(
        "Timestamp parsing failed."
    )


df["time_s"] = (
    df["timestamp"]
    - df["timestamp"].iloc[0]
).dt.total_seconds()


print(
    f"Raw rows: {len(df)}"
)


# ============================================================
# LOAD AI VELOCITY
# ============================================================

print(
    "\nLoading AI velocity..."
)

ai = pd.read_csv(
    AI_FILE
)

ai["timestamp"] = pd.to_datetime(
    ai["timestamp"]
)

print(
    f"AI rows: {len(ai)}"
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


# ============================================================
# LOAD BIAS
# ============================================================

print(
    "\nLoading IMU bias..."
)

bias_df = pd.read_csv(
    BIAS_FILE
)

bias = dict(
    zip(
        bias_df["sensor"],
        bias_df["bias"]
    )
)

noise = dict(
    zip(
        bias_df["sensor"],
        bias_df["noise_std"]
    )
)


print(
    "\nInitial accelerometer bias:"
)

print(
    bias["accelerometer_x"],
    bias["accelerometer_y"],
    bias["accelerometer_z"]
)

print(
    "\nInitial gyroscope bias:"
)

print(
    bias["gyroscope_x"],
    bias["gyroscope_y"],
    bias["gyroscope_z"]
)


# ============================================================
# SENSOR COLUMNS
# ============================================================

ACC_X = "ACCELEROMETER X (m/s²)"
ACC_Y = "ACCELEROMETER Y (m/s²)"
ACC_Z = "ACCELEROMETER Z (m/s²)"

GYRO_ROLL = "GYROSCOPE Roll (rad/s)"
GYRO_PITCH = "GYROSCOPE Pitch (rad/s)"
GYRO_YAW = "GYROSCOPE Yaw (rad/s)"

GPS_LAT = "GPS LATITUDE (degrees)"
GPS_LON = "GPS LONGITUDE (degrees)"

(
    MAG_X,
    MAG_Y,
    MAG_Z
) = find_magnetometer_columns(
    df
)


# ============================================================
# CONVERT SENSOR DATA
# ============================================================

ax = df[ACC_X].to_numpy(dtype=float)
ay = df[ACC_Y].to_numpy(dtype=float)
az = df[ACC_Z].to_numpy(dtype=float)

gx = df[GYRO_ROLL].to_numpy(dtype=float)
gy = df[GYRO_PITCH].to_numpy(dtype=float)
gz = df[GYRO_YAW].to_numpy(dtype=float)

mx = df[MAG_X].to_numpy(dtype=float)
my = df[MAG_Y].to_numpy(dtype=float)
mz = df[MAG_Z].to_numpy(dtype=float)

lat = df[GPS_LAT].to_numpy(dtype=float)
lon = df[GPS_LON].to_numpy(dtype=float)


# ============================================================
# INITIAL BIAS
# ============================================================

initial_ba = np.array([

    bias["accelerometer_x"],
    bias["accelerometer_y"],
    bias["accelerometer_z"]

])

initial_bg = np.array([

    bias["gyroscope_x"],
    bias["gyroscope_y"],
    bias["gyroscope_z"]

])


# ============================================================
# MERGE AI VELOCITY
# ============================================================

ai_small = ai[
    [
        "timestamp",
        "vnorth_ai_mps",
        "veast_ai_mps"
    ]
].copy()


df = pd.merge_asof(

    df.sort_values("timestamp"),

    ai_small.sort_values("timestamp"),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        milliseconds=60
    )

)


print(
    f"\nRows after AI merge: {len(df)}"
)


# Re-extract time after merge

time_s = df[
    "time_s"
].to_numpy(
    dtype=float
)

ax = df[ACC_X].to_numpy(dtype=float)
ay = df[ACC_Y].to_numpy(dtype=float)
az = df[ACC_Z].to_numpy(dtype=float)

gx = df[GYRO_ROLL].to_numpy(dtype=float)
gy = df[GYRO_PITCH].to_numpy(dtype=float)
gz = df[GYRO_YAW].to_numpy(dtype=float)

mx = df[MAG_X].to_numpy(dtype=float)
my = df[MAG_Y].to_numpy(dtype=float)
mz = df[MAG_Z].to_numpy(dtype=float)

lat = df[GPS_LAT].to_numpy(dtype=float)
lon = df[GPS_LON].to_numpy(dtype=float)

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
    f"Valid AI velocity rows: "
    f"{ai_valid.sum()}"
)


# ============================================================
# REFERENCE ORIGIN
# ============================================================

origin_lat = lat[0]
origin_lon = lon[0]

EARTH_RADIUS = 6371000.0

lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

origin_lat_rad = np.radians(
    origin_lat
)

origin_lon_rad = np.radians(
    origin_lon
)

gnss_n = (
    lat_rad - origin_lat_rad
) * EARTH_RADIUS

gnss_e = (
    lon_rad - origin_lon_rad
) * EARTH_RADIUS * np.cos(
    origin_lat_rad
)


# ============================================================
# INITIAL ORIENTATION
# ============================================================

ax0 = ax[0] - initial_ba[0]
ay0 = ay[0] - initial_ba[1]
az0 = az[0] - initial_ba[2]

roll, pitch = gravity_roll_pitch(
    ax0,
    ay0,
    az0
)

yaw = magnetometer_heading(
    mx[0],
    my[0],
    mz[0],
    roll,
    pitch
)


print(
    "\nInitial orientation:"
)

print(
    f"Roll : {np.degrees(roll):.3f}°"
)

print(
    f"Pitch: {np.degrees(pitch):.3f}°"
)

print(
    f"Yaw  : {np.degrees(yaw):.3f}°"
)


# ============================================================
# EKF STATE
#
# x =
# [N, E, Vn, Ve, bax, bay, baz, bgx, bgy, bgz]
# ============================================================

STATE_SIZE = 10

x = np.zeros(
    STATE_SIZE
)

# Start at GNSS origin

x[0] = 0.0
x[1] = 0.0

# Initial velocity from AI

if ai_valid[0]:

    x[2] = ai_vn[0]
    x[3] = ai_ve[0]

# Initial biases

x[4:7] = initial_ba
x[7:10] = initial_bg


# ============================================================
# INITIAL COVARIANCE
# ============================================================

P = np.diag([

    INITIAL_POSITION_STD ** 2,
    INITIAL_POSITION_STD ** 2,

    INITIAL_VELOCITY_STD ** 2,
    INITIAL_VELOCITY_STD ** 2,

    INITIAL_ACCEL_BIAS_STD ** 2,
    INITIAL_ACCEL_BIAS_STD ** 2,
    INITIAL_ACCEL_BIAS_STD ** 2,

    INITIAL_GYRO_BIAS_STD ** 2,
    INITIAL_GYRO_BIAS_STD ** 2,
    INITIAL_GYRO_BIAS_STD ** 2

])


# ============================================================
# STORAGE
# ============================================================

estimated_n = np.zeros(len(df))
estimated_e = np.zeros(len(df))

estimated_vn = np.zeros(len(df))
estimated_ve = np.zeros(len(df))

estimated_bax = np.zeros(len(df))
estimated_bay = np.zeros(len(df))
estimated_baz = np.zeros(len(df))

estimated_bgx = np.zeros(len(df))
estimated_bgy = np.zeros(len(df))
estimated_bgz = np.zeros(len(df))

estimated_roll = np.zeros(len(df))
estimated_pitch = np.zeros(len(df))
estimated_yaw = np.zeros(len(df))

position_error = np.zeros(len(df))

imu_acc_n = np.zeros(len(df))
imu_acc_e = np.zeros(len(df))


# ============================================================
# EKF LOOP
# ============================================================

print(
    "\nRunning physical IMU propagation + EKF..."
)


previous_time = time_s[0]


for i in range(len(df)):

    if i == 0:

        estimated_n[i] = x[0]
        estimated_e[i] = x[1]

        estimated_vn[i] = x[2]
        estimated_ve[i] = x[3]

        estimated_bax[i] = x[4]
        estimated_bay[i] = x[5]
        estimated_baz[i] = x[6]

        estimated_bgx[i] = x[7]
        estimated_bgy[i] = x[8]
        estimated_bgz[i] = x[9]

        estimated_roll[i] = roll
        estimated_pitch[i] = pitch
        estimated_yaw[i] = yaw

        position_error[i] = 0.0

        continue


    current_time = time_s[i]

    dt = (
        current_time
        - previous_time
    )

    previous_time = current_time


    if (
        dt <= 0
        or dt > MAX_DT
    ):

        dt = 0.1


    # ========================================================
    # BIAS-CORRECTED GYRO
    # ========================================================

    gyro_x = (
        gx[i]
        - x[7]
    )

    gyro_y = (
        gy[i]
        - x[8]
    )

    gyro_z = (
        gz[i]
        - x[9]
    )


    # ========================================================
    # GYRO ORIENTATION PROPAGATION
    # ========================================================

    roll += gyro_x * dt
    pitch += gyro_y * dt
    yaw += gyro_z * dt

    yaw = wrap_angle(yaw)


    # ========================================================
    # ACCELEROMETER BIAS CORRECTION
    # ========================================================

    acc_body = np.array([

        ax[i] - x[4],
        ay[i] - x[5],
        az[i] - x[6]

    ])


    # ========================================================
    # BODY -> NAVIGATION
    # ========================================================

    R = rotation_matrix(

        roll,
        pitch,
        yaw

    )


    acc_nav = (
        R @ acc_body
    )


    # ========================================================
    # GRAVITY REMOVAL
    # ========================================================

    linear_acc = (

        acc_nav
        - np.array([
            0.0,
            0.0,
            GRAVITY
        ])

    )


    acc_n = linear_acc[0]
    acc_e = linear_acc[1]


    imu_acc_n[i] = acc_n
    imu_acc_e[i] = acc_e


    # ========================================================
    # PHYSICAL STATE PROPAGATION
    # ========================================================

    old_vn = x[2]
    old_ve = x[3]


    x[0] += (
        old_vn * dt
        + 0.5 * acc_n * dt * dt
    )

    x[1] += (
        old_ve * dt
        + 0.5 * acc_e * dt * dt
    )


    x[2] += (
        acc_n * dt
    )

    x[3] += (
        acc_e * dt
    )


    # ========================================================
    # STATE TRANSITION MATRIX
    # ========================================================

    F = np.eye(
        STATE_SIZE
    )

    F[0, 2] = dt
    F[1, 3] = dt


    # ========================================================
    # PROCESS NOISE
    # ========================================================

    Q = np.zeros(
        (STATE_SIZE, STATE_SIZE)
    )


    accel_q = (
        ACCEL_PROCESS_STD ** 2
    )

    bias_a_q = (
        ACCEL_BIAS_RW ** 2
    )

    bias_g_q = (
        GYRO_BIAS_RW ** 2
    )


    Q[2, 2] = (
        accel_q * dt * dt
    )

    Q[3, 3] = (
        accel_q * dt * dt
    )

    Q[4, 4] = bias_a_q * dt
    Q[5, 5] = bias_a_q * dt
    Q[6, 6] = bias_a_q * dt

    Q[7, 7] = bias_g_q * dt
    Q[8, 8] = bias_g_q * dt
    Q[9, 9] = bias_g_q * dt


    # ========================================================
    # COVARIANCE PROPAGATION
    # ========================================================

    P = (
        F
        @ P
        @ F.T
        + Q
    )


    # ========================================================
    # AI VELOCITY UPDATE
    # ========================================================

    if ai_valid[i]:

        z = np.array([

            ai_vn[i],
            ai_ve[i]

        ])


        h = np.array([

            x[2],
            x[3]

        ])


        innovation = (
            z - h
        )


        H = np.zeros(
            (2, STATE_SIZE)
        )

        H[0, 2] = 1.0
        H[1, 3] = 1.0


        R_ai = np.diag([

            AI_VELOCITY_STD ** 2,
            AI_VELOCITY_STD ** 2

        ])


        S = (
            H
            @ P
            @ H.T
            + R_ai
        )


        K = (
            P
            @ H.T
            @ np.linalg.inv(S)
        )


        x = (
            x
            + K @ innovation
        )


        I = np.eye(
            STATE_SIZE
        )


        P = (
            I - K @ H
        ) @ P


    # ========================================================
    # GNSS POSITION UPDATE
    # ========================================================

    in_outage = (

        OUTAGE_START
        <= time_s[i]
        <= OUTAGE_END

    )


    if not in_outage:

        z_gnss = np.array([

            gnss_n[i],
            gnss_e[i]

        ])


        h_gnss = np.array([

            x[0],
            x[1]

        ])


        innovation = (
            z_gnss
            - h_gnss
        )


        H = np.zeros(
            (2, STATE_SIZE)
        )

        H[0, 0] = 1.0
        H[1, 1] = 1.0


        R_gnss = np.diag([

            GNSS_POSITION_STD ** 2,
            GNSS_POSITION_STD ** 2

        ])


        S = (
            H
            @ P
            @ H.T
            + R_gnss
        )


        K = (
            P
            @ H.T
            @ np.linalg.inv(S)
        )


        x = (
            x
            + K @ innovation
        )


        I = np.eye(
            STATE_SIZE
        )


        P = (
            I - K @ H
        ) @ P


    # ========================================================
    # STORE
    # ========================================================

    estimated_n[i] = x[0]
    estimated_e[i] = x[1]

    estimated_vn[i] = x[2]
    estimated_ve[i] = x[3]

    estimated_bax[i] = x[4]
    estimated_bay[i] = x[5]
    estimated_baz[i] = x[6]

    estimated_bgx[i] = x[7]
    estimated_bgy[i] = x[8]
    estimated_bgz[i] = x[9]

    estimated_roll[i] = roll
    estimated_pitch[i] = pitch
    estimated_yaw[i] = yaw


    # ========================================================
    # POSITION ERROR
    # ========================================================

    position_error[i] = np.sqrt(

        (
            estimated_n[i]
            - gnss_n[i]
        ) ** 2

        +

        (
            estimated_e[i]
            - gnss_e[i]
        ) ** 2

    )


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

results = pd.DataFrame({

    "timestamp":
        df["timestamp"],

    "time_s":
        time_s,

    "gnss_n":
        gnss_n,

    "gnss_e":
        gnss_e,

    "estimated_n":
        estimated_n,

    "estimated_e":
        estimated_e,

    "estimated_vn":
        estimated_vn,

    "estimated_ve":
        estimated_ve,

    "ai_vn":
        ai_vn,

    "ai_ve":
        ai_ve,

    "imu_acc_n":
        imu_acc_n,

    "imu_acc_e":
        imu_acc_e,

    "accel_bias_x":
        estimated_bax,

    "accel_bias_y":
        estimated_bay,

    "accel_bias_z":
        estimated_baz,

    "gyro_bias_x":
        estimated_bgx,

    "gyro_bias_y":
        estimated_bgy,

    "gyro_bias_z":
        estimated_bgz,

    "roll_deg":
        np.degrees(estimated_roll),

    "pitch_deg":
        np.degrees(estimated_pitch),

    "yaw_deg":
        np.degrees(estimated_yaw),

    "position_error":
        position_error

})


# ============================================================
# OUTAGE RESULTS
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
        "No outage samples found."
    )


mean_error = (
    outage["position_error"]
    .mean()
)

median_error = (
    outage["position_error"]
    .median()
)

final_error = (
    outage["position_error"]
    .iloc[-1]
)

max_error = (
    outage["position_error"]
    .max()
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_file = os.path.join(

    PROCESSED_DIR,

    "prototype_v9_5_2_imu_ai_ekf_results.csv"

)

results.to_csv(
    result_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "V9.5.2 RESULTS"
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
    f"Outage samples: {len(outage)}"
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
    f"Maximum outage error: "
    f"{max_error:.3f} m"
)


print(
    "\nFinal estimated biases:"
)

print(
    f"Accel: "
    f"[{estimated_bax[-1]:.6f}, "
    f"{estimated_bay[-1]:.6f}, "
    f"{estimated_baz[-1]:.6f}]"
)

print(
    f"Gyro : "
    f"[{estimated_bgx[-1]:.6f}, "
    f"{estimated_bgy[-1]:.6f}, "
    f"{estimated_bgz[-1]:.6f}]"
)


# ============================================================
# TRAJECTORY PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.plot(

    results["gnss_e"],
    results["gnss_n"],

    label="GNSS reference"

)

plt.plot(

    results["estimated_e"],
    results["estimated_n"],

    label="V9.5.2 IMU + AI + EKF"

)

plt.axvspan(
    results["gnss_e"].iloc[0],
    results["gnss_e"].iloc[0],
    alpha=0
)

plt.xlabel(
    "East (m)"
)

plt.ylabel(
    "North (m)"
)

plt.title(
    "V9.5.2 Navigation Trajectory"
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

    "prototype_v9_5_2_imu_ai_ekf_trajectory.png"

)

plt.savefig(
    trajectory_file,
    dpi=200
)

plt.close()


# ============================================================
# ERROR PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(

    results["time_s"],

    results["position_error"],

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
    "V9.5.2 Position Error"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


error_file = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_5_2_imu_ai_ekf_error.png"

)

plt.savefig(
    error_file,
    dpi=200
)

plt.close()


# ============================================================
# BIAS PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    results["time_s"],
    results["accel_bias_x"],
    label="Accel bias X"
)

plt.plot(
    results["time_s"],
    results["accel_bias_y"],
    label="Accel bias Y"
)

plt.plot(
    results["time_s"],
    results["accel_bias_z"],
    label="Accel bias Z"
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
    "V9.5.2 Accelerometer Bias Estimates"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


bias_file = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_5_2_accel_bias.png"

)

plt.savefig(
    bias_file,
    dpi=200
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "metric": [

        "outage_samples",
        "mean_position_error_m",
        "median_position_error_m",
        "final_position_error_m",
        "maximum_position_error_m"

    ],

    "value": [

        len(outage),
        mean_error,
        median_error,
        final_error,
        max_error

    ]

})


summary_file = os.path.join(

    PROCESSED_DIR,

    "prototype_v9_5_2_imu_ai_ekf_summary.csv"

)

summary.to_csv(
    summary_file,
    index=False
)


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
    bias_file
)

print(
    "\nV9.5.2 complete."
)

print(
    "Do not tune parameters yet. Inspect the results first."
)