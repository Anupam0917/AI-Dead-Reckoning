import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V9.5.1
# STABILIZED ORIENTATION + GRAVITY COMPENSATION
#
# Purpose:
#   1. Estimate roll/pitch from accelerometer + gravity
#   2. Integrate gyro for short-term orientation changes
#   3. Use magnetometer for yaw correction
#   4. Remove gravity in navigation frame
#   5. Validate horizontal acceleration
#
# This is a sensor/orientation validation stage.
# It is NOT yet the final IEKF.
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

ANALYSIS_START = 8500.0
ANALYSIS_END = 8560.0

# Complementary filter
GYRO_WEIGHT = 0.98
ACCEL_WEIGHT = 0.02
MAG_WEIGHT = 0.02

GRAVITY = 9.80665

# Magnetometer validity range
MAG_MIN = 5.0
MAG_MAX = 100.0

MAX_DT = 0.5


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
# PARSE TIMESTAMP
# ============================================================

def parse_time(df):

    time_col = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"

    if time_col not in df.columns:

        raise ValueError(
            f"Missing timestamp column: {time_col}"
        )

    df["timestamp"] = pd.to_datetime(
        df[time_col],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    if df["timestamp"].isna().any():

        count = df["timestamp"].isna().sum()

        raise ValueError(
            f"Could not parse {count} timestamps."
        )

    df["time_s"] = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    return df


# ============================================================
# ANGLE UTILITIES
# ============================================================

def wrap_angle(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def angle_difference(
    target,
    current
):

    return wrap_angle(
        target - current
    )


# ============================================================
# GRAVITY-BASED ROLL/PITCH
# ============================================================

def gravity_roll_pitch(
    gx,
    gy,
    gz
):

    roll = np.arctan2(
        gy,
        gz
    )

    pitch = np.arctan2(
        -gx,
        np.sqrt(
            gy ** 2
            + gz ** 2
        )
    )

    return roll, pitch


# ============================================================
# TILT-COMPENSATED MAGNETOMETER HEADING
# ============================================================

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

    heading = np.arctan2(
        -my_level,
        mx_level
    )

    return wrap_angle(
        heading
    )


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
# FIND MAGNETOMETER COLUMNS ROBUSTLY
# ============================================================

def find_magnetometer_columns(df):

    found = {}

    for column in df.columns:

        name = str(column).strip()

        # Remove spaces and parentheses/unit symbols so that
        # mojibake in µT does not matter.

        normalized = (
            name
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
            "\nCould not automatically identify all "
            "magnetometer columns."
        )

        print(
            "\nAll columns containing 'MAGNETIC':"
        )

        for column in df.columns:

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


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("\n" + "=" * 70)
print("V9.5.1 ORIENTATION + GRAVITY COMPENSATION")
print("=" * 70)

print("\nLoading raw dataset...")

df = load_csv(
    RAW_FILE
)

df = parse_time(
    df
)

print(
    f"Rows: {len(df)}"
)


# ============================================================
# 2. REQUIRED NON-MAGNETOMETER COLUMNS
# ============================================================

required_columns = [

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


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# 3. FIND MAGNETOMETER COLUMNS
# ============================================================

(
    MAG_X_COLUMN,
    MAG_Y_COLUMN,
    MAG_Z_COLUMN
) = find_magnetometer_columns(
    df
)


print(
    "\nDetected magnetometer columns:"
)

print(
    "X:",
    repr(MAG_X_COLUMN)
)

print(
    "Y:",
    repr(MAG_Y_COLUMN)
)

print(
    "Z:",
    repr(MAG_Z_COLUMN)
)


# ============================================================
# 4. LOAD BIAS FILE
# ============================================================

print(
    "\nLoading IMU bias estimates..."
)

bias_df = pd.read_csv(
    BIAS_FILE
)

required_bias_columns = [
    "sensor",
    "bias",
    "noise_std"
]

for column in required_bias_columns:

    if column not in bias_df.columns:

        raise ValueError(
            f"Bias file missing column: {column}"
        )


bias_dict = dict(
    zip(
        bias_df["sensor"],
        bias_df["bias"]
    )
)


print(
    "\nBias values:"
)

for sensor, value in bias_dict.items():

    print(
        f"  {sensor}: {value:.6f}"
    )


# ============================================================
# 5. EXTRACT ACCELEROMETER
# ============================================================

ax = df[
    "ACCELEROMETER X (m/s²)"
].to_numpy(
    dtype=float
)

ay = df[
    "ACCELEROMETER Y (m/s²)"
].to_numpy(
    dtype=float
)

az = df[
    "ACCELEROMETER Z (m/s²)"
].to_numpy(
    dtype=float
)


# ============================================================
# 6. EXTRACT GYROSCOPE
# ============================================================

gx = df[
    "GYROSCOPE Roll (rad/s)"
].to_numpy(
    dtype=float
)

gy = df[
    "GYROSCOPE Pitch (rad/s)"
].to_numpy(
    dtype=float
)

gz = df[
    "GYROSCOPE Yaw (rad/s)"
].to_numpy(
    dtype=float
)


# ============================================================
# 7. EXTRACT MAGNETOMETER
# ============================================================

mx = df[
    MAG_X_COLUMN
].to_numpy(
    dtype=float
)

my = df[
    MAG_Y_COLUMN
].to_numpy(
    dtype=float
)

mz = df[
    MAG_Z_COLUMN
].to_numpy(
    dtype=float
)


# ============================================================
# 8. APPLY INITIAL IMU BIAS
# ============================================================

ax = ax - bias_dict.get(
    "accelerometer_x",
    0.0
)

ay = ay - bias_dict.get(
    "accelerometer_y",
    0.0
)

az = az - bias_dict.get(
    "accelerometer_z",
    0.0
)


gx = gx - bias_dict.get(
    "gyroscope_x",
    0.0
)

gy = gy - bias_dict.get(
    "gyroscope_y",
    0.0
)

gz = gz - bias_dict.get(
    "gyroscope_z",
    0.0
)


# ============================================================
# 9. SENSOR MAGNITUDES
# ============================================================

accel_magnitude = np.sqrt(

    ax ** 2
    + ay ** 2
    + az ** 2

)


mag_magnitude = np.sqrt(

    mx ** 2
    + my ** 2
    + mz ** 2

)


# ============================================================
# 10. INITIAL ORIENTATION
# ============================================================

roll0, pitch0 = gravity_roll_pitch(

    ax[0],
    ay[0],
    az[0]

)


yaw0 = magnetometer_heading(

    mx[0],
    my[0],
    mz[0],

    roll0,
    pitch0

)


print(
    "\nInitial orientation:"
)

print(
    f"Roll : {np.degrees(roll0):.3f}°"
)

print(
    f"Pitch: {np.degrees(pitch0):.3f}°"
)

print(
    f"Yaw  : {np.degrees(yaw0):.3f}°"
)


# ============================================================
# 11. STORAGE
# ============================================================

n = len(df)


roll_est = np.zeros(n)
pitch_est = np.zeros(n)
yaw_est = np.zeros(n)

roll_acc = np.zeros(n)
pitch_acc = np.zeros(n)
yaw_mag = np.zeros(n)

linear_acc_n = np.zeros(n)
linear_acc_e = np.zeros(n)
linear_acc_down = np.zeros(n)

gravity_error = np.zeros(n)

mag_valid = np.zeros(
    n,
    dtype=bool
)


# ============================================================
# 12. INITIAL VALUES
# ============================================================

roll_est[0] = roll0
pitch_est[0] = pitch0
yaw_est[0] = yaw0

roll_acc[0] = roll0
pitch_acc[0] = pitch0
yaw_mag[0] = yaw0


# ============================================================
# 13. ORIENTATION ESTIMATION
# ============================================================

print(
    "\nRunning orientation estimator..."
)


previous_time = df[
    "time_s"
].iloc[0]


for i in range(
    1,
    n
):

    current_time = df[
        "time_s"
    ].iloc[i]


    dt = (
        current_time
        - previous_time
    )


    if (
        dt <= 0
        or dt > MAX_DT
    ):

        dt = 0.1


    previous_time = current_time


    # --------------------------------------------------------
    # GRAVITY-BASED ROLL/PITCH
    # --------------------------------------------------------

    r_acc, p_acc = gravity_roll_pitch(

        ax[i],
        ay[i],
        az[i]

    )


    roll_acc[i] = r_acc
    pitch_acc[i] = p_acc


    # --------------------------------------------------------
    # GYRO PROPAGATION
    # --------------------------------------------------------

    roll_gyro = (

        roll_est[i - 1]
        + gx[i] * dt

    )


    pitch_gyro = (

        pitch_est[i - 1]
        + gy[i] * dt

    )


    yaw_gyro = (

        yaw_est[i - 1]
        + gz[i] * dt

    )


    # --------------------------------------------------------
    # COMPLEMENTARY ROLL
    # --------------------------------------------------------

    roll_est[i] = (

        GYRO_WEIGHT
        * roll_gyro

        +

        ACCEL_WEIGHT
        * r_acc

    )


    # --------------------------------------------------------
    # COMPLEMENTARY PITCH
    # --------------------------------------------------------

    pitch_est[i] = (

        GYRO_WEIGHT
        * pitch_gyro

        +

        ACCEL_WEIGHT
        * p_acc

    )


    # --------------------------------------------------------
    # MAGNETOMETER VALIDITY
    # --------------------------------------------------------

    valid_mag = (

        mag_magnitude[i]
        >= MAG_MIN

        and

        mag_magnitude[i]
        <= MAG_MAX

    )


    mag_valid[i] = valid_mag


    # --------------------------------------------------------
    # YAW
    # --------------------------------------------------------

    if valid_mag:

        yaw_m = magnetometer_heading(

            mx[i],
            my[i],
            mz[i],

            roll_est[i],
            pitch_est[i]

        )

        yaw_mag[i] = yaw_m


        yaw_error = angle_difference(

            yaw_m,
            yaw_gyro

        )


        yaw_est[i] = (

            yaw_gyro

            +

            MAG_WEIGHT
            * yaw_error

        )

    else:

        yaw_est[i] = yaw_gyro


    yaw_est[i] = wrap_angle(
        yaw_est[i]
    )


    # ========================================================
    # BODY ACCELERATION
    # ========================================================

    acc_body = np.array([

        ax[i],
        ay[i],
        az[i]

    ])


    # ========================================================
    # BODY -> NAVIGATION FRAME
    # ========================================================

    R_bn = rotation_matrix(

        roll_est[i],
        pitch_est[i],
        yaw_est[i]

    )


    acc_nav = (

        R_bn
        @ acc_body

    )


    # ========================================================
    # GRAVITY COMPENSATION
    # ========================================================

    gravity_vector = np.array([

        0.0,
        0.0,
        GRAVITY

    ])


    linear_nav = (

        acc_nav
        - gravity_vector

    )


    linear_acc_n[i] = linear_nav[0]
    linear_acc_e[i] = linear_nav[1]
    linear_acc_down[i] = linear_nav[2]


    # ========================================================
    # GRAVITY MAGNITUDE ERROR
    # ========================================================

    estimated_gravity_magnitude = np.linalg.norm(
        acc_nav
    )


    gravity_error[i] = (

        estimated_gravity_magnitude
        - GRAVITY

    )


# ============================================================
# 14. RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame({

    "timestamp":
        df["timestamp"],

    "time_s":
        df["time_s"],

    "roll_deg":
        np.degrees(roll_est),

    "pitch_deg":
        np.degrees(pitch_est),

    "yaw_deg":
        np.degrees(yaw_est),

    "roll_acc_deg":
        np.degrees(roll_acc),

    "pitch_acc_deg":
        np.degrees(pitch_acc),

    "yaw_mag_deg":
        np.degrees(yaw_mag),

    "accel_magnitude":
        accel_magnitude,

    "mag_magnitude":
        mag_magnitude,

    "linear_acc_n":
        linear_acc_n,

    "linear_acc_e":
        linear_acc_e,

    "linear_acc_down":
        linear_acc_down,

    "gravity_error":
        gravity_error,

    "mag_valid":
        mag_valid

})


# ============================================================
# 15. SAVE RESULTS
# ============================================================

result_file = os.path.join(

    PROCESSED_DIR,

    "prototype_v9_5_1_orientation_results.csv"

)


results.to_csv(

    result_file,

    index=False

)


# ============================================================
# 16. ANALYSIS WINDOW
# ============================================================

outage = results[

    (
        results["time_s"]
        >= ANALYSIS_START
    )

    &

    (
        results["time_s"]
        <= ANALYSIS_END
    )

].copy()


if len(outage) == 0:

    raise RuntimeError(
        "No samples found in analysis window."
    )


# ============================================================
# 17. HORIZONTAL ACCELERATION
# ============================================================

horizontal_accel = np.sqrt(

    outage["linear_acc_n"] ** 2

    +

    outage["linear_acc_e"] ** 2

)


# ============================================================
# 18. MAGNETOMETER VALIDITY
# ============================================================

mag_valid_fraction = (

    outage["mag_valid"].mean()
    * 100.0

)


# ============================================================
# 19. PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "V9.5.1 ORIENTATION RESULTS"
)

print(
    "=" * 70
)

print(
    f"Analysis window: "
    f"{ANALYSIS_START:.0f}s -> "
    f"{ANALYSIS_END:.0f}s"
)

print(
    f"Samples: {len(outage)}"
)


print(
    "\nOrientation statistics:"
)

print(
    f"Roll mean   : "
    f"{outage['roll_deg'].mean():.3f}°"
)

print(
    f"Roll std    : "
    f"{outage['roll_deg'].std():.3f}°"
)

print(
    f"Pitch mean  : "
    f"{outage['pitch_deg'].mean():.3f}°"
)

print(
    f"Pitch std   : "
    f"{outage['pitch_deg'].std():.3f}°"
)

print(
    f"Yaw mean    : "
    f"{outage['yaw_deg'].mean():.3f}°"
)

print(
    f"Yaw std     : "
    f"{outage['yaw_deg'].std():.3f}°"
)


print(
    "\nGravity diagnostic:"
)

print(
    f"Mean gravity magnitude error: "
    f"{outage['gravity_error'].mean():.5f} m/s²"
)

print(
    f"Median gravity magnitude error: "
    f"{outage['gravity_error'].median():.5f} m/s²"
)

print(
    f"Std gravity magnitude error: "
    f"{outage['gravity_error'].std():.5f} m/s²"
)


print(
    "\nHorizontal acceleration:"
)

print(
    f"Mean magnitude: "
    f"{horizontal_accel.mean():.4f} m/s²"
)

print(
    f"Median magnitude: "
    f"{horizontal_accel.median():.4f} m/s²"
)

print(
    f"95th percentile: "
    f"{horizontal_accel.quantile(0.95):.4f} m/s²"
)

print(
    f"Maximum: "
    f"{horizontal_accel.max():.4f} m/s²"
)


print(
    "\nMagnetometer validity:"
)

print(
    f"Valid samples: "
    f"{mag_valid_fraction:.2f}%"
)


# ============================================================
# 20. ORIENTATION PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(
    results["time_s"],
    results["roll_deg"],
    label="Roll"
)

plt.plot(
    results["time_s"],
    results["pitch_deg"],
    label="Pitch"
)

plt.plot(
    results["time_s"],
    results["yaw_deg"],
    label="Yaw"
)

plt.axvspan(
    ANALYSIS_START,
    ANALYSIS_END,
    alpha=0.2,
    label="Analysis window"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Angle (degrees)"
)

plt.title(
    "V9.5.1 Estimated Orientation"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


orientation_plot = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_5_1_orientation.png"

)


plt.savefig(

    orientation_plot,

    dpi=200

)

plt.close()


# ============================================================
# 21. GRAVITY ERROR PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(

    results["time_s"],

    results["gravity_error"],

    label="Gravity magnitude error"

)

plt.axhline(

    0.0,

    linestyle="--"

)

plt.axvspan(

    ANALYSIS_START,

    ANALYSIS_END,

    alpha=0.2,

    label="Analysis window"

)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Error (m/s²)"
)

plt.title(
    "V9.5.1 Gravity Compensation Diagnostic"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


gravity_plot = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_5_1_gravity_error.png"

)


plt.savefig(

    gravity_plot,

    dpi=200

)

plt.close()


# ============================================================
# 22. HORIZONTAL ACCELERATION PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.plot(

    results["time_s"],

    results["linear_acc_n"],

    label="North acceleration"

)

plt.plot(

    results["time_s"],

    results["linear_acc_e"],

    label="East acceleration"

)

plt.axhline(

    0.0,

    linestyle="--"

)

plt.axvspan(

    ANALYSIS_START,

    ANALYSIS_END,

    alpha=0.2,

    label="Analysis window"

)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Linear acceleration (m/s²)"
)

plt.title(
    "V9.5.1 Horizontal Acceleration"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


accel_plot = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_5_1_horizontal_acceleration.png"

)


plt.savefig(

    accel_plot,

    dpi=200

)

plt.close()


# ============================================================
# 23. MAGNETOMETER PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(

    results["time_s"],

    results["mag_magnitude"],

    label="Magnetic field magnitude"

)

plt.axhline(

    MAG_MIN,

    linestyle="--",

    label="Minimum threshold"

)

plt.axhline(

    MAG_MAX,

    linestyle="--",

    label="Maximum threshold"

)

plt.axvspan(

    ANALYSIS_START,

    ANALYSIS_END,

    alpha=0.2,

    label="Analysis window"

)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Magnetic field magnitude (µT)"
)

plt.title(
    "V9.5.1 Magnetometer Diagnostic"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


mag_plot = os.path.join(

    OUTPUT_DIR,

    "prototype_v9_5_1_magnetometer.png"

)


plt.savefig(

    mag_plot,

    dpi=200

)

plt.close()


# ============================================================
# 24. SUMMARY CSV
# ============================================================

summary_file = os.path.join(

    PROCESSED_DIR,

    "prototype_v9_5_1_orientation_summary.csv"

)


summary = pd.DataFrame({

    "metric": [

        "roll_mean_deg",
        "roll_std_deg",

        "pitch_mean_deg",
        "pitch_std_deg",

        "yaw_mean_deg",
        "yaw_std_deg",

        "gravity_error_mean_mps2",
        "gravity_error_median_mps2",
        "gravity_error_std_mps2",

        "horizontal_accel_mean_mps2",
        "horizontal_accel_median_mps2",
        "horizontal_accel_p95_mps2",
        "horizontal_accel_max_mps2",

        "mag_valid_percent"

    ],

    "value": [

        outage["roll_deg"].mean(),
        outage["roll_deg"].std(),

        outage["pitch_deg"].mean(),
        outage["pitch_deg"].std(),

        outage["yaw_deg"].mean(),
        outage["yaw_deg"].std(),

        outage["gravity_error"].mean(),
        outage["gravity_error"].median(),
        outage["gravity_error"].std(),

        horizontal_accel.mean(),
        horizontal_accel.median(),
        horizontal_accel.quantile(0.95),
        horizontal_accel.max(),

        mag_valid_fraction

    ]

})


summary.to_csv(

    summary_file,

    index=False

)


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
    summary_file
)

print(
    orientation_plot
)

print(
    gravity_plot
)

print(
    accel_plot
)

print(
    mag_plot
)


print(
    "\nV9.5.1 complete."
)

print(
    "Next step will be selected from the diagnostics."
)