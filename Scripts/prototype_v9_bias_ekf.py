import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V9.4 STEP 2
# BIAS-AWARE ADAPTIVE EKF
#
# State:
# [North, East, Vnorth, Veast,
#  AccelBiasX, AccelBiasY, AccelBiasZ,
#  GyroBiasX, GyroBiasY, GyroBiasZ]
#
# The current V9 AI model provides Vnorth / Veast.
# GNSS position is used when available.
# During simulated GNSS outage, the filter continues
# using AI velocity.
# ============================================================

print("=" * 70)
print("V9.4 BIAS-AWARE ADAPTIVE EKF")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

AI_PATH = (
    "data/processed/"
    "prototype_v9_ai_velocity_results.csv"
)

BIAS_PATH = (
    "data/processed/"
    "prototype_v9_imu_bias.csv"
)

TARGET_PATH = (
    "data/processed/"
    "prototype_v9_velocity_targets.csv"
)

OUTPUT_DIR = "outputs"
PROCESSED_DIR = "data/processed"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# LOAD AI VELOCITY RESULTS
# ------------------------------------------------------------

print("\nLoading AI velocity predictions...")

df = pd.read_csv(
    AI_PATH
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print(
    "AI test samples:",
    len(df)
)


# ------------------------------------------------------------
# LOAD GLOBAL TIME
# ------------------------------------------------------------

print("\nLoading global timeline...")

timeline = pd.read_csv(
    TARGET_PATH,
    usecols=[
        "timestamp",
        "time_s"
    ]
)

timeline["timestamp"] = pd.to_datetime(
    timeline["timestamp"]
)


df = pd.merge_asof(
    df.sort_values("timestamp"),
    timeline.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("150ms")
)


df = df.dropna(
    subset=["time_s"]
).reset_index(drop=True)


print(
    "Global test period:",
    f"{df['time_s'].iloc[0]:.2f}s"
    " -> "
    f"{df['time_s'].iloc[-1]:.2f}s"
)


# ------------------------------------------------------------
# LOAD BIAS INITIALIZATION
# ------------------------------------------------------------

print("\nLoading IMU bias estimates...")

bias_df = pd.read_csv(
    BIAS_PATH
)

bias_map = dict(
    zip(
        bias_df["sensor"],
        bias_df["bias"]
    )
)


# Initial accelerometer bias estimate
initial_accel_bias = np.array([
    bias_map["accelerometer_x"],
    bias_map["accelerometer_y"],
    bias_map["accelerometer_z"]
])


# Initial gyroscope bias estimate
initial_gyro_bias = np.array([
    bias_map["gyroscope_x"],
    bias_map["gyroscope_y"],
    bias_map["gyroscope_z"]
])


print(
    "\nInitial accelerometer bias:",
    initial_accel_bias
)

print(
    "Initial gyroscope bias:",
    initial_gyro_bias
)


# ------------------------------------------------------------
# TIME STEP
# ------------------------------------------------------------

df["dt"] = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
)

df["dt"] = (
    df["dt"]
    .clip(
        lower=0.01,
        upper=0.5
    )
)

df["dt"] = (
    df["dt"]
    .fillna(0.1)
)


# ------------------------------------------------------------
# AI VELOCITY
# ------------------------------------------------------------

ai_vn = (
    df["vnorth_ai_mps"]
    .values
)

ai_ve = (
    df["veast_ai_mps"]
    .values
)


# GNSS-derived reference velocity
true_vn = (
    df["vnorth_true_mps"]
    .values
)

true_ve = (
    df["veast_true_mps"]
    .values
)


# ------------------------------------------------------------
# EKF IMPLEMENTATION
# ------------------------------------------------------------

class BiasAwareEKF:

    def __init__(
        self,
        initial_accel_bias,
        initial_gyro_bias
    ):

        # ----------------------------------------------------
        # STATE
        #
        # [N, E, Vn, Ve,
        #  bax, bay, baz,
        #  bgx, bgy, bgz]
        # ----------------------------------------------------

        self.x = np.zeros(10)

        self.x[4:7] = (
            initial_accel_bias
        )

        self.x[7:10] = (
            initial_gyro_bias
        )


        # ----------------------------------------------------
        # INITIAL COVARIANCE
        # ----------------------------------------------------

        self.P = np.eye(10)

        # Position uncertainty
        self.P[0, 0] = 10.0
        self.P[1, 1] = 10.0

        # Velocity uncertainty
        self.P[2, 2] = 1.0
        self.P[3, 3] = 1.0

        # Accelerometer bias uncertainty
        self.P[4, 4] = 0.05
        self.P[5, 5] = 0.05
        self.P[6, 6] = 0.05

        # Gyroscope bias uncertainty
        self.P[7, 7] = 0.01
        self.P[8, 8] = 0.01
        self.P[9, 9] = 0.01


        # ----------------------------------------------------
        # BASE PROCESS NOISE
        # ----------------------------------------------------

        self.Q = np.eye(10) * 0.001

        self.Q[0, 0] = 0.01
        self.Q[1, 1] = 0.01

        self.Q[2, 2] = 0.05
        self.Q[3, 3] = 0.05

        # Bias random walk
        self.Q[4, 4] = 0.0001
        self.Q[5, 5] = 0.0001
        self.Q[6, 6] = 0.0001

        self.Q[7, 7] = 0.00001
        self.Q[8, 8] = 0.00001
        self.Q[9, 9] = 0.00001


        # ----------------------------------------------------
        # VELOCITY MEASUREMENT NOISE
        # ----------------------------------------------------

        self.R_velocity = np.eye(2) * 1.0

        # GNSS position measurement noise
        self.R_gnss = np.eye(2) * 4.0


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(
        self,
        vn_measurement,
        ve_measurement,
        dt,
        confidence=1.0
    ):

        # ----------------------------------------------------
        # AI velocity acts as a velocity measurement.
        #
        # We propagate position using AI velocity.
        # ----------------------------------------------------

        self.x[0] += (
            self.x[2] * dt
        )

        self.x[1] += (
            self.x[3] * dt
        )


        # ----------------------------------------------------
        # VELOCITY MODEL
        #
        # AI velocity becomes the prediction.
        # ----------------------------------------------------

        self.x[2] = (
            vn_measurement
        )

        self.x[3] = (
            ve_measurement
        )


        # ----------------------------------------------------
        # BIAS RANDOM WALK
        # ----------------------------------------------------

        F = np.eye(10)

        F[0, 2] = dt
        F[1, 3] = dt


        # ----------------------------------------------------
        # ADAPTIVE PROCESS NOISE
        #
        # Lower confidence → larger uncertainty.
        # ----------------------------------------------------

        confidence = np.clip(
            confidence,
            0.1,
            1.0
        )


        adaptive_factor = (
            1.0 /
            confidence
        )


        Q_adaptive = (
            self.Q *
            adaptive_factor
        )


        # ----------------------------------------------------
        # COVARIANCE PROPAGATION
        # ----------------------------------------------------

        self.P = (
            F
            @ self.P
            @ F.T
            +
            Q_adaptive * dt
        )


    # --------------------------------------------------------
    # AI VELOCITY UPDATE
    # --------------------------------------------------------

    def update_velocity(
        self,
        vn,
        ve,
        confidence=1.0
    ):

        z = np.array([
            vn,
            ve
        ])


        H = np.zeros(
            (2, 10)
        )

        H[0, 2] = 1.0
        H[1, 3] = 1.0


        # Lower confidence means larger measurement noise.

        confidence = np.clip(
            confidence,
            0.1,
            1.0
        )


        R = (
            self.R_velocity
            /
            confidence
        )


        y = (
            z -
            H @ self.x
        )


        S = (
            H
            @ self.P
            @ H.T
            +
            R
        )


        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )


        self.x = (
            self.x
            +
            K @ y
        )


        I = np.eye(
            len(self.x)
        )


        self.P = (
            (I - K @ H)
            @ self.P
        )


    # --------------------------------------------------------
    # GNSS POSITION UPDATE
    # --------------------------------------------------------

    def update_gnss(
        self,
        north,
        east
    ):

        z = np.array([
            north,
            east
        ])


        H = np.zeros(
            (2, 10)
        )

        H[0, 0] = 1.0
        H[1, 1] = 1.0


        y = (
            z -
            H @ self.x
        )


        S = (
            H
            @ self.P
            @ H.T
            +
            self.R_gnss
        )


        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )


        self.x = (
            self.x
            +
            K @ y
        )


        I = np.eye(
            len(self.x)
        )


        self.P = (
            (I - K @ H)
            @ self.P
        )


# ------------------------------------------------------------
# LOCAL GNSS REFERENCE
# ------------------------------------------------------------

# We don't have latitude/longitude in the AI result file,
# so use integrated GNSS-derived velocity as a consistent
# local reference for this experiment.


reference_n = np.zeros(
    len(df)
)

reference_e = np.zeros(
    len(df)
)


for i in range(1, len(df)):

    dt = df["dt"].iloc[i]

    reference_n[i] = (
        reference_n[i - 1]
        + true_vn[i] * dt
    )

    reference_e[i] = (
        reference_e[i - 1]
        + true_ve[i] * dt
    )


# ------------------------------------------------------------
# RUN FILTER
# ------------------------------------------------------------

print("\nRunning bias-aware EKF...")


ekf = BiasAwareEKF(
    initial_accel_bias,
    initial_gyro_bias
)


estimated_n = np.zeros(
    len(df)
)

estimated_e = np.zeros(
    len(df)
)

estimated_vn = np.zeros(
    len(df)
)

estimated_ve = np.zeros(
    len(df)
)

estimated_accel_bias = np.zeros(
    (len(df), 3)
)

estimated_gyro_bias = np.zeros(
    (len(df), 3)
)


# ------------------------------------------------------------
# SIMULATED GNSS AVAILABILITY
# ------------------------------------------------------------

# We simulate a GNSS outage around 8500s.
#
# Before outage:
#   GNSS available
#
# During outage:
#   GNSS unavailable
#
# After outage:
#   GNSS returns


OUTAGE_START = 8500

OUTAGE_END = 8560


# ------------------------------------------------------------
# MAIN FILTER LOOP
# ------------------------------------------------------------

for i in range(
    len(df)
):

    dt = df["dt"].iloc[i]


    # --------------------------------------------------------
    # AI confidence
    #
    # Initial implementation uses fixed confidence.
    # V9.6 will replace this with learned reliability.
    # --------------------------------------------------------

    confidence = 0.8


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    ekf.predict(
        ai_vn[i],
        ai_ve[i],
        dt,
        confidence
    )


    # --------------------------------------------------------
    # AI VELOCITY UPDATE
    # --------------------------------------------------------

    ekf.update_velocity(
        ai_vn[i],
        ai_ve[i],
        confidence
    )


    # --------------------------------------------------------
    # GNSS UPDATE
    # --------------------------------------------------------

    current_time = (
        df["time_s"].iloc[i]
    )


    gnss_available = not (
        OUTAGE_START
        <= current_time
        <= OUTAGE_END
    )


    if gnss_available:

        ekf.update_gnss(
            reference_n[i],
            reference_e[i]
        )


    # --------------------------------------------------------
    # SAVE STATE
    # --------------------------------------------------------

    estimated_n[i] = (
        ekf.x[0]
    )

    estimated_e[i] = (
        ekf.x[1]
    )

    estimated_vn[i] = (
        ekf.x[2]
    )

    estimated_ve[i] = (
        ekf.x[3]
    )


    estimated_accel_bias[i] = (
        ekf.x[4:7]
    )

    estimated_gyro_bias[i] = (
        ekf.x[7:10]
    )


# ------------------------------------------------------------
# ERROR
# ------------------------------------------------------------

position_error = np.sqrt(
    (
        estimated_n
        -
        reference_n
    ) ** 2
    +
    (
        estimated_e
        -
        reference_e
    ) ** 2
)


# ------------------------------------------------------------
# OUTAGE RESULTS
# ------------------------------------------------------------

outage_mask = (
    (df["time_s"] >= OUTAGE_START)
    &
    (df["time_s"] <= OUTAGE_END)
)


outage_indices = np.where(
    outage_mask
)[0]


print("\n" + "=" * 70)
print("V9.4 EKF RESULTS")
print("=" * 70)


if len(outage_indices) > 0:

    outage_error = (
        position_error[
            outage_indices
        ]
    )


    print(
        "\nGNSS outage:",
        f"{OUTAGE_START}s -> "
        f"{OUTAGE_END}s"
    )

    print(
        "Outage samples:",
        len(outage_indices)
    )

    print(
        "\nMean position error:",
        f"{np.mean(outage_error):.3f} m"
    )

    print(
        "Final outage error:",
        f"{outage_error[-1]:.3f} m"
    )

    print(
        "Maximum outage error:",
        f"{np.max(outage_error):.3f} m"
    )


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results = pd.DataFrame({

    "timestamp":
        df["timestamp"],

    "time_s":
        df["time_s"],

    "gnss_available":
        ~outage_mask,

    "reference_north_m":
        reference_n,

    "reference_east_m":
        reference_e,

    "ekf_north_m":
        estimated_n,

    "ekf_east_m":
        estimated_e,

    "reference_vnorth_mps":
        true_vn,

    "reference_veast_mps":
        true_ve,

    "ekf_vnorth_mps":
        estimated_vn,

    "ekf_veast_mps":
        estimated_ve,

    "position_error_m":
        position_error,

    "accel_bias_x":
        estimated_accel_bias[:, 0],

    "accel_bias_y":
        estimated_accel_bias[:, 1],

    "accel_bias_z":
        estimated_accel_bias[:, 2],

    "gyro_bias_x":
        estimated_gyro_bias[:, 0],

    "gyro_bias_y":
        estimated_gyro_bias[:, 1],

    "gyro_bias_z":
        estimated_gyro_bias[:, 2]
})


output_csv = (
    "data/processed/"
    "prototype_v9_bias_ekf_results.csv"
)


results.to_csv(
    output_csv,
    index=False
)


# ------------------------------------------------------------
# PLOT 1
# TRAJECTORY
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 8)
)


plt.plot(
    reference_e,
    reference_n,
    label="Reference"
)


plt.plot(
    estimated_e,
    estimated_n,
    label="Bias-aware EKF"
)


plt.xlabel(
    "East (m)"
)

plt.ylabel(
    "North (m)"
)

plt.title(
    "V9.4 Bias-Aware EKF Trajectory"
)

plt.legend()

plt.axis(
    "equal"
)

plt.grid(
    True
)

plt.tight_layout()


trajectory_path = (
    "outputs/"
    "prototype_v9_bias_ekf_trajectory.png"
)


plt.savefig(
    trajectory_path,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# PLOT 2
# POSITION ERROR
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)


plt.plot(
    df["time_s"],
    position_error,
    label="Position error"
)


plt.axvspan(
    OUTAGE_START,
    OUTAGE_END,
    alpha=0.25,
    label="GNSS outage"
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Position error (m)"
)

plt.title(
    "V9.4 Bias-Aware EKF Position Error"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


error_path = (
    "outputs/"
    "prototype_v9_bias_ekf_error.png"
)


plt.savefig(
    error_path,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# PLOT 3
# ACCELEROMETER BIAS
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)


plt.plot(
    df["time_s"],
    estimated_accel_bias[:, 0],
    label="Accel bias X"
)

plt.plot(
    df["time_s"],
    estimated_accel_bias[:, 1],
    label="Accel bias Y"
)

plt.plot(
    df["time_s"],
    estimated_accel_bias[:, 2],
    label="Accel bias Z"
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Bias (m/s²)"
)

plt.title(
    "V9.4 Estimated Accelerometer Bias"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


accel_bias_path = (
    "outputs/"
    "prototype_v9_accel_bias_estimation.png"
)


plt.savefig(
    accel_bias_path,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# PLOT 4
# GYROSCOPE BIAS
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)


plt.plot(
    df["time_s"],
    estimated_gyro_bias[:, 0],
    label="Gyro bias X"
)

plt.plot(
    df["time_s"],
    estimated_gyro_bias[:, 1],
    label="Gyro bias Y"
)

plt.plot(
    df["time_s"],
    estimated_gyro_bias[:, 2],
    label="Gyro bias Z"
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Bias (rad/s)"
)

plt.title(
    "V9.4 Estimated Gyroscope Bias"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


gyro_bias_path = (
    "outputs/"
    "prototype_v9_gyro_bias_estimation.png"
)


plt.savefig(
    gyro_bias_path,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V9.4 STEP 2 COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(
    "1.",
    output_csv
)

print(
    "2.",
    trajectory_path
)

print(
    "3.",
    error_path
)

print(
    "4.",
    accel_bias_path
)

print(
    "5.",
    gyro_bias_path
)

print("\nPipeline:")

print(
    "IMU bias initialization"
    " -> AI velocity"
    " -> bias-aware EKF"
    " -> GNSS correction"
)

print("\nSimulated outage:")

print(
    f"{OUTAGE_START}s -> "
    f"{OUTAGE_END}s"
)

print("=" * 70)