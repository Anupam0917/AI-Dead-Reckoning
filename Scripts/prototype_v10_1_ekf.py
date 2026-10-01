"""
V10.1 - AI Velocity + IMU EKF

State:
    [north, east, vn, ve]

Prediction:
    IMU linear acceleration

Measurements:
    AI North/East velocity
    GNSS North/East position

GNSS outage:
    8500 -> 8560 seconds

Purpose:
    Establish a clean EKF baseline before implementing
    the full invariant EKF formulation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    / "prototype_v10_1_full_ai_velocity.csv"
)

REFERENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "canonical_gnss_reference.csv"
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


# ------------------------------------------------------------
# Process noise
# ------------------------------------------------------------

ACCELERATION_NOISE = 0.8


# ------------------------------------------------------------
# GNSS position measurement
# ------------------------------------------------------------

GNSS_STD = 5.0


# ------------------------------------------------------------
# AI velocity measurement
#
# These values are intentionally based on the observed
# model uncertainty rather than an arbitrary high trust.
# ------------------------------------------------------------

AI_STD_N_FLOOR = 0.50

AI_STD_E_FLOOR = 0.50


# ============================================================
# LOAD RAW DATA
# ============================================================

print("=" * 80)
print("V10.1 AI VELOCITY + IMU EKF")
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


for column in [
    ACC_X,
    ACC_Y,
    ACC_Z,
    GRAV_X,
    GRAV_Y,
    GRAV_Z,
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# LOAD AI VELOCITY
# ============================================================

print(
    "\nLoading full AI velocity..."
)

ai = pd.read_csv(
    AI_FILE
)

ai["timestamp"] = pd.to_datetime(
    ai["timestamp"],
    errors="coerce",
)

print(
    "AI rows:",
    len(ai)
)


# ============================================================
# LOAD GNSS REFERENCE
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

reference = reference[
    [
        "timestamp",
        "north_m",
        "east_m",
    ]
].copy()


# ============================================================
# MERGE AI
# ============================================================

df = pd.merge_asof(
    df.sort_values(
        "timestamp"
    ),
    ai[
        [
            "timestamp",
            "vn_ai_mps",
            "ve_ai_mps",
            "vn_std_mps",
            "ve_std_mps",
        ]
    ].sort_values(
        "timestamp"
    ),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(
        milliseconds=60
    ),
)


# ============================================================
# MERGE GNSS
# ============================================================

df = pd.merge_asof(
    df.sort_values(
        "timestamp"
    ),
    reference.sort_values(
        "timestamp"
    ),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(
        milliseconds=60
    ),
)


df = df.dropna(
    subset=[
        "vn_ai_mps",
        "ve_ai_mps",
        "vn_std_mps",
        "ve_std_mps",
        "north_m",
        "east_m",
    ]
).reset_index(
    drop=True
)


print(
    "Usable rows:",
    len(df)
)


# ============================================================
# INITIAL STATE
# ============================================================

first = df.iloc[0]


# State:
#
# x[0] = north
# x[1] = east
# x[2] = north velocity
# x[3] = east velocity


x = np.array(
    [
        float(first["north_m"]),
        float(first["east_m"]),
        float(first["vn_ai_mps"]),
        float(first["ve_ai_mps"]),
    ],
    dtype=float,
)


# ============================================================
# INITIAL COVARIANCE
# ============================================================

P = np.diag(
    [
        5.0 ** 2,
        5.0 ** 2,
        2.0 ** 2,
        2.0 ** 2,
    ]
)


# ============================================================
# RESULTS
# ============================================================

results = []


previous_time = float(
    df["time_s"].iloc[0]
)


# ============================================================
# MAIN EKF LOOP
# ============================================================

print(
    "\nRunning EKF..."
)


for _, row in df.iterrows():

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
    # Protect against timestamp gaps
    # --------------------------------------------------------

    if (
        dt <= 0
        or dt > 0.5
    ):
        dt = 0.1


    # ========================================================
    # LINEAR ACCELERATION
    # ========================================================

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
        row[GRAV_X]
    )

    gy = float(
        row[GRAV_Y]
    )

    gz = float(
        row[GRAV_Z]
    )


    # Gravity compensation.

    lax = ax - gx
    lay = ay - gy
    laz = az - gz


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We don't know reliable phone-to-vehicle heading yet.
    #
    # Therefore use the sensor-frame horizontal acceleration
    # only as a small prediction term.
    #
    # AI velocity is responsible for correcting the velocity.
    # --------------------------------------------------------

    accel_n = lax

    accel_e = lay


    # ========================================================
    # PREDICTION
    # ========================================================

    F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


    B = np.array(
        [
            [0.5 * dt * dt, 0],
            [0, 0.5 * dt * dt],
            [dt, 0],
            [0, dt],
        ],
        dtype=float,
    )


    acceleration = np.array(
        [
            accel_n,
            accel_e,
        ]
    )


    x = (
        F @ x
        +
        B @ acceleration
    )


    # ========================================================
    # PROCESS NOISE
    # ========================================================

    q = (
        ACCELERATION_NOISE
        ** 2
    )


    Q = q * np.array(
        [
            [
                dt ** 4 / 4,
                0,
                dt ** 3 / 2,
                0,
            ],
            [
                0,
                dt ** 4 / 4,
                0,
                dt ** 3 / 2,
            ],
            [
                dt ** 3 / 2,
                0,
                dt ** 2,
                0,
            ],
            [
                0,
                dt ** 3 / 2,
                0,
                dt ** 2,
            ],
        ]
    )


    P = (
        F
        @ P
        @ F.T
        +
        Q
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

        z = np.array(
            [
                float(row["north_m"]),
                float(row["east_m"]),
            ]
        )


        H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )


        R = np.diag(
            [
                GNSS_STD ** 2,
                GNSS_STD ** 2,
            ]
        )


        innovation = (
            z
            -
            H @ x
        )


        S = (
            H
            @ P
            @ H.T
            +
            R
        )


        K = (
            P
            @ H.T
            @ np.linalg.inv(S)
        )


        x = (
            x
            +
            K
            @ innovation
        )


        I = np.eye(4)


        P = (
            I
            -
            K @ H
        ) @ P


    # ========================================================
    # AI VELOCITY UPDATE
    # ========================================================

    ai_vn = float(
        row["vn_ai_mps"]
    )

    ai_ve = float(
        row["ve_ai_mps"]
    )


    std_n = max(
        float(
            row["vn_std_mps"]
        ),
        AI_STD_N_FLOOR,
    )


    std_e = max(
        float(
            row["ve_std_mps"]
        ),
        AI_STD_E_FLOOR,
    )


    # AI velocity measurement.

    z_ai = np.array(
        [
            ai_vn,
            ai_ve,
        ]
    )


    H_ai = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


    R_ai = np.diag(
        [
            std_n ** 2,
            std_e ** 2,
        ]
    )


    innovation_ai = (
        z_ai
        -
        H_ai @ x
    )


    S_ai = (
        H_ai
        @ P
        @ H_ai.T
        +
        R_ai
    )


    K_ai = (
        P
        @ H_ai.T
        @ np.linalg.inv(S_ai)
    )


    x = (
        x
        +
        K_ai
        @ innovation_ai
    )


    I = np.eye(4)


    P = (
        I
        -
        K_ai @ H_ai
    ) @ P


    # Joseph-style numerical stabilization.

    P = (
        P
        +
        P.T
    ) / 2.0


    # ========================================================
    # ERROR
    # ========================================================

    error = np.sqrt(
        (
            x[0]
            -
            float(row["north_m"])
        ) ** 2
        +
        (
            x[1]
            -
            float(row["east_m"])
        ) ** 2
    )


    # ========================================================
    # SAVE
    # ========================================================

    results.append(
        {
            "timestamp":
                row["timestamp"],

            "time_s":
                current_time,

            "gnss_available":
                gnss_available,

            "ekf_north_m":
                x[0],

            "ekf_east_m":
                x[1],

            "ekf_vn_mps":
                x[2],

            "ekf_ve_mps":
                x[3],

            "ai_vn_mps":
                ai_vn,

            "ai_ve_mps":
                ai_ve,

            "ai_vn_std":
                std_n,

            "ai_ve_std":
                std_e,

            "reference_north_m":
                float(row["north_m"]),

            "reference_east_m":
                float(row["east_m"]),

            "position_error_m":
                error,

            "accel_n_mps2":
                accel_n,

            "accel_e_mps2":
                accel_e,
        }
    )


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# OUTAGE
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
        "No outage samples found."
    )


# ============================================================
# METRICS
# ============================================================

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


dn = np.diff(
    outage[
        "reference_north_m"
    ].to_numpy()
)

de = np.diff(
    outage[
        "reference_east_m"
    ].to_numpy()
)


reference_distance = float(
    np.sum(
        np.sqrt(
            dn ** 2
            +
            de ** 2
        )
    )
)


drift = (
    final_error
    /
    reference_distance
    *
    100.0
)


# ============================================================
# PRINT
# ============================================================

print("\n")
print("=" * 80)
print("V10.1 EKF RESULTS")
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
    f" {drift:.3f}%"
)


# ============================================================
# SAVE
# ============================================================

results_file = (
    OUTPUT_DIR
    /
    "prototype_v10_1_ekf_results.csv"
)

results_df.to_csv(
    results_file,
    index=False,
)


summary_file = (
    OUTPUT_DIR
    /
    "prototype_v10_1_ekf_summary.csv"
)

pd.DataFrame(
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
                drift,
        }
    ]
).to_csv(
    summary_file,
    index=False,
)


# ============================================================
# TRAJECTORY
# ============================================================

plt.figure(
    figsize=(12, 8)
)

plt.plot(
    outage[
        "reference_east_m"
    ],
    outage[
        "reference_north_m"
    ],
    label="GNSS reference",
)

plt.plot(
    outage[
        "ekf_east_m"
    ],
    outage[
        "ekf_north_m"
    ],
    label="V10.1 EKF",
)

plt.xlabel(
    "East (m)"
)

plt.ylabel(
    "North (m)"
)

plt.title(
    "V10.1 EKF During GNSS Outage"
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
    "prototype_v10_1_ekf_trajectory.png"
)

plt.savefig(
    trajectory_file,
    dpi=200,
)

plt.close()


# ============================================================
# ERROR
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
    "V10.1 EKF Position Error"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

error_file = (
    OUTPUTS
    /
    "prototype_v10_1_ekf_error.png"
)

plt.savefig(
    error_file,
    dpi=200,
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("V10.1 EKF COMPLETE")
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