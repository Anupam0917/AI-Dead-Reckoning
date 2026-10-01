import os
import numpy as np
import pandas as pd


# ============================================================
# V10.14
# AI VELOCITY + ERROR-STATE FILTER
#
# IMPORTANT:
# No raw accelerometer integration is used for position.
#
# AI velocity is treated as a pseudo-measurement.
# GNSS provides position correction when available.
#
# State:
#   [North, East, Vnorth, Veast, Vnorth_bias, Veast_bias]
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


AI_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_1_full_ai_velocity.csv"
)


TARGET_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v9_velocity_targets.csv"
)


OUTPUT_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_14_ai_velocity_filter.csv"
)


# ============================================================
# SETTINGS
# ============================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

MAX_DT = 0.5


# ------------------------------------------------------------
# Initial uncertainty
# ------------------------------------------------------------

POSITION_STD = 2.0
VELOCITY_STD = 1.0

VELOCITY_BIAS_STD = 0.5


# ------------------------------------------------------------
# Process noise
# ------------------------------------------------------------

POSITION_NOISE = 0.02
VELOCITY_NOISE = 0.30

BIAS_RANDOM_WALK = 0.002


# ------------------------------------------------------------
# AI velocity measurement noise
# ------------------------------------------------------------

AI_STD = 0.8


# ------------------------------------------------------------
# GNSS position noise
# ------------------------------------------------------------

GNSS_STD = 3.0


# ------------------------------------------------------------
# Bias learning before outage
# ------------------------------------------------------------

BIAS_CAL_START = 7000.0
BIAS_CAL_END = 8400.0


# ============================================================
# HELPER
# ============================================================

def safe_inverse(matrix):

    try:

        return np.linalg.inv(matrix)

    except np.linalg.LinAlgError:

        return np.linalg.pinv(matrix)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("V10.14 AI VELOCITY ERROR-STATE FILTER")
print("=" * 80)

print(
    "\nNo raw accelerometer integration."
)

print(
    "AI velocity = pseudo-measurement."
)

print(
    "GNSS = position correction when available."
)


# ============================================================
# 1. LOAD AI
# ============================================================

print("\n[1] Loading AI velocity...")

ai = pd.read_csv(
    AI_FILE
)

ai["timestamp"] = pd.to_datetime(
    ai["timestamp"]
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


print(
    "Target rows:",
    len(target)
)


# ============================================================
# 3. CHECK AI COLUMNS
# ============================================================

required_ai = [

    "vn_ai_mps",

    "ve_ai_mps"

]


for column in required_ai:

    if column not in ai.columns:

        raise ValueError(
            f"Missing AI column: {column}"
        )


# ============================================================
# 4. MERGE AI + TARGET
# ============================================================

print("\n[3] Synchronizing AI and reference...")


data = pd.merge_asof(

    target[
        [
            "timestamp",
            "time_s",
            "vn_mps",
            "ve_mps"
        ]
    ].sort_values("timestamp"),

    ai[
        [
            "timestamp",
            "vn_ai_mps",
            "ve_ai_mps"
        ]
    ].sort_values("timestamp"),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        "50ms"
    )
)


data = data.dropna(
    subset=[
        "time_s",
        "vn_mps",
        "ve_mps",
        "vn_ai_mps",
        "ve_ai_mps"
    ]
).reset_index(
    drop=True
)


print(
    "Usable rows:",
    len(data)
)


# ============================================================
# 5. CREATE REFERENCE POSITION
#
# Target-derived reference is intentionally used here
# so that V10.14 can be compared consistently with our
# V10.10/V10.13 common benchmark.
# ============================================================

print("\n[4] Creating target-derived position reference...")


time_values = data[
    "time_s"
].values


vn_reference = data[
    "vn_mps"
].values


ve_reference = data[
    "ve_mps"
].values


reference_north = np.zeros(
    len(data)
)

reference_east = np.zeros(
    len(data)
)


for i in range(
    1,
    len(data)
):

    dt = np.clip(

        time_values[i]
        -
        time_values[i - 1],

        0.001,

        MAX_DT

    )


    reference_north[i] = (

        reference_north[i - 1]

        +

        vn_reference[i - 1]
        *
        dt

    )


    reference_east[i] = (

        reference_east[i - 1]

        +

        ve_reference[i - 1]
        *
        dt

    )


data[
    "reference_north"
] = reference_north


data[
    "reference_east"
] = reference_east


# ============================================================
# 6. ESTIMATE AI VELOCITY BIAS
#
# IMPORTANT:
# Only PRE-OUTAGE data is used.
#
# This prevents future outage information from leaking
# into the filter.
# ============================================================

print("\n[5] Estimating pre-outage AI velocity bias...")


calibration = data[
    (
        data["time_s"]
        >= BIAS_CAL_START
    )
    &
    (
        data["time_s"]
        <= BIAS_CAL_END
    )
].copy()


north_bias = (

    calibration["vn_ai_mps"]
    -
    calibration["vn_mps"]

).median()


east_bias = (

    calibration["ve_ai_mps"]
    -
    calibration["ve_mps"]

).median()


print(
    "Estimated North AI bias:",
    round(
        north_bias,
        6
    ),
    "m/s"
)


print(
    "Estimated East AI bias:",
    round(
        east_bias,
        6
    ),
    "m/s"
)


# ============================================================
# 7. STATE
#
# x =
#
# [ North
#   East
#   Vnorth
#   Veast
#   Vnorth_bias
#   Veast_bias ]
# ============================================================

x = np.zeros(
    6,
    dtype=float
)


# Start position at beginning
# of the complete trajectory.

x[0] = reference_north[0]

x[1] = reference_east[0]


# Initial velocity from AI

x[2] = (
    data[
        "vn_ai_mps"
    ].iloc[0]
    -
    north_bias
)


x[3] = (
    data[
        "ve_ai_mps"
    ].iloc[0]
    -
    east_bias
)


# Start with calibrated bias

x[4] = north_bias

x[5] = east_bias


# ============================================================
# 8. INITIAL COVARIANCE
# ============================================================

P = np.diag([

    POSITION_STD ** 2,

    POSITION_STD ** 2,

    VELOCITY_STD ** 2,

    VELOCITY_STD ** 2,

    VELOCITY_BIAS_STD ** 2,

    VELOCITY_BIAS_STD ** 2

])


# ============================================================
# 9. STORAGE
# ============================================================

results = []


# ============================================================
# 10. FILTER LOOP
# ============================================================

print("\n[6] Running AI velocity filter...")


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
    # GNSS AVAILABILITY
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


    # ========================================================
    # PREDICTION
    #
    # Position comes ONLY from the filtered velocity.
    # No accelerometer is integrated.
    # ========================================================

    x[0] = (

        x[0]
        +
        x[2]
        *
        dt

    )


    x[1] = (

        x[1]
        +
        x[3]
        *
        dt

    )


    # Bias is modeled as slowly varying.

    # Velocity itself remains the previous
    # filtered velocity until AI measurement
    # update below.


    # --------------------------------------------------------
    # STATE TRANSITION MATRIX
    # --------------------------------------------------------

    F = np.eye(6)


    F[0, 2] = dt

    F[1, 3] = dt


    # --------------------------------------------------------
    # PROCESS NOISE
    # --------------------------------------------------------

    Q = np.diag([

        POSITION_NOISE
        *
        dt,

        POSITION_NOISE
        *
        dt,

        VELOCITY_NOISE
        *
        dt,

        VELOCITY_NOISE
        *
        dt,

        BIAS_RANDOM_WALK
        *
        dt,

        BIAS_RANDOM_WALK
        *
        dt

    ])


    P = (

        F
        @ P
        @ F.T
        +
        Q

    )


    # ========================================================
    # AI VELOCITY UPDATE
    # ========================================================

    ai_vn = float(
        row[
            "vn_ai_mps"
        ]
    )


    ai_ve = float(
        row[
            "ve_ai_mps"
        ]
    )


    # Measurement:
    #
    # z = velocity + bias
    #
    # We estimate the bias state explicitly.

    z_ai = np.array([

        ai_vn,

        ai_ve

    ])


    H_ai = np.array([

        [0, 0, 1, 0, 1, 0],

        [0, 0, 0, 1, 0, 1]

    ], dtype=float)


    predicted_ai = np.array([

        x[2] + x[4],

        x[3] + x[5]

    ])


    innovation = (

        z_ai
        -
        predicted_ai

    )


    # --------------------------------------------------------
    # AI covariance
    # --------------------------------------------------------

    R_ai = np.diag([

        AI_STD ** 2,

        AI_STD ** 2

    ])


    S = (

        H_ai
        @ P
        @ H_ai.T
        +
        R_ai

    )


    K = (

        P
        @ H_ai.T
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
        K @ H_ai

    ) @ P


    # ========================================================
    # GNSS POSITION UPDATE
    # ========================================================

    if gnss_available:

        z_gnss = np.array([

            row[
                "reference_north"
            ],

            row[
                "reference_east"
            ]

        ])


        H_gnss = np.array([

            [1, 0, 0, 0, 0, 0],

            [0, 1, 0, 0, 0, 0]

        ], dtype=float)


        predicted_gnss = np.array([

            x[0],

            x[1]

        ])


        innovation_gnss = (

            z_gnss
            -
            predicted_gnss

        )


        R_gnss = np.diag([

            GNSS_STD ** 2,

            GNSS_STD ** 2

        ])


        S_gnss = (

            H_gnss
            @ P
            @ H_gnss.T
            +
            R_gnss

        )


        K_gnss = (

            P
            @ H_gnss.T
            @ safe_inverse(S_gnss)

        )


        x = (

            x
            +
            K_gnss
            @ innovation_gnss

        )


        P = (

            np.eye(6)
            -
            K_gnss @ H_gnss

        ) @ P


    # ========================================================
    # ERROR
    # ========================================================

    position_error = np.sqrt(

        (
            x[0]
            -
            row[
                "reference_north"
            ]
        ) ** 2

        +

        (
            x[1]
            -
            row[
                "reference_east"
            ]
        ) ** 2

    )


    # ========================================================
    # SAVE
    # ========================================================

    results.append({

        "timestamp":
            row["timestamp"],

        "time_s":
            current_time,

        "gnss_available":
            gnss_available,

        "reference_north":
            row[
                "reference_north"
            ],

        "reference_east":
            row[
                "reference_east"
            ],

        "estimated_north":
            x[0],

        "estimated_east":
            x[1],

        "estimated_vn":
            x[2],

        "estimated_ve":
            x[3],

        "estimated_vn_bias":
            x[4],

        "estimated_ve_bias":
            x[5],

        "ai_vn":
            ai_vn,

        "ai_ve":
            ai_ve,

        "position_error_m":
            position_error

    })


# ============================================================
# 11. RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    results
)


# ============================================================
# 12. EXTRACT OUTAGE
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
        "No outage rows found."
    )


# ============================================================
# 13. RESET OUTAGE TRAJECTORY
# ============================================================

ref_n = outage[
    "reference_north"
].values


ref_e = outage[
    "reference_east"
].values


est_n = outage[
    "estimated_north"
].values


est_e = outage[
    "estimated_east"
].values


# Remove position offset at outage start.

ref_n = (
    ref_n
    -
    ref_n[0]
)

ref_e = (
    ref_e
    -
    ref_e[0]
)


est_n = (
    est_n
    -
    est_n[0]
)

est_e = (
    est_e
    -
    est_e[0]
)


# ============================================================
# 14. POSITION ERROR
# ============================================================

position_error = np.sqrt(

    (
        est_n
        -
        ref_n
    ) ** 2

    +

    (
        est_e
        -
        ref_e
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
# 15. REFERENCE DISTANCE
# ============================================================

reference_distance = np.sum(

    np.sqrt(

        np.diff(ref_n) ** 2
        +
        np.diff(ref_e) ** 2

    )

)


estimated_distance = np.sum(

    np.sqrt(

        np.diff(est_n) ** 2
        +
        np.diff(est_e) ** 2

    )

)


drift_percent = (

    abs(

        estimated_distance
        -
        reference_distance

    )

    /

    reference_distance

    *

    100

)


# ============================================================
# 16. VELOCITY METRICS
# ============================================================

true_vn = data[
    (
        data["time_s"]
        >= OUTAGE_START
    )
    &
    (
        data["time_s"]
        <= OUTAGE_END
    )
]["vn_mps"].values


true_ve = data[
    (
        data["time_s"]
        >= OUTAGE_START
    )
    &
    (
        data["time_s"]
        <= OUTAGE_END
    )
]["ve_mps"].values


filter_vn = outage[
    "estimated_vn"
].values


filter_ve = outage[
    "estimated_ve"
].values


vn_mae = np.mean(

    np.abs(
        filter_vn
        -
        true_vn
    )

)


ve_mae = np.mean(

    np.abs(
        filter_ve
        -
        true_ve
    )

)


# ============================================================
# 17. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("V10.14 OUTAGE RESULTS")
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
        estimated_distance,
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


print(
    "\nFinal estimated Vnorth bias:",
    round(
        x[4],
        6
    ),
    "m/s"
)


print(
    "Final estimated Veast bias:",
    round(
        x[5],
        6
    ),
    "m/s"
)


# ============================================================
# 18. SAVE
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
print("V10.14 COMPLETE")
print("=" * 80)