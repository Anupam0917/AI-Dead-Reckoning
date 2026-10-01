import os
import numpy as np
import pandas as pd


# ============================================================
# V10.15
# AI VELOCITY + FROZEN BIAS ERROR-STATE FILTER
#
# No raw accelerometer integration.
#
# AI velocity is treated as a pseudo-measurement.
#
# AI bias is estimated BEFORE the GNSS outage and then frozen.
#
# State:
#   [North, East, Vnorth, Veast]
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
    "prototype_v10_15_frozen_bias_filter.csv"
)


# ============================================================
# SETTINGS
# ============================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

MAX_DT = 0.5


# ============================================================
# AI BIAS CALIBRATION
# ============================================================

BIAS_CAL_START = 7000.0
BIAS_CAL_END = 8400.0


# ============================================================
# FILTER PARAMETERS
# ============================================================

POSITION_STD = 2.0
VELOCITY_STD = 1.0

POSITION_PROCESS_NOISE = 0.02
VELOCITY_PROCESS_NOISE = 0.20

GNSS_STD = 3.0


# AI velocity measurement uncertainty.
AI_NORTH_STD = 0.75
AI_EAST_STD = 0.45


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
print("V10.15 FROZEN-BIAS AI VELOCITY FILTER")
print("=" * 80)

print("\nNo raw accelerometer integration.")
print("AI velocity = pseudo-measurement.")
print("AI bias estimated before outage.")
print("AI bias frozen during outage.")


# ============================================================
# 1. LOAD AI VELOCITY
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
# 2. LOAD TARGET VELOCITY
# ============================================================

print("\n[2] Loading target velocity...")

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
# 3. CHECK REQUIRED AI COLUMNS
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

print("\n[3] Synchronizing AI and target...")

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
# 5. CREATE TARGET-DERIVED REFERENCE TRAJECTORY
# ============================================================

print(
    "\n[4] Creating reference trajectory..."
)


time_s = data[
    "time_s"
].to_numpy(
    dtype=float
)


vn_true = data[
    "vn_mps"
].to_numpy(
    dtype=float
)


ve_true = data[
    "ve_mps"
].to_numpy(
    dtype=float
)


reference_north = np.zeros(
    len(data),
    dtype=float
)

reference_east = np.zeros(
    len(data),
    dtype=float
)


for i in range(
    1,
    len(data)
):

    dt = np.clip(

        time_s[i]
        -
        time_s[i - 1],

        0.001,

        MAX_DT

    )


    reference_north[i] = (

        reference_north[i - 1]

        +

        vn_true[i - 1]
        *
        dt

    )


    reference_east[i] = (

        reference_east[i - 1]

        +

        ve_true[i - 1]
        *
        dt

    )


data["reference_north"] = (
    reference_north
)

data["reference_east"] = (
    reference_east
)


# ============================================================
# 6. ESTIMATE PRE-OUTAGE AI BIAS
# ============================================================

print(
    "\n[5] Estimating pre-outage AI bias..."
)


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


if len(calibration) == 0:

    raise RuntimeError(
        "No calibration samples found."
    )


north_bias = (

    calibration[
        "vn_ai_mps"
    ]
    -
    calibration[
        "vn_mps"
    ]

).median()


east_bias = (

    calibration[
        "ve_ai_mps"
    ]
    -
    calibration[
        "ve_mps"
    ]

).median()


print(
    "Frozen North bias:",
    round(
        north_bias,
        6
    ),
    "m/s"
)

print(
    "Frozen East bias:",
    round(
        east_bias,
        6
    ),
    "m/s"
)


# ============================================================
# 7. INITIAL STATE
#
# x =
#
# [ North
#   East
#   Vnorth
#   Veast ]
# ============================================================

x = np.zeros(
    4,
    dtype=float
)


x[0] = reference_north[0]

x[1] = reference_east[0]


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


# ============================================================
# 8. INITIAL COVARIANCE
# ============================================================

P = np.diag([

    POSITION_STD ** 2,

    POSITION_STD ** 2,

    VELOCITY_STD ** 2,

    VELOCITY_STD ** 2

])


# ============================================================
# 9. STORAGE
# ============================================================

results = []


# ============================================================
# 10. FILTER LOOP
# ============================================================

print(
    "\n[6] Running filter..."
)


previous_time = None


for _, row in data.iterrows():

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
    # STATE PREDICTION
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


    F = np.eye(
        4,
        dtype=float
    )

    F[0, 2] = dt
    F[1, 3] = dt


    Q = np.diag([

        POSITION_PROCESS_NOISE * dt,

        POSITION_PROCESS_NOISE * dt,

        VELOCITY_PROCESS_NOISE * dt,

        VELOCITY_PROCESS_NOISE * dt

    ])


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
    # AI VELOCITY UPDATE
    # ========================================================

    ai_vn = float(
        row["vn_ai_mps"]
    )

    ai_ve = float(
        row["ve_ai_mps"]
    )


    # Apply ONLY the pre-outage frozen bias.

    corrected_ai_vn = (
        ai_vn
        -
        north_bias
    )

    corrected_ai_ve = (
        ai_ve
        -
        east_bias
    )


    z = np.array([

        corrected_ai_vn,

        corrected_ai_ve

    ])


    H = np.array([

        [0, 0, 1, 0],

        [0, 0, 0, 1]

    ], dtype=float)


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

        AI_NORTH_STD ** 2,

        AI_EAST_STD ** 2

    ])


    S = (

        H
        @
        P
        @
        H.T

        +

        R_ai

    )


    K = (

        P
        @
        H.T
        @
        safe_inverse(S)

    )


    x = (
        x
        +
        K
        @
        innovation
    )


    P = (

        np.eye(4)
        -
        K @ H

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


        H_gps = np.array([

            [1, 0, 0, 0],

            [0, 1, 0, 0]

        ], dtype=float)


        predicted_gps = np.array([

            x[0],

            x[1]

        ])


        innovation_gps = (

            z_gnss
            -
            predicted_gps

        )


        R_gps = np.diag([

            GNSS_STD ** 2,

            GNSS_STD ** 2

        ])


        S_gps = (

            H_gps
            @
            P
            @
            H_gps.T

            +

            R_gps

        )


        K_gps = (

            P
            @
            H_gps.T
            @
            safe_inverse(S_gps)

        )


        x = (

            x
            +
            K_gps
            @
            innovation_gps

        )


        P = (

            np.eye(4)
            -
            K_gps @ H_gps

        ) @ P


    # ========================================================
    # POSITION ERROR
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
    # SAVE RESULT
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

        "ai_vn":
            ai_vn,

        "ai_ve":
            ai_ve,

        "corrected_ai_vn":
            corrected_ai_vn,

        "corrected_ai_ve":
            corrected_ai_ve,

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
].copy().reset_index(
    drop=True
)


if len(outage) == 0:

    raise RuntimeError(
        "No outage rows found."
    )


print(
    "\nOutage samples:",
    len(outage)
)


# ============================================================
# 13. COPY ARRAYS
#
# IMPORTANT:
# .copy() prevents the NumPy read-only array error.
# ============================================================

ref_n = outage[
    "reference_north"
].to_numpy(
    dtype=float,
    copy=True
)


ref_e = outage[
    "reference_east"
].to_numpy(
    dtype=float,
    copy=True
)


est_n = outage[
    "estimated_north"
].to_numpy(
    dtype=float,
    copy=True
)


est_e = outage[
    "estimated_east"
].to_numpy(
    dtype=float,
    copy=True
)


# ============================================================
# 14. RESET BOTH TRAJECTORIES
# ============================================================

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
# 15. POSITION ERROR
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
# 16. DISTANCE
# ============================================================

reference_distance = np.sum(

    np.sqrt(

        np.diff(ref_n) ** 2

        +

        np.diff(ref_e) ** 2

    )

)


filter_distance = np.sum(

    np.sqrt(

        np.diff(est_n) ** 2

        +

        np.diff(est_e) ** 2

    )

)


if reference_distance > 0:

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

else:

    drift_percent = np.nan


# ============================================================
# 17. VELOCITY METRICS
# ============================================================

outage_data = data[
    (
        data["time_s"]
        >= OUTAGE_START
    )
    &
    (
        data["time_s"]
        <= OUTAGE_END
    )
].copy().reset_index(
    drop=True
)


# Match lengths safely.

n = min(
    len(outage),
    len(outage_data)
)


true_vn = outage_data[
    "vn_mps"
].to_numpy(
    dtype=float,
    copy=True
)[:n]


true_ve = outage_data[
    "ve_mps"
].to_numpy(
    dtype=float,
    copy=True
)[:n]


filter_vn = outage[
    "estimated_vn"
].to_numpy(
    dtype=float,
    copy=True
)[:n]


filter_ve = outage[
    "estimated_ve"
].to_numpy(
    dtype=float,
    copy=True
)[:n]


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
# 18. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("V10.15 OUTAGE RESULTS")
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
# 19. SAVE RESULTS
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
print("V10.15 COMPLETE")
print("=" * 80)