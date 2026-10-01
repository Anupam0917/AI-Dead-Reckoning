import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# V10.16
# MULTI-OUTAGE BENCHMARK
#
# Existing best AI model:
#     GYRO + MAG
#
# No EKF.
# No new tuning.
# No new model architecture.
#
# Purpose:
#     Test whether the current RF velocity model works
#     across multiple GNSS-denied windows.
# ============================================================


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


OUTPUT_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_16_multi_outage.csv"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOWS = [

    # start, duration
    (7500.0, 30.0),
    (7500.0, 60.0),
    (7500.0, 120.0),

    (8500.0, 30.0),
    (8500.0, 60.0),
    (8500.0, 120.0),

    (9500.0, 30.0),
    (9500.0, 60.0),
    (9500.0, 120.0),

    (10000.0, 30.0),
    (10000.0, 60.0),
    (10000.0, 120.0)

]


TRAIN_RATIO = 0.70

N_ESTIMATORS = 150

MAX_DEPTH = 20

MIN_SAMPLES_LEAF = 3

RANDOM_STATE = 42

MAX_DT = 0.5


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("V10.16 MULTI-OUTAGE RF GYRO+MAG BENCHMARK")
print("=" * 80)

print(
    "\nExisting best AI model:"
)

print(
    "Random Forest using GYRO + MAG"
)

print(
    "\nNo EKF."
)

print(
    "No new architecture."

)

print(
    "No outage-specific tuning."
)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("\n[1] Loading raw dataset...")


raw = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)


raw.columns = (
    raw.columns
    .str.strip()
)


print(
    "Rows:",
    len(raw)
)


# ============================================================
# 2. PARSE TIMESTAMP
# ============================================================

raw["timestamp"] = pd.to_datetime(

    raw[
        "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
    ],

    format="%Y-%m-%d %H:%M:%S:%f",

    errors="coerce"

)


# ============================================================
# 3. FIND MAGNETOMETER
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

    raise RuntimeError(
        "Magnetometer columns could not be detected."
    )


print(
    "\nMagnetometer:"
)

print(
    "X:",
    mag_x
)

print(
    "Y:",
    mag_y
)

print(
    "Z:",
    mag_z
)


# ============================================================
# 4. NUMERIC HELPER
# ============================================================

def numeric(column):

    return pd.to_numeric(
        raw[column],
        errors="coerce"
    )


# ============================================================
# 5. CREATE SENSOR DATA
# ============================================================

print(
    "\n[2] Preparing GYRO + MAG features..."
)


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

    "mag_x":
        numeric(mag_x),

    "mag_y":
        numeric(mag_y),

    "mag_z":
        numeric(mag_z)

})


# ============================================================
# 6. LOAD TARGET
# ============================================================

print(
    "\n[3] Loading velocity targets..."
)


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
# 7. MERGE
# ============================================================

print(
    "\n[4] Synchronizing sensors and targets..."
)


data = pd.merge_asof(

    target[
        [
            "timestamp",
            "time_s",
            "vn_mps",
            "ve_mps"
        ]
    ].sort_values(
        "timestamp"
    ),

    sensor.sort_values(
        "timestamp"
    ),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        "50ms"
    )

)


# ============================================================
# 8. FEATURE ENGINEERING
#
# Keep this intentionally aligned with the
# GYRO + MAG family rather than inventing
# a new representation.
# ============================================================

print(
    "\n[5] Creating temporal sensor features..."
)


base_sensor_columns = [

    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",

    "mag_x",
    "mag_y",
    "mag_z"

]


feature_data = data[
    base_sensor_columns
].copy()


# Rolling windows used by the existing
# RF feature family.

ROLLING_WINDOWS = [
    5,
    10,
    20
]


features = pd.DataFrame(
    index=data.index
)


for column in base_sensor_columns:

    # Raw value

    features[
        column
    ] = feature_data[
        column
    ]


    for window in ROLLING_WINDOWS:

        features[
            f"{column}_mean_{window}"
        ] = (
            feature_data[
                column
            ]
            .rolling(
                window,
                min_periods=1
            )
            .mean()
        )


        features[
            f"{column}_std_{window}"
        ] = (
            feature_data[
                column
            ]
            .rolling(
                window,
                min_periods=1
            )
            .std()
            .fillna(0)
        )


# Magnetometer magnitude

features[
    "mag_magnitude"
] = np.sqrt(

    feature_data["mag_x"] ** 2
    +
    feature_data["mag_y"] ** 2
    +
    feature_data["mag_z"] ** 2

)


# Gyroscope magnitude

features[
    "gyro_magnitude"
] = np.sqrt(

    feature_data["gyro_yaw"] ** 2
    +
    feature_data["gyro_pitch"] ** 2
    +
    feature_data["gyro_roll"] ** 2

)


# ============================================================
# 9. CLEAN
# ============================================================

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)


features = features.fillna(
    0
)


data = data.loc[
    features.index
].copy()


data = data.reset_index(
    drop=True
)

features = features.reset_index(
    drop=True
)


# ============================================================
# 10. CHRONOLOGICAL SPLIT
# ============================================================

n = len(data)


train_end = int(
    n * TRAIN_RATIO
)


X = features.to_numpy(
    dtype=np.float32
)


y_north = data[
    "vn_mps"
].to_numpy(
    dtype=np.float32
)


y_east = data[
    "ve_mps"
].to_numpy(
    dtype=np.float32
)


X_train = X[
    :train_end
]


X_test = X[
    train_end:
]


y_north_train = y_north[
    :train_end
]


y_north_test = y_north[
    train_end:
]


y_east_train = y_east[
    :train_end
]


y_east_test = y_east[
    train_end:
]


print(
    "\nTraining rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# ============================================================
# 11. TRAIN RANDOM FORESTS
# ============================================================

print(
    "\n[6] Training North velocity RF..."
)


rf_north = RandomForestRegressor(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    random_state=RANDOM_STATE,

    n_jobs=-1

)


rf_north.fit(
    X_train,
    y_north_train
)


print(
    "North model complete."
)


print(
    "\n[7] Training East velocity RF..."
)


rf_east = RandomForestRegressor(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    random_state=RANDOM_STATE,

    n_jobs=-1

)


rf_east.fit(
    X_train,
    y_east_train
)


print(
    "East model complete."
)


# ============================================================
# 12. PREDICT COMPLETE DATASET
# ============================================================

print(
    "\n[8] Generating AI velocity..."
)


vn_ai = rf_north.predict(
    X
)


ve_ai = rf_east.predict(
    X
)


data[
    "vn_ai"
] = vn_ai


data[
    "ve_ai"
] = ve_ai


# ============================================================
# 13. BENCHMARK FUNCTION
# ============================================================

def evaluate_window(
    start,
    duration
):

    end = (
        start
        +
        duration
    )


    window = data[
        (
            data["time_s"]
            >= start
        )
        &
        (
            data["time_s"]
            <= end
        )
    ].copy()


    if len(window) < 10:

        return None


    time = window[
        "time_s"
    ].to_numpy(
        dtype=float
    )


    true_vn = window[
        "vn_mps"
    ].to_numpy(
        dtype=float
    )


    true_ve = window[
        "ve_mps"
    ].to_numpy(
        dtype=float
    )


    pred_vn = window[
        "vn_ai"
    ].to_numpy(
        dtype=float
    )


    pred_ve = window[
        "ve_ai"
    ].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # Velocity errors
    # --------------------------------------------------------

    north_mae = np.mean(
        np.abs(
            pred_vn
            -
            true_vn
        )
    )


    east_mae = np.mean(
        np.abs(
            pred_ve
            -
            true_ve
        )
    )


    # --------------------------------------------------------
    # Integrate true velocity
    # --------------------------------------------------------

    true_north = np.zeros(
        len(window)
    )

    true_east = np.zeros(
        len(window)
    )


    # Integrate AI velocity

    ai_north = np.zeros(
        len(window)
    )

    ai_east = np.zeros(
        len(window)
    )


    for i in range(
        1,
        len(window)
    ):

        dt = np.clip(

            time[i]
            -
            time[i - 1],

            0.001,

            MAX_DT

        )


        true_north[i] = (

            true_north[i - 1]

            +

            true_vn[i - 1]
            *
            dt

        )


        true_east[i] = (

            true_east[i - 1]

            +

            true_ve[i - 1]
            *
            dt

        )


        ai_north[i] = (

            ai_north[i - 1]

            +

            pred_vn[i - 1]
            *
            dt

        )


        ai_east[i] = (

            ai_east[i - 1]

            +

            pred_ve[i - 1]
            *
            dt

        )


    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    position_error = np.sqrt(

        (
            ai_north
            -
            true_north
        ) ** 2

        +

        (
            ai_east
            -
            true_east
        ) ** 2

    )


    mean_error = np.mean(
        position_error
    )


    final_error = (
        position_error[-1]
    )


    max_error = np.max(
        position_error
    )


    # --------------------------------------------------------
    # Distances
    # --------------------------------------------------------

    true_distance = np.sum(

        np.sqrt(

            np.diff(true_north) ** 2

            +

            np.diff(true_east) ** 2

        )

    )


    ai_distance = np.sum(

        np.sqrt(

            np.diff(ai_north) ** 2

            +

            np.diff(ai_east) ** 2

        )

    )


    if true_distance > 0:

        drift = (

            abs(
                ai_distance
                -
                true_distance
            )

            /

            true_distance

            *

            100

        )

    else:

        drift = np.nan


    return {

        "start_s":
            start,

        "duration_s":
            duration,

        "samples":
            len(window),

        "north_velocity_mae_mps":
            north_mae,

        "east_velocity_mae_mps":
            east_mae,

        "mean_position_error_m":
            mean_error,

        "final_position_error_m":
            final_error,

        "max_position_error_m":
            max_error,

        "reference_distance_m":
            true_distance,

        "ai_distance_m":
            ai_distance,

        "drift_percent":
            drift

    }


# ============================================================
# 14. RUN ALL OUTAGES
# ============================================================

print(
    "\n[9] Running multi-outage benchmark..."
)


benchmark_results = []


for start, duration in WINDOWS:

    print(
        f"\nTesting {start:.0f}s "
        f"for {duration:.0f}s..."
    )


    result = evaluate_window(
        start,
        duration
    )


    if result is None:

        print(
            "Skipped: insufficient samples."
        )

    else:

        benchmark_results.append(
            result
        )

        print(
            "Final error:",
            round(
                result[
                    "final_position_error_m"
                ],
                3
            ),
            "m"
        )

        print(
            "Drift:",
            round(
                result[
                    "drift_percent"
                ],
                3
            ),
            "%"
        )


# ============================================================
# 15. RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    benchmark_results
)


# ============================================================
# 16. SUMMARY
# ============================================================

if len(results) == 0:

    raise RuntimeError(
        "No valid outage windows."
    )


mean_final = results[
    "final_position_error_m"
].mean()


median_final = results[
    "final_position_error_m"
].median()


max_final = results[
    "final_position_error_m"
].max()


mean_position = results[
    "mean_position_error_m"
].mean()


mean_drift = results[
    "drift_percent"
].mean()


median_drift = results[
    "drift_percent"
].median()


mean_north_mae = results[
    "north_velocity_mae_mps"
].mean()


mean_east_mae = results[
    "east_velocity_mae_mps"
].mean()


# ============================================================
# 17. PRINT TABLE
# ============================================================

print("\n")
print("=" * 110)
print("V10.16 MULTI-OUTAGE RESULTS")
print("=" * 110)


print(
    results[
        [
            "start_s",
            "duration_s",
            "samples",
            "north_velocity_mae_mps",
            "east_velocity_mae_mps",
            "mean_position_error_m",
            "final_position_error_m",
            "max_position_error_m",
            "reference_distance_m",
            "ai_distance_m",
            "drift_percent"
        ]
    ].round(3).to_string(
        index=False
    )
)


# ============================================================
# 18. SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("V10.16 SUMMARY")
print("=" * 80)


print(
    "\nValid outage windows:",
    len(results)
)


print(
    "\nMean North velocity MAE:",
    round(
        mean_north_mae,
        4
    ),
    "m/s"
)


print(
    "Mean East velocity MAE:",
    round(
        mean_east_mae,
        4
    ),
    "m/s"
)


print(
    "\nMean position error:",
    round(
        mean_position,
        3
    ),
    "m"
)


print(
    "Mean final position error:",
    round(
        mean_final,
        3
    ),
    "m"
)


print(
    "Median final position error:",
    round(
        median_final,
        3
    ),
    "m"
)


print(
    "Maximum final position error:",
    round(
        max_final,
        3
    ),
    "m"
)


print(
    "\nMean drift:",
    round(
        mean_drift,
        3
    ),
    "%"
)


print(
    "Median drift:",
    round(
        median_drift,
        3
    ),
    "%"
)


# ============================================================
# 19. SAVE
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
print("V10.16 COMPLETE")
print("=" * 80)