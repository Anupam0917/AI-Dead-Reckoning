import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# V10.17
# ERROR DIAGNOSIS
#
# Purpose:
#   Investigate why the V10.16 RF GYRO+MAG model degrades
#   during later GNSS-denied windows.
#
# Questions:
#   1. Does AI velocity error increase with time?
#   2. Is the problem mainly North or East velocity?
#   3. Does sensor distribution change?
#   4. Are magnetometer / gyro values different later?
#   5. Does model error correlate with sensor magnitude?
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

OUTPUT_CSV = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_17_error_diagnosis.csv"
)

OUTPUT_PNG = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_17_error_diagnosis.png"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70

N_ESTIMATORS = 150

MAX_DEPTH = 20

MIN_SAMPLES_LEAF = 3

RANDOM_STATE = 42

BIN_SIZE = 100.0


print("=" * 80)
print("V10.17 ERROR DIAGNOSIS")
print("=" * 80)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("\n[1] Loading raw dataset...")

raw = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

raw.columns = raw.columns.str.strip()

print("Rows:", len(raw))


# ============================================================
# 2. TIMESTAMP
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


if mag_x is None or mag_y is None or mag_z is None:

    raise RuntimeError(
        "Could not detect magnetometer columns."
    )


print("\nMagnetometer columns:")

print("X:", mag_x)
print("Y:", mag_y)
print("Z:", mag_z)


# ============================================================
# 4. NUMERIC SENSOR DATA
# ============================================================

def numeric(column):

    return pd.to_numeric(
        raw[column],
        errors="coerce"
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
# 5. LOAD TARGETS
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
# 6. SYNCHRONIZE
# ============================================================

print("\n[3] Synchronizing data...")

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
# 7. FEATURE ENGINEERING
# ============================================================

print("\n[4] Creating GYRO + MAG features...")

base_columns = [

    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",

    "mag_x",
    "mag_y",
    "mag_z"

]


features = pd.DataFrame(
    index=data.index
)


for column in base_columns:

    features[column] = data[column]

    for window in [5, 10, 20]:

        features[
            f"{column}_mean_{window}"
        ] = (
            data[column]
            .rolling(
                window,
                min_periods=1
            )
            .mean()
        )

        features[
            f"{column}_std_{window}"
        ] = (
            data[column]
            .rolling(
                window,
                min_periods=1
            )
            .std()
            .fillna(0)
        )


# Magnitude features

features["mag_magnitude"] = np.sqrt(

    data["mag_x"] ** 2
    +
    data["mag_y"] ** 2
    +
    data["mag_z"] ** 2

)


features["gyro_magnitude"] = np.sqrt(

    data["gyro_yaw"] ** 2
    +
    data["gyro_pitch"] ** 2
    +
    data["gyro_roll"] ** 2

)


# ============================================================
# 8. CLEAN
# ============================================================

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

features = features.fillna(0)

data = data.reset_index(
    drop=True
)

features = features.reset_index(
    drop=True
)


# ============================================================
# 9. TRAIN / TEST SPLIT
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


print("\nTraining rows:", train_end)

print(
    "Testing rows:",
    n - train_end
)


print(
    "Training time:",
    round(
        data["time_s"].iloc[0],
        2
    ),
    "to",
    round(
        data["time_s"].iloc[train_end - 1],
        2
    ),
    "seconds"
)

print(
    "Testing time:",
    round(
        data["time_s"].iloc[train_end],
        2
    ),
    "to",
    round(
        data["time_s"].iloc[-1],
        2
    ),
    "seconds"
)


# ============================================================
# 10. TRAIN RF MODELS
# ============================================================

print("\n[5] Training North RF...")

rf_north = RandomForestRegressor(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    random_state=RANDOM_STATE,

    n_jobs=-1

)

rf_north.fit(
    X[:train_end],
    y_north[:train_end]
)


print("North model complete.")


print("\n[6] Training East RF...")

rf_east = RandomForestRegressor(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    random_state=RANDOM_STATE,

    n_jobs=-1

)

rf_east.fit(
    X[:train_end],
    y_east[:train_end]
)


print("East model complete.")


# ============================================================
# 11. PREDICT
# ============================================================

print("\n[7] Predicting entire dataset...")

data["vn_ai"] = rf_north.predict(X)

data["ve_ai"] = rf_east.predict(X)


# ============================================================
# 12. VELOCITY ERRORS
# ============================================================

data["north_error"] = (
    data["vn_ai"]
    -
    data["vn_mps"]
)

data["east_error"] = (
    data["ve_ai"]
    -
    data["ve_mps"]
)


data["north_abs_error"] = np.abs(
    data["north_error"]
)

data["east_abs_error"] = np.abs(
    data["east_error"]
)


data["speed_true"] = np.sqrt(

    data["vn_mps"] ** 2
    +
    data["ve_mps"] ** 2

)


data["speed_ai"] = np.sqrt(

    data["vn_ai"] ** 2
    +
    data["ve_ai"] ** 2

)


data["speed_error"] = (
    data["speed_ai"]
    -
    data["speed_true"]
)


data["speed_abs_error"] = np.abs(
    data["speed_error"]
)


# ============================================================
# 13. SENSOR MAGNITUDES
# ============================================================

data["mag_magnitude"] = np.sqrt(

    data["mag_x"] ** 2
    +
    data["mag_y"] ** 2
    +
    data["mag_z"] ** 2

)


data["gyro_magnitude"] = np.sqrt(

    data["gyro_yaw"] ** 2
    +
    data["gyro_pitch"] ** 2
    +
    data["gyro_roll"] ** 2

)


# ============================================================
# 14. TRAIN / TEST FLAG
# ============================================================

data["dataset_split"] = np.where(

    np.arange(len(data))
    <
    train_end,

    "train",

    "test"

)


# ============================================================
# 15. TIME BINS
# ============================================================

data["time_bin_s"] = (

    np.floor(
        data["time_s"]
        /
        BIN_SIZE
    )
    *
    BIN_SIZE

)


# ============================================================
# 16. AGGREGATE TIME-BIN ERROR
# ============================================================

print("\n[8] Calculating time-dependent error...")

diagnosis = (

    data
    .groupby(
        [
            "dataset_split",
            "time_bin_s"
        ]
    )
    .agg(

        samples=(
            "time_s",
            "count"
        ),

        mean_north_abs_error=(
            "north_abs_error",
            "mean"
        ),

        mean_east_abs_error=(
            "east_abs_error",
            "mean"
        ),

        mean_speed_abs_error=(
            "speed_abs_error",
            "mean"
        ),

        mean_north_bias=(
            "north_error",
            "mean"
        ),

        mean_east_bias=(
            "east_error",
            "mean"
        ),

        mean_mag=(
            "mag_magnitude",
            "mean"
        ),

        std_mag=(
            "mag_magnitude",
            "std"
        ),

        mean_gyro=(
            "gyro_magnitude",
            "mean"
        ),

        std_gyro=(
            "gyro_magnitude",
            "std"
        ),

        mean_true_speed=(
            "speed_true",
            "mean"
        )

    )

    .reset_index()

)


# ============================================================
# 17. SPECIFIC OUTAGE WINDOWS
# ============================================================

OUTAGES = [

    (7500, 30),
    (7500, 60),
    (7500, 120),

    (8500, 30),
    (8500, 60),
    (8500, 120),

    (9500, 30),
    (9500, 60),
    (9500, 120),

    (10000, 30),
    (10000, 60),
    (10000, 120)

]


outage_rows = []


for start, duration in OUTAGES:

    end = start + duration

    subset = data[
        (
            data["time_s"]
            >= start
        )
        &
        (
            data["time_s"]
            <= end
        )
    ]


    if len(subset) == 0:
        continue


    outage_rows.append({

        "start_s":
            start,

        "duration_s":
            duration,

        "north_mae_mps":
            subset[
                "north_abs_error"
            ].mean(),

        "east_mae_mps":
            subset[
                "east_abs_error"
            ].mean(),

        "speed_mae_mps":
            subset[
                "speed_abs_error"
            ].mean(),

        "north_bias_mps":
            subset[
                "north_error"
            ].mean(),

        "east_bias_mps":
            subset[
                "east_error"
            ].mean(),

        "mean_mag":
            subset[
                "mag_magnitude"
            ].mean(),

        "mean_gyro":
            subset[
                "gyro_magnitude"
            ].mean(),

        "mean_speed":
            subset[
                "speed_true"
            ].mean()

    })


outage_diagnosis = pd.DataFrame(
    outage_rows
)


# ============================================================
# 18. PRINT TIME-BIN SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("TIME-DEPENDENT ERROR")
print("=" * 110)


test_diagnosis = diagnosis[
    diagnosis[
        "dataset_split"
    ]
    ==
    "test"
]


print(

    test_diagnosis[
        [
            "time_bin_s",
            "mean_north_abs_error",
            "mean_east_abs_error",
            "mean_speed_abs_error",
            "mean_north_bias",
            "mean_east_bias",
            "mean_mag",
            "mean_gyro",
            "mean_true_speed"
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )

)


# ============================================================
# 19. PRINT OUTAGE DIAGNOSIS
# ============================================================

print("\n")
print("=" * 110)
print("OUTAGE-SPECIFIC DIAGNOSIS")
print("=" * 110)


print(

    outage_diagnosis.round(
        4
    ).to_string(
        index=False
    )

)


# ============================================================
# 20. CORRELATION ANALYSIS
# ============================================================

print("\n")
print("=" * 80)
print("ERROR / SENSOR CORRELATION")
print("=" * 80)


test = data[
    data["dataset_split"]
    ==
    "test"
].copy()


correlation_columns = [

    "north_error",
    "east_error",
    "speed_error",

    "mag_magnitude",
    "gyro_magnitude",

    "speed_true"

]


correlations = test[
    correlation_columns
].corr()


print(
    correlations.round(
        4
    )
)


# ============================================================
# 21. TRAIN VS TEST SENSOR DISTRIBUTION
# ============================================================

train = data[
    data["dataset_split"]
    ==
    "train"
]

test = data[
    data["dataset_split"]
    ==
    "test"
]


print("\n")
print("=" * 80)
print("TRAIN VS TEST SENSOR DISTRIBUTION")
print("=" * 80)


distribution_rows = []


for column in [

    "mag_magnitude",
    "gyro_magnitude",
    "speed_true"

]:

    distribution_rows.append({

        "feature":
            column,

        "train_mean":
            train[column].mean(),

        "train_std":
            train[column].std(),

        "test_mean":
            test[column].mean(),

        "test_std":
            test[column].std(),

        "test_min":
            test[column].min(),

        "test_max":
            test[column].max()

    })


distribution = pd.DataFrame(
    distribution_rows
)


print(
    distribution.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# 22. SAVE CSV
# ============================================================

print("\n[9] Saving diagnosis CSV...")

diagnosis.to_csv(
    OUTPUT_CSV,
    index=False
)


print(
    "Saved:",
    OUTPUT_CSV
)


# ============================================================
# 23. PLOT
# ============================================================

print("\n[10] Creating diagnostic plot...")


fig, axes = plt.subplots(
    3,
    1,
    figsize=(13, 12),
    sharex=True
)


# ------------------------------------------------------------
# Plot 1: North/East error
# ------------------------------------------------------------

axes[0].plot(

    test["time_s"],

    test["north_abs_error"],

    linewidth=0.8,

    label="North absolute error"

)

axes[0].plot(

    test["time_s"],

    test["east_abs_error"],

    linewidth=0.8,

    label="East absolute error"

)


axes[0].axvspan(
    7500,
    7620,
    alpha=0.12
)

axes[0].axvspan(
    8500,
    8620,
    alpha=0.12
)

axes[0].axvspan(
    9500,
    9620,
    alpha=0.12
)

axes[0].axvspan(
    10000,
    10120,
    alpha=0.12
)


axes[0].set_ylabel(
    "Velocity error (m/s)"
)

axes[0].set_title(
    "AI Velocity Error Across Held-Out Test Region"
)

axes[0].legend()

axes[0].grid(
    True,
    alpha=0.3
)


# ------------------------------------------------------------
# Plot 2: Magnetometer
# ------------------------------------------------------------

axes[1].plot(

    test["time_s"],

    test["mag_magnitude"],

    linewidth=0.8,

    label="Magnetometer magnitude"

)


axes[1].set_ylabel(
    "Magnetic magnitude"
)

axes[1].set_title(
    "Magnetometer Signal"
)

axes[1].legend()

axes[1].grid(
    True,
    alpha=0.3
)


# ------------------------------------------------------------
# Plot 3: Gyroscope
# ------------------------------------------------------------

axes[2].plot(

    test["time_s"],

    test["gyro_magnitude"],

    linewidth=0.8,

    label="Gyroscope magnitude"

)


axes[2].set_xlabel(
    "Time (s)"
)

axes[2].set_ylabel(
    "Gyro magnitude"
)

axes[2].set_title(
    "Gyroscope Signal"
)

axes[2].legend()

axes[2].grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    OUTPUT_PNG,
    dpi=200
)


plt.close()


print(
    "Saved:",
    OUTPUT_PNG
)


# ============================================================
# 24. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 80)
print("V10.17 COMPLETE")
print("=" * 80)

print(
    "\nThe diagnosis will tell us whether the degradation is"
)

print(
    "primarily related to sensor distribution, velocity bias,"
)

print(
    "time-dependent model error, or later-drive behavior."
)

print(
    "\nDo NOT build another filter until these results are inspected."
)