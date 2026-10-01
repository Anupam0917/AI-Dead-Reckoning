"""
V10.1 - Full-Dataset AI Velocity Prediction

Purpose:
Generate V9.8-style AI North/East velocity predictions
for ALL samples so the subsequent EKF has continuous
AI velocity measurements.

GNSS is used only as the supervised training target.

No GNSS position is used as an input feature.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


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

TARGET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "prototype_v9_velocity_targets.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "prototype_v10_1_full_ai_velocity.csv"
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 150

MAX_DEPTH = 20

MIN_SAMPLES_LEAF = 3

TRAIN_RATIO = 0.70


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("V10.1 FULL-DATASET AI VELOCITY")
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


# ============================================================
# NUMERIC
# ============================================================

base_columns = [
    ACC_X,
    ACC_Y,
    ACC_Z,
    GRAV_X,
    GRAV_Y,
    GRAV_Z,
    GYRO_YAW,
    GYRO_PITCH,
    GYRO_ROLL,
]

for column in base_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# MAGNETOMETER
# ============================================================

mag_columns = {}

for axis in ["X", "Y", "Z"]:

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
            f"Magnetometer {axis} not found."
        )

    mag_columns[axis] = matches[0]

    df[
        mag_columns[axis]
    ] = pd.to_numeric(
        df[
            mag_columns[axis]
        ],
        errors="coerce",
    )


print("\nMagnetometer:")

for axis, column in mag_columns.items():

    print(
        f"{axis}: {column}"
    )


# ============================================================
# LOAD TARGETS
# ============================================================

print(
    "\nLoading velocity targets..."
)

targets = pd.read_csv(
    TARGET_FILE
)

targets["timestamp"] = pd.to_datetime(
    targets["timestamp"],
    errors="coerce",
)


targets = targets[
    [
        "timestamp",
        "vn_mps",
        "ve_mps",
    ]
].copy()


# ============================================================
# MERGE TARGET
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

targets = targets.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


df = pd.merge_asof(
    df,
    targets,
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(
        milliseconds=60
    ),
)


print(
    "Rows after merge:",
    len(df)
)


# ============================================================
# SENSOR VARIABLES
# ============================================================

ax = df[ACC_X]
ay = df[ACC_Y]
az = df[ACC_Z]

gx = df[GRAV_X]
gy = df[GRAV_Y]
gz = df[GRAV_Z]

gyro_yaw = df[GYRO_YAW]
gyro_pitch = df[GYRO_PITCH]
gyro_roll = df[GYRO_ROLL]

mx = df[
    mag_columns["X"]
]

my = df[
    mag_columns["Y"]
]

mz = df[
    mag_columns["Z"]
]


# ============================================================
# DERIVED FEATURES
# ============================================================

lax = ax - gx
lay = ay - gy
laz = az - gz


accel_mag = np.sqrt(
    ax ** 2
    +
    ay ** 2
    +
    az ** 2
)


linear_accel_mag = np.sqrt(
    lax ** 2
    +
    lay ** 2
    +
    laz ** 2
)


gravity_mag = np.sqrt(
    gx ** 2
    +
    gy ** 2
    +
    gz ** 2
)


gyro_mag = np.sqrt(
    gyro_yaw ** 2
    +
    gyro_pitch ** 2
    +
    gyro_roll ** 2
)


mag_mag = np.sqrt(
    mx ** 2
    +
    my ** 2
    +
    mz ** 2
)


# ============================================================
# FEATURE TABLE
# ============================================================

features = pd.DataFrame(
    {
        "ax": ax,
        "ay": ay,
        "az": az,

        "gx": gx,
        "gy": gy,
        "gz": gz,

        "lax": lax,
        "lay": lay,
        "laz": laz,

        "accel_mag":
            accel_mag,

        "linear_accel_mag":
            linear_accel_mag,

        "gravity_mag":
            gravity_mag,

        "gyro_yaw":
            gyro_yaw,

        "gyro_pitch":
            gyro_pitch,

        "gyro_roll":
            gyro_roll,

        "gyro_mag":
            gyro_mag,

        "mx": mx,
        "my": my,
        "mz": mz,

        "mag_mag":
            mag_mag,
    }
)


# ============================================================
# ROLLING FEATURES
# ============================================================

rolling_columns = [
    "lax",
    "lay",
    "laz",
    "linear_accel_mag",
    "gyro_mag",
    "mag_mag",
]


WINDOW = 20


for column in rolling_columns:

    features[
        f"{column}_mean"
    ] = (
        features[column]
        .rolling(
            WINDOW,
            min_periods=1,
        )
        .mean()
    )

    features[
        f"{column}_std"
    ] = (
        features[column]
        .rolling(
            WINDOW,
            min_periods=1,
        )
        .std()
        .fillna(0)
    )


# ============================================================
# BUILD DATASET
# ============================================================

feature_columns = list(
    features.columns
)


data = features.copy()

data["timestamp"] = (
    df["timestamp"]
)

data["vn_true"] = (
    df["vn_mps"]
)

data["ve_true"] = (
    df["ve_mps"]
)


data = data.replace(
    [np.inf, -np.inf],
    np.nan,
)


data = data.dropna(
    subset=[
        *feature_columns,
        "vn_true",
        "ve_true",
    ]
).reset_index(
    drop=True
)


print(
    "\nUsable samples:",
    len(data)
)

print(
    "Features:",
    len(feature_columns)
)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split = int(
    len(data)
    *
    TRAIN_RATIO
)


train = data.iloc[
    :split
].copy()

test = data.iloc[
    split:
].copy()


X_train = train[
    feature_columns
]

X_test = test[
    feature_columns
]

y_train_n = train[
    "vn_true"
]

y_train_e = train[
    "ve_true"
]


# ============================================================
# TRAIN MODELS
# ============================================================

print(
    "\nTraining North model..."
)

model_n = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model_n.fit(
    X_train,
    y_train_n,
)


print(
    "Training East model..."
)

model_e = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE + 1,
    n_jobs=-1,
)

model_e.fit(
    X_train,
    y_train_e,
)


# ============================================================
# PREDICT ALL DATA
# ============================================================

print(
    "\nPredicting complete dataset..."
)

X_all = data[
    feature_columns
].to_numpy()


# Tree predictions for uncertainty

north_tree_predictions = np.column_stack(
    [
        tree.predict(
            X_all
        )
        for tree in model_n.estimators_
    ]
)


east_tree_predictions = np.column_stack(
    [
        tree.predict(
            X_all
        )
        for tree in model_e.estimators_
    ]
)


vn_ai = (
    north_tree_predictions
    .mean(axis=1)
)

ve_ai = (
    east_tree_predictions
    .mean(axis=1)
)


vn_std = (
    north_tree_predictions
    .std(axis=1)
)

ve_std = (
    east_tree_predictions
    .std(axis=1)
)


# ============================================================
# OUTPUT
# ============================================================

result = pd.DataFrame(
    {
        "timestamp":
            data[
                "timestamp"
            ].to_numpy(),

        "vn_true_mps":
            data[
                "vn_true"
            ].to_numpy(),

        "ve_true_mps":
            data[
                "ve_true"
            ].to_numpy(),

        "vn_ai_mps":
            vn_ai,

        "ve_ai_mps":
            ve_ai,

        "vn_std_mps":
            vn_std,

        "ve_std_mps":
            ve_std,
    }
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# TEST METRICS
# ============================================================

test_mask = np.arange(
    len(data)
) >= split


true_n = (
    result.loc[
        test_mask,
        "vn_true_mps"
    ]
    .to_numpy()
)

pred_n = (
    result.loc[
        test_mask,
        "vn_ai_mps"
    ]
    .to_numpy()
)

true_e = (
    result.loc[
        test_mask,
        "ve_true_mps"
    ]
    .to_numpy()
)

pred_e = (
    result.loc[
        test_mask,
        "ve_ai_mps"
    ]
    .to_numpy()
)


north_mae = np.mean(
    np.abs(
        true_n
        -
        pred_n
    )
)


east_mae = np.mean(
    np.abs(
        true_e
        -
        pred_e
    )
)


north_rmse = np.sqrt(
    np.mean(
        (
            true_n
            -
            pred_n
        ) ** 2
    )
)


east_rmse = np.sqrt(
    np.mean(
        (
            true_e
            -
            pred_e
        ) ** 2
    )
)


true_speed = np.sqrt(
    true_n ** 2
    +
    true_e ** 2
)


pred_speed = np.sqrt(
    pred_n ** 2
    +
    pred_e ** 2
)


speed_mae = np.mean(
    np.abs(
        true_speed
        -
        pred_speed
    )
)


# ============================================================
# PRINT
# ============================================================

print("\n")
print("=" * 80)
print("V10.1 FULL AI VELOCITY RESULTS")
print("=" * 80)

print(
    f"\nNorth MAE:"
    f" {north_mae:.4f} m/s"
)

print(
    f"North RMSE:"
    f" {north_rmse:.4f} m/s"
)

print(
    f"\nEast MAE:"
    f" {east_mae:.4f} m/s"
)

print(
    f"East RMSE:"
    f" {east_rmse:.4f} m/s"
)

print(
    f"\nSpeed MAE:"
    f" {speed_mae:.4f} m/s"
)

print(
    f"Speed MAE:"
    f" {speed_mae * 3.6:.4f} km/h"
)


print(
    "\nAI predictions saved:"
)

print(
    OUTPUT_FILE
)


print(
    "\nNext:"
)

print(
    "Run V10.1 EKF using this complete AI velocity file."
)