"""
V9.8-B
Uncertainty-Aware AI Velocity Estimator

Input:
    Smartphone IMU + magnetometer

Output:
    Vnorth
    Veast
    uncertainty for Vnorth
    uncertainty for Veast

GNSS is used ONLY to generate supervised targets.

The model uses an ensemble of Random Forest regressors.
Prediction spread across trees provides an empirical
uncertainty estimate.

This is a prototype uncertainty model, not a calibrated
probabilistic model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)

PLOT_DIR = (
    ROOT
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.70

N_ESTIMATORS = 150

MAX_DEPTH = 20

MIN_SAMPLES_LEAF = 3

ROLLING_WINDOW = 10


# ============================================================
# LOAD RAW DATA
# ============================================================

print("=" * 80)
print("V9.8-B UNCERTAINTY-AWARE AI VELOCITY")
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
    len(df),
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
# REQUIRED SENSOR COLUMNS
# ============================================================

sensor_columns = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",

    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)",

    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
]


for column in sensor_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing sensor column: {column}"
        )


# ============================================================
# MAGNETOMETER DETECTION
# ============================================================

mag_columns = {}

for axis in ["X", "Y", "Z"]:

    matches = [
        c
        for c in df.columns
        if (
            "MAGNETIC FIELD" in str(c).upper()
            and f" {axis}" in str(c)
        )
    ]

    if not matches:

        raise ValueError(
            f"Magnetometer {axis} column not found."
        )

    mag_columns[axis] = matches[0]


print("\nDetected magnetometer columns:")

for axis, column in mag_columns.items():

    print(
        f"{axis}: {column}"
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for column in sensor_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


for column in mag_columns.values():

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# LOAD VELOCITY TARGETS
# ============================================================

print("\nLoading velocity targets...")

targets = pd.read_csv(
    TARGET_FILE
)

targets["timestamp"] = pd.to_datetime(
    targets["timestamp"],
    errors="coerce",
)

required_targets = [
    "timestamp",
    "vn_mps",
    "ve_mps",
]


for column in required_targets:

    if column not in targets.columns:

        raise ValueError(
            f"Missing target column: {column}"
        )


# ============================================================
# MERGE
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
    targets[
        [
            "timestamp",
            "vn_mps",
            "ve_mps",
        ]
    ],
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(
        milliseconds=60
    ),
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating features...")


ax = df[
    "ACCELEROMETER X (m/s²)"
]

ay = df[
    "ACCELEROMETER Y (m/s²)"
]

az = df[
    "ACCELEROMETER Z (m/s²)"
]


gx = df[
    "GRAVITY X (m/s²)"
]

gy = df[
    "GRAVITY Y (m/s²)"
]

gz = df[
    "GRAVITY Z (m/s²)"
]


gyro_yaw = df[
    "GYROSCOPE Yaw (rad/s)"
]

gyro_pitch = df[
    "GYROSCOPE Pitch (rad/s)"
]

gyro_roll = df[
    "GYROSCOPE Roll (rad/s)"
]


mx = df[
    mag_columns["X"]
]

my = df[
    mag_columns["Y"]
]

mz = df[
    mag_columns["Z"]
]


# ------------------------------------------------------------
# Linear acceleration
# ------------------------------------------------------------

lax = ax - gx
lay = ay - gy
laz = az - gz


# ------------------------------------------------------------
# Magnitudes
# ------------------------------------------------------------

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
# BUILD FEATURE TABLE
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

        "accel_mag": accel_mag,
        "linear_accel_mag": linear_accel_mag,
        "gravity_mag": gravity_mag,

        "gyro_yaw": gyro_yaw,
        "gyro_pitch": gyro_pitch,
        "gyro_roll": gyro_roll,
        "gyro_mag": gyro_mag,

        "mx": mx,
        "my": my,
        "mz": mz,
        "mag_mag": mag_mag,
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


for column in rolling_columns:

    features[
        f"{column}_mean"
    ] = (
        features[column]
        .rolling(
            ROLLING_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    features[
        f"{column}_std"
    ] = (
        features[column]
        .rolling(
            ROLLING_WINDOW,
            min_periods=1,
        )
        .std()
        .fillna(0)
    )


# ============================================================
# VALID DATA
# ============================================================

feature_columns = list(
    features.columns
)

data = pd.concat(
    [
        df[
            [
                "timestamp",
                "vn_mps",
                "ve_mps",
            ]
        ],
        features,
    ],
    axis=1,
)

data = data.dropna(
    subset=[
        *feature_columns,
        "vn_mps",
        "ve_mps",
    ]
).reset_index(
    drop=True
)


print(
    "\nUsable samples:",
    len(data),
)

print(
    "Features:",
    len(feature_columns),
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(
    len(data)
    *
    TRAIN_RATIO
)

train = data.iloc[
    :split_index
].copy()

test = data.iloc[
    split_index:
].copy()


X_train = train[
    feature_columns
]

X_test = test[
    feature_columns
]

y_train_n = train[
    "vn_mps"
]

y_test_n = test[
    "vn_mps"
]

y_train_e = train[
    "ve_mps"
]

y_test_e = test[
    "ve_mps"
]


print("\nTraining samples:", len(train))
print("Testing samples :", len(test))


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining North velocity model...")

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


print("Training East velocity model...")

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
# TREE-LEVEL PREDICTIONS
# ============================================================

print("\nCalculating uncertainty...")

X_test_np = X_test.to_numpy()


# North predictions from individual trees

north_tree_predictions = np.column_stack(
    [
        tree.predict(X_test_np)
        for tree in model_n.estimators_
    ]
)


# East predictions from individual trees

east_tree_predictions = np.column_stack(
    [
        tree.predict(X_test_np)
        for tree in model_e.estimators_
    ]
)


# ============================================================
# MEAN PREDICTION
# ============================================================

vn_pred = (
    north_tree_predictions
    .mean(axis=1)
)

ve_pred = (
    east_tree_predictions
    .mean(axis=1)
)


# ============================================================
# UNCERTAINTY
# ============================================================

vn_std = (
    north_tree_predictions
    .std(axis=1)
)

ve_std = (
    east_tree_predictions
    .std(axis=1)
)


# 95% empirical uncertainty interval

vn_uncertainty_95 = (
    1.96
    *
    vn_std
)

ve_uncertainty_95 = (
    1.96
    *
    ve_std
)


# ============================================================
# SPEED
# ============================================================

true_speed = np.sqrt(
    y_test_n.to_numpy() ** 2
    +
    y_test_e.to_numpy() ** 2
)

ai_speed = np.sqrt(
    vn_pred ** 2
    +
    ve_pred ** 2
)


# ============================================================
# METRICS
# ============================================================

n_mae = mean_absolute_error(
    y_test_n,
    vn_pred,
)

n_rmse = np.sqrt(
    mean_squared_error(
        y_test_n,
        vn_pred,
    )
)

n_r2 = r2_score(
    y_test_n,
    vn_pred,
)


e_mae = mean_absolute_error(
    y_test_e,
    ve_pred,
)

e_rmse = np.sqrt(
    mean_squared_error(
        y_test_e,
        ve_pred,
    )
)

e_r2 = r2_score(
    y_test_e,
    ve_pred,
)


speed_mae = mean_absolute_error(
    true_speed,
    ai_speed,
)

speed_rmse = np.sqrt(
    mean_squared_error(
        true_speed,
        ai_speed,
    )
)


# ============================================================
# UNCERTAINTY STATISTICS
# ============================================================

print("\n")
print("=" * 80)
print("V9.8-B RESULTS")
print("=" * 80)

print(
    f"\nNorth MAE : {n_mae:.4f} m/s"
)

print(
    f"North RMSE: {n_rmse:.4f} m/s"
)

print(
    f"North R²  : {n_r2:.4f}"
)

print(
    f"\nEast MAE  : {e_mae:.4f} m/s"
)

print(
    f"East RMSE : {e_rmse:.4f} m/s"
)

print(
    f"East R²   : {e_r2:.4f}"
)

print(
    f"\nSpeed MAE : "
    f"{speed_mae:.4f} m/s"
    f" ({speed_mae * 3.6:.4f} km/h)"
)

print(
    f"Speed RMSE: "
    f"{speed_rmse:.4f} m/s"
)


print(
    "\nNorth uncertainty:"
)

print(
    f"Mean std: "
    f"{vn_std.mean():.4f} m/s"
)

print(
    f"Median std: "
    f"{np.median(vn_std):.4f} m/s"
)

print(
    f"95th percentile: "
    f"{np.percentile(vn_std, 95):.4f} m/s"
)


print(
    "\nEast uncertainty:"
)

print(
    f"Mean std: "
    f"{ve_std.mean():.4f} m/s"
)

print(
    f"Median std: "
    f"{np.median(ve_std):.4f} m/s"
)

print(
    f"95th percentile: "
    f"{np.percentile(ve_std, 95):.4f} m/s"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

result = pd.DataFrame(
    {
        "timestamp": test[
            "timestamp"
        ].to_numpy(),

        "vn_true_mps": y_test_n.to_numpy(),

        "ve_true_mps": y_test_e.to_numpy(),

        "vn_ai_mps": vn_pred,

        "ve_ai_mps": ve_pred,

        "vn_std_mps": vn_std,

        "ve_std_mps": ve_std,

        "vn_uncertainty_95_mps":
            vn_uncertainty_95,

        "ve_uncertainty_95_mps":
            ve_uncertainty_95,

        "true_speed_mps":
            true_speed,

        "ai_speed_mps":
            ai_speed,

        "true_speed_kmh":
            true_speed * 3.6,

        "ai_speed_kmh":
            ai_speed * 3.6,
    }
)


output_csv = (
    OUTPUT_DIR
    /
    "prototype_v9_8_uncertainty_ai_results.csv"
)

result.to_csv(
    output_csv,
    index=False,
)


# ============================================================
# PLOT 1: UNCERTAINTY
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    result["vn_std_mps"],
    label="North uncertainty",
)

plt.plot(
    result["ve_std_mps"],
    label="East uncertainty",
)

plt.xlabel(
    "Test sample"
)

plt.ylabel(
    "Prediction standard deviation (m/s)"
)

plt.title(
    "AI Velocity Prediction Uncertainty"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

uncertainty_plot = (
    PLOT_DIR
    /
    "prototype_v9_8_uncertainty.png"
)

plt.savefig(
    uncertainty_plot,
    dpi=200,
)

plt.close()


# ============================================================
# PLOT 2: TRUE VS AI SPEED
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    result["true_speed_kmh"].to_numpy(),
    label="GNSS target speed",
)

plt.plot(
    result["ai_speed_kmh"].to_numpy(),
    label="AI speed",
    alpha=0.8,
)

plt.xlabel(
    "Test sample"
)

plt.ylabel(
    "Speed (km/h)"
)

plt.title(
    "AI Velocity Model: True vs Predicted Speed"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

speed_plot = (
    PLOT_DIR
    /
    "prototype_v9_8_ai_speed.png"
)

plt.savefig(
    speed_plot,
    dpi=200,
)

plt.close()


# ============================================================
# PLOT 3: UNCERTAINTY VS ERROR
# ============================================================

north_error = np.abs(
    result["vn_true_mps"]
    -
    result["vn_ai_mps"]
)

east_error = np.abs(
    result["ve_true_mps"]
    -
    result["ve_ai_mps"]
)

total_error = np.sqrt(
    north_error ** 2
    +
    east_error ** 2
)

total_uncertainty = np.sqrt(
    result["vn_std_mps"] ** 2
    +
    result["ve_std_mps"] ** 2
)

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    total_uncertainty,
    total_error,
    s=5,
    alpha=0.25,
)

plt.xlabel(
    "Predicted uncertainty (m/s)"
)

plt.ylabel(
    "Actual velocity error (m/s)"
)

plt.title(
    "AI Uncertainty vs Actual Velocity Error"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

scatter_plot = (
    PLOT_DIR
    /
    "prototype_v9_8_uncertainty_vs_error.png"
)

plt.savefig(
    scatter_plot,
    dpi=200,
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance_north":
            model_n.feature_importances_,
        "importance_east":
            model_e.feature_importances_,
    }
)

importance[
    "importance_mean"
] = (
    importance[
        [
            "importance_north",
            "importance_east",
        ]
    ]
    .mean(axis=1)
)

importance = importance.sort_values(
    "importance_mean",
    ascending=False,
)

importance_file = (
    OUTPUT_DIR
    /
    "prototype_v9_8_uncertainty_feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False,
)


# ============================================================
# FINISH
# ============================================================

print("\n")
print("=" * 80)
print("V9.8-B COMPLETE")
print("=" * 80)

print(
    "\nPrediction file:"
)

print(
    output_csv
)

print(
    "\nUncertainty plot:"
)

print(
    uncertainty_plot
)

print(
    "\nSpeed plot:"
)

print(
    speed_plot
)

print(
    "\nUncertainty vs error:"
)

print(
    scatter_plot
)

print(
    "\nFeature importance:"
)

print(
    importance_file
)

print(
    "\nNext:"
)

print(
    "Use the uncertainty estimates inside the "
    "dead-reckoning fusion layer."
)