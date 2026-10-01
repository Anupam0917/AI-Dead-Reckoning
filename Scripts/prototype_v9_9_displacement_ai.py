"""
V9.9 - Direct Displacement AI

Instead of predicting velocity and integrating it repeatedly,
the model predicts displacement over a short temporal window:

    ΔNorth
    ΔEast

Input:
    Smartphone accelerometer
    Gravity
    Gyroscope
    Magnetometer

Target:
    GNSS-derived displacement over the prediction horizon

GNSS is used only for supervised target generation.

This is an experimental prototype. Performance must be
evaluated on held-out time periods.
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

RANDOM_STATE = 42

TRAIN_RATIO = 0.70

N_ESTIMATORS = 150

MAX_DEPTH = 20

MIN_SAMPLES_LEAF = 3

WINDOW = 20

# 20 samples at approximately 10 Hz
# represents approximately 2 seconds.

HORIZON = 10

# Predict displacement from the current temporal
# sensor context toward the point HORIZON samples ahead.


# ============================================================
# LOAD RAW DATA
# ============================================================

print("=" * 80)
print("V9.9 DIRECT DISPLACEMENT AI")
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
# MAGNETOMETER
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


print("\nMagnetometer:")

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
# LOAD CANONICAL REFERENCE
# ============================================================

print("\nLoading canonical GNSS reference...")

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
        "time_s",
        "north_m",
        "east_m",
    ]
].copy()


# ============================================================
# MERGE REFERENCE
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

reference = reference.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

df = pd.merge_asof(
    df,
    reference,
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(
        milliseconds=60
    ),
)

print(
    "Merged rows:",
    len(df)
)


# ============================================================
# SENSOR FEATURES
# ============================================================

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


# ============================================================
# LINEAR ACCELERATION
# ============================================================

lax = ax - gx
lay = ay - gy
laz = az - gz


# ============================================================
# MAGNITUDES
# ============================================================

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
# TARGET DISPLACEMENT
# ============================================================

north = df[
    "north_m"
].to_numpy()

east = df[
    "east_m"
].to_numpy()


target_north = (
    np.roll(
        north,
        -HORIZON,
    )
    -
    north
)

target_east = (
    np.roll(
        east,
        -HORIZON,
    )
    -
    east
)


# Last HORIZON rows cannot have a future target.

target_north[
    -HORIZON:
] = np.nan

target_east[
    -HORIZON:
] = np.nan


# ============================================================
# BUILD TRAINING DATA
# ============================================================

data = features.copy()

data["timestamp"] = (
    df["timestamp"]
)

data["time_s"] = (
    df["time_s"]
)

data["delta_north_m"] = (
    target_north
)

data["delta_east_m"] = (
    target_east
)


feature_columns = list(
    features.columns
)


data = data.dropna(
    subset=[
        *feature_columns,
        "delta_north_m",
        "delta_east_m",
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
    "delta_north_m"
]

y_test_n = test[
    "delta_north_m"
]

y_train_e = train[
    "delta_east_m"
]

y_test_e = test[
    "delta_east_m"
]


print(
    "\nTraining samples:",
    len(train)
)

print(
    "Testing samples:",
    len(test)
)


# ============================================================
# TRAIN NORTH MODEL
# ============================================================

print(
    "\nTraining North displacement model..."
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


# ============================================================
# TRAIN EAST MODEL
# ============================================================

print(
    "Training East displacement model..."
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
# PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)

X_test_np = X_test.to_numpy()


# Mean prediction across trees

north_tree_predictions = np.column_stack(
    [
        tree.predict(
            X_test_np
        )
        for tree in model_n.estimators_
    ]
)

east_tree_predictions = np.column_stack(
    [
        tree.predict(
            X_test_np
        )
        for tree in model_e.estimators_
    ]
)


delta_n_pred = (
    north_tree_predictions
    .mean(axis=1)
)

delta_e_pred = (
    east_tree_predictions
    .mean(axis=1)
)


# ============================================================
# UNCERTAINTY
# ============================================================

delta_n_std = (
    north_tree_predictions
    .std(axis=1)
)

delta_e_std = (
    east_tree_predictions
    .std(axis=1)
)


# ============================================================
# METRICS
# ============================================================

n_mae = mean_absolute_error(
    y_test_n,
    delta_n_pred,
)

n_rmse = np.sqrt(
    mean_squared_error(
        y_test_n,
        delta_n_pred,
    )
)

n_r2 = r2_score(
    y_test_n,
    delta_n_pred,
)


e_mae = mean_absolute_error(
    y_test_e,
    delta_e_pred,
)

e_rmse = np.sqrt(
    mean_squared_error(
        y_test_e,
        delta_e_pred,
    )
)

e_r2 = r2_score(
    y_test_e,
    delta_e_pred,
)


true_distance = np.sqrt(
    y_test_n.to_numpy() ** 2
    +
    y_test_e.to_numpy() ** 2
)

predicted_distance = np.sqrt(
    delta_n_pred ** 2
    +
    delta_e_pred ** 2
)

distance_mae = mean_absolute_error(
    true_distance,
    predicted_distance,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("V9.9 RESULTS")
print("=" * 80)

print(
    f"\nNorth displacement MAE:"
    f" {n_mae:.4f} m"
)

print(
    f"North displacement RMSE:"
    f" {n_rmse:.4f} m"
)

print(
    f"North displacement R²:"
    f" {n_r2:.4f}"
)

print(
    f"\nEast displacement MAE:"
    f" {e_mae:.4f} m"
)

print(
    f"East displacement RMSE:"
    f" {e_rmse:.4f} m"
)

print(
    f"East displacement R²:"
    f" {e_r2:.4f}"
)

print(
    f"\nDisplacement magnitude MAE:"
    f" {distance_mae:.4f} m"
)

print(
    f"\nNorth uncertainty mean:"
    f" {delta_n_std.mean():.4f} m"
)

print(
    f"East uncertainty mean:"
    f" {delta_e_std.mean():.4f} m"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

result = pd.DataFrame(
    {
        "timestamp":
            test["timestamp"].to_numpy(),

        "time_s":
            test["time_s"].to_numpy(),

        "true_delta_north_m":
            y_test_n.to_numpy(),

        "true_delta_east_m":
            y_test_e.to_numpy(),

        "ai_delta_north_m":
            delta_n_pred,

        "ai_delta_east_m":
            delta_e_pred,

        "delta_north_std_m":
            delta_n_std,

        "delta_east_std_m":
            delta_e_std,

        "true_displacement_m":
            true_distance,

        "ai_displacement_m":
            predicted_distance,
    }
)


output_file = (
    OUTPUT_DIR
    /
    "prototype_v9_9_displacement_ai_results.csv"
)

result.to_csv(
    output_file,
    index=False,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "feature": feature_columns,

        "north_importance":
            model_n.feature_importances_,

        "east_importance":
            model_e.feature_importances_,
    }
)

importance["mean_importance"] = (
    importance[
        [
            "north_importance",
            "east_importance",
        ]
    ]
    .mean(axis=1)
)

importance = importance.sort_values(
    "mean_importance",
    ascending=False,
)

importance_file = (
    OUTPUT_DIR
    /
    "prototype_v9_9_displacement_feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False,
)


# ============================================================
# PLOT: TRUE VS AI DISPLACEMENT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    result["true_displacement_m"].to_numpy(),
    label="GNSS displacement",
)

plt.plot(
    result["ai_displacement_m"].to_numpy(),
    label="AI displacement",
    alpha=0.8,
)

plt.xlabel(
    "Test sample"
)

plt.ylabel(
    "Displacement (m)"
)

plt.title(
    "V9.9 Direct Displacement Prediction"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plot_file = (
    OUTPUTS
    /
    "prototype_v9_9_displacement.png"
)

plt.savefig(
    plot_file,
    dpi=200,
)

plt.close()


# ============================================================
# PLOT: UNCERTAINTY
# ============================================================

total_uncertainty = np.sqrt(
    delta_n_std ** 2
    +
    delta_e_std ** 2
)

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    total_uncertainty
)

plt.xlabel(
    "Test sample"
)

plt.ylabel(
    "Displacement uncertainty (m)"
)

plt.title(
    "V9.9 Displacement Prediction Uncertainty"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

uncertainty_plot = (
    OUTPUTS
    /
    "prototype_v9_9_uncertainty.png"
)

plt.savefig(
    uncertainty_plot,
    dpi=200,
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("V9.9 COMPLETE")
print("=" * 80)

print(
    "\nPrediction file:"
)

print(
    output_file
)

print(
    "\nFeature importance:"
)

print(
    importance_file
)

print(
    "\nDisplacement plot:"
)

print(
    plot_file
)

print(
    "\nUncertainty plot:"
)

print(
    uncertainty_plot
)

print(
    "\nNext:"
)

print(
    "Evaluate the learned displacement model "
    "inside the GNSS-denied navigation window."
)