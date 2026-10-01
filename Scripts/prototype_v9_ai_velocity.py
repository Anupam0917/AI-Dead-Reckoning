import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

import matplotlib.pyplot as plt


# ============================================================
# V9 STEP 2
# AI NORTH/EAST VELOCITY ESTIMATOR
# ============================================================

print("=" * 70)
print("V9 AI NORTH/EAST VELOCITY ESTIMATION")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

DATA_PATH = (
    "data/raw/"
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

TARGET_PATH = "data/processed/prototype_v9_velocity_targets.csv"

OUTPUT_DIR = "outputs"
PROCESSED_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD RAW DATA
# ------------------------------------------------------------

print("\nLoading raw IO-VNBD dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252"
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print("Rows loaded:", len(df))


# ------------------------------------------------------------
# LOAD V9 TARGETS
# ------------------------------------------------------------

print("\nLoading V9 velocity targets...")

targets = pd.read_csv(TARGET_PATH)

print("Target rows:", len(targets))


# ------------------------------------------------------------
# PARSE TIMESTAMPS
# ------------------------------------------------------------

timestamp_col = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"

df["timestamp"] = pd.to_datetime(
    df[timestamp_col],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)

targets["timestamp"] = pd.to_datetime(
    targets["timestamp"],
    errors="coerce"
)


# ------------------------------------------------------------
# SENSOR COLUMNS
# ------------------------------------------------------------

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

    "MAGNETIC FIELD X (Î¼T)",
    "MAGNETIC FIELD Y (Î¼T)",
    "MAGNETIC FIELD Z (Î¼T)"
]


print("\nChecking sensor columns...")

missing = [
    c for c in sensor_columns
    if c not in df.columns
]

if missing:
    print("Missing columns:")
    for c in missing:
        print(" ", c)

    raise ValueError(
        "Required sensor columns are missing."
    )

print("All sensor columns found.")


# ------------------------------------------------------------
# BUILD SENSOR DATAFRAME
# ------------------------------------------------------------

data = df[
    ["timestamp"] + sensor_columns
].copy()


# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

for col in sensor_columns:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )


# ------------------------------------------------------------
# BASIC SENSOR CLEANING
# ------------------------------------------------------------

data[sensor_columns] = (
    data[sensor_columns]
    .replace([np.inf, -np.inf], np.nan)
)

data[sensor_columns] = (
    data[sensor_columns]
    .interpolate(
        method="linear",
        limit_direction="both"
    )
)

data[sensor_columns] = (
    data[sensor_columns]
    .fillna(0)
)


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

print("\nCreating sensor features...")


# Accelerometer
ax = data["ACCELEROMETER X (m/s²)"]
ay = data["ACCELEROMETER Y (m/s²)"]
az = data["ACCELEROMETER Z (m/s²)"]


# Gravity
gx = data["GRAVITY X (m/s²)"]
gy = data["GRAVITY Y (m/s²)"]
gz = data["GRAVITY Z (m/s²)"]


# Linear acceleration
data["LIN_ACC_X"] = ax - gx
data["LIN_ACC_Y"] = ay - gy
data["LIN_ACC_Z"] = az - gz


# Accelerometer magnitude
data["ACC_MAG"] = np.sqrt(
    ax**2 +
    ay**2 +
    az**2
)


# Linear acceleration magnitude
data["LIN_ACC_MAG"] = np.sqrt(
    data["LIN_ACC_X"]**2 +
    data["LIN_ACC_Y"]**2 +
    data["LIN_ACC_Z"]**2
)


# Gyroscope
gyro_columns = [
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)"
]


data["GYRO_MAG"] = np.sqrt(
    data["GYROSCOPE Yaw (rad/s)"]**2 +
    data["GYROSCOPE Pitch (rad/s)"]**2 +
    data["GYROSCOPE Roll (rad/s)"]**2
)


# Magnetometer
mag_columns = [
    "MAGNETIC FIELD X (Î¼T)",
    "MAGNETIC FIELD Y (Î¼T)",
    "MAGNETIC FIELD Z (Î¼T)"
]


data["MAG_MAG"] = np.sqrt(
    data["MAGNETIC FIELD X (Î¼T)"]**2 +
    data["MAGNETIC FIELD Y (Î¼T)"]**2 +
    data["MAGNETIC FIELD Z (Î¼T)"]**2
)


# ------------------------------------------------------------
# ROLLING FEATURES
# ------------------------------------------------------------

rolling_columns = [
    "LIN_ACC_X",
    "LIN_ACC_Y",
    "LIN_ACC_Z",
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
    "MAGNETIC FIELD X (Î¼T)",
    "MAGNETIC FIELD Y (Î¼T)",
    "MAGNETIC FIELD Z (Î¼T)"
]


for col in rolling_columns:

    data[col + "_MEAN"] = (
        data[col]
        .rolling(
            window=10,
            min_periods=1
        )
        .mean()
    )

    data[col + "_STD"] = (
        data[col]
        .rolling(
            window=10,
            min_periods=1
        )
        .std()
        .fillna(0)
    )


# ------------------------------------------------------------
# FEATURE LIST
# ------------------------------------------------------------

feature_columns = sensor_columns + [
    "LIN_ACC_X",
    "LIN_ACC_Y",
    "LIN_ACC_Z",
    "ACC_MAG",
    "LIN_ACC_MAG",
    "GYRO_MAG",
    "MAG_MAG"
]


for col in rolling_columns:
    feature_columns.append(col + "_MEAN")
    feature_columns.append(col + "_STD")


print("\nTotal features:", len(feature_columns))


# ------------------------------------------------------------
# MERGE TARGETS
# ------------------------------------------------------------

print("\nMerging AI inputs with V9 targets...")

target_columns = [
    "vn_mps",
    "ve_mps"
]

targets = targets[
    ["timestamp"] + target_columns
].copy()


data = pd.merge_asof(
    data.sort_values("timestamp"),
    targets.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("150ms")
)


# ------------------------------------------------------------
# REMOVE INVALID TARGETS
# ------------------------------------------------------------

data = data.dropna(
    subset=target_columns
).reset_index(drop=True)


print("Training samples available:", len(data))


# ------------------------------------------------------------
# PREPARE X / Y
# ------------------------------------------------------------

X = data[feature_columns].values

Y = data[target_columns].values


# ------------------------------------------------------------
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ------------------------------------------------------------

split_index = int(
    len(data) * 0.70
)

X_train = X[:split_index]
Y_train = Y[:split_index]

X_test = X[split_index:]
Y_test = Y[split_index:]


time_train = data["timestamp"].iloc[:split_index]

time_test = data["timestamp"].iloc[split_index:]


print("\nChronological split")
print("-------------------")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Training period:",
    time_train.iloc[0],
    "->",
    time_train.iloc[-1]
)

print(
    "Testing period:",
    time_test.iloc[0],
    "->",
    time_test.iloc[-1]
)


# ------------------------------------------------------------
# TRAIN RANDOM FOREST
# ------------------------------------------------------------

print("\nTraining Random Forest...")


model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    Y_train
)


print("Training complete.")


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

print("\nGenerating velocity predictions...")


Y_pred = model.predict(X_test)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

vn_true = Y_test[:, 0]
ve_true = Y_test[:, 1]

vn_pred = Y_pred[:, 0]
ve_pred = Y_pred[:, 1]


vn_mae = mean_absolute_error(
    vn_true,
    vn_pred
)

ve_mae = mean_absolute_error(
    ve_true,
    ve_pred
)


vn_rmse = np.sqrt(
    mean_squared_error(
        vn_true,
        vn_pred
    )
)

ve_rmse = np.sqrt(
    mean_squared_error(
        ve_true,
        ve_pred
    )
)


vn_r2 = r2_score(
    vn_true,
    vn_pred
)

ve_r2 = r2_score(
    ve_true,
    ve_pred
)


# Speed magnitude
true_speed = np.sqrt(
    vn_true**2 +
    ve_true**2
)

pred_speed = np.sqrt(
    vn_pred**2 +
    ve_pred**2
)


speed_mae = mean_absolute_error(
    true_speed,
    pred_speed
)

speed_rmse = np.sqrt(
    mean_squared_error(
        true_speed,
        pred_speed
    )
)


print("\n" + "=" * 70)
print("V9 AI VELOCITY RESULTS")
print("=" * 70)

print("\nNorth velocity:")
print(
    f"  MAE  : {vn_mae:.4f} m/s"
)

print(
    f"  RMSE : {vn_rmse:.4f} m/s"
)

print(
    f"  R²   : {vn_r2:.4f}"
)


print("\nEast velocity:")
print(
    f"  MAE  : {ve_mae:.4f} m/s"
)

print(
    f"  RMSE : {ve_rmse:.4f} m/s"
)

print(
    f"  R²   : {ve_r2:.4f}"
)


print("\nSpeed magnitude:")
print(
    f"  MAE  : {speed_mae:.4f} m/s"
)

print(
    f"  RMSE : {speed_rmse:.4f} m/s"
)

print(
    f"  MAE  : {speed_mae * 3.6:.4f} km/h"
)


# ------------------------------------------------------------
# SAVE PREDICTIONS
# ------------------------------------------------------------

results = pd.DataFrame({

    "timestamp": time_test.values,

    "vnorth_true_mps": vn_true,

    "veast_true_mps": ve_true,

    "vnorth_ai_mps": vn_pred,

    "veast_ai_mps": ve_pred,

    "true_speed_mps": true_speed,

    "ai_speed_mps": pred_speed,

    "true_speed_kmh": true_speed * 3.6,

    "ai_speed_kmh": pred_speed * 3.6
})


results_path = (
    "data/processed/"
    "prototype_v9_ai_velocity_results.csv"
)


results.to_csv(
    results_path,
    index=False
)


# ------------------------------------------------------------
# PLOT 1: VELOCITY COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(14, 7))

plot_n = min(
    3000,
    len(results)
)

plt.plot(
    results["timestamp"].iloc[:plot_n],
    results["vnorth_true_mps"].iloc[:plot_n],
    label="True Vnorth"
)

plt.plot(
    results["timestamp"].iloc[:plot_n],
    results["vnorth_ai_mps"].iloc[:plot_n],
    label="AI Vnorth"
)

plt.plot(
    results["timestamp"].iloc[:plot_n],
    results["veast_true_mps"].iloc[:plot_n],
    label="True Veast"
)

plt.plot(
    results["timestamp"].iloc[:plot_n],
    results["veast_ai_mps"].iloc[:plot_n],
    label="AI Veast"
)

plt.xlabel("Time")

plt.ylabel(
    "Velocity (m/s)"
)

plt.title(
    "V9 AI North/East Velocity Estimation"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


velocity_plot = (
    "outputs/"
    "prototype_v9_ai_velocity.png"
)

plt.savefig(
    velocity_plot,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# PLOT 2: SPEED COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(14, 7))

plot_n = min(
    3000,
    len(results)
)

plt.plot(
    results["timestamp"].iloc[:plot_n],
    results["true_speed_kmh"].iloc[:plot_n],
    label="GNSS Target Speed"
)

plt.plot(
    results["timestamp"].iloc[:plot_n],
    results["ai_speed_kmh"].iloc[:plot_n],
    label="AI Speed"
)

plt.xlabel("Time")

plt.ylabel(
    "Speed (km/h)"
)

plt.title(
    "V9 AI Speed Estimation from Smartphone Sensors"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


speed_plot = (
    "outputs/"
    "prototype_v9_ai_speed.png"
)

plt.savefig(
    speed_plot,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

importance = pd.DataFrame({

    "feature": feature_columns,

    "importance": model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


importance_path = (
    "data/processed/"
    "prototype_v9_feature_importance.csv"
)


importance.to_csv(
    importance_path,
    index=False
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V9 STEP 2 COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(
    "1.",
    results_path
)

print(
    "2.",
    velocity_plot
)

print(
    "3.",
    speed_plot
)

print(
    "4.",
    importance_path
)

print("\nAI input:")
print("Smartphone IMU + magnetometer")

print("\nAI output:")
print("Vnorth + Veast")

print("\nGNSS is used only as supervised training target.")
print("=" * 70)