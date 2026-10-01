import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    "prototype_v10_9_sensor_ablation.csv"
)


print("=" * 70)
print("V10.9 SENSOR ABLATION STUDY")
print("=" * 70)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("\n[1] Loading raw dataset...")

raw = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

raw.columns = raw.columns.str.strip()

raw["timestamp"] = pd.to_datetime(
    raw["DATE (YYYY-MO-DD HH-MI-SS_SSS)"],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)

print("Rows:", len(raw))


# ============================================================
# 2. FIND MAGNETOMETER
# ============================================================

mag_x = None
mag_y = None
mag_z = None

for column in raw.columns:

    name = column.upper()

    if "MAGNETIC FIELD" in name:

        if " X " in name:
            mag_x = column

        elif " Y " in name:
            mag_y = column

        elif " Z " in name:
            mag_z = column


print("\nMagnetometer columns:")
print("X:", mag_x)
print("Y:", mag_y)
print("Z:", mag_z)


# ============================================================
# 3. CONVERT SENSOR DATA
# ============================================================

print("\n[2] Preparing sensors...")


def num(column):
    return pd.to_numeric(
        raw[column],
        errors="coerce"
    )


features = pd.DataFrame({
    "timestamp": raw["timestamp"],

    "acc_x": num(
        "ACCELEROMETER X (m/s²)"
    ),

    "acc_y": num(
        "ACCELEROMETER Y (m/s²)"
    ),

    "acc_z": num(
        "ACCELEROMETER Z (m/s²)"
    ),

    "gravity_x": num(
        "GRAVITY X (m/s²)"
    ),

    "gravity_y": num(
        "GRAVITY Y (m/s²)"
    ),

    "gravity_z": num(
        "GRAVITY Z (m/s²)"
    ),

    "gyro_yaw": num(
        "GYROSCOPE Yaw (rad/s)"
    ),

    "gyro_pitch": num(
        "GYROSCOPE Pitch (rad/s)"
    ),

    "gyro_roll": num(
        "GYROSCOPE Roll (rad/s)"
    ),

    "mag_x": num(mag_x),

    "mag_y": num(mag_y),

    "mag_z": num(mag_z)
})


# ============================================================
# 4. DERIVED FEATURES
# ============================================================

print("[3] Creating derived features...")


features["acc_magnitude"] = np.sqrt(
    features["acc_x"] ** 2 +
    features["acc_y"] ** 2 +
    features["acc_z"] ** 2
)

features["gravity_magnitude"] = np.sqrt(
    features["gravity_x"] ** 2 +
    features["gravity_y"] ** 2 +
    features["gravity_z"] ** 2
)

features["gyro_magnitude"] = np.sqrt(
    features["gyro_yaw"] ** 2 +
    features["gyro_pitch"] ** 2 +
    features["gyro_roll"] ** 2
)

features["mag_magnitude"] = np.sqrt(
    features["mag_x"] ** 2 +
    features["mag_y"] ** 2 +
    features["mag_z"] ** 2
)

features["linear_acc_x"] = (
    features["acc_x"] -
    features["gravity_x"]
)

features["linear_acc_y"] = (
    features["acc_y"] -
    features["gravity_y"]
)

features["linear_acc_z"] = (
    features["acc_z"] -
    features["gravity_z"]
)

features["linear_acc_magnitude"] = np.sqrt(
    features["linear_acc_x"] ** 2 +
    features["linear_acc_y"] ** 2 +
    features["linear_acc_z"] ** 2
)


# ============================================================
# 5. ROLLING FEATURES
# ============================================================

print("[4] Creating rolling features...")

rolling_columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "linear_acc_x",
    "linear_acc_y",
    "linear_acc_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",
    "mag_x",
    "mag_y",
    "mag_z"
]

rolling_parts = []

for window in [5, 10, 20]:

    print("Window:", window)

    temp = {}

    for column in rolling_columns:

        temp[
            column + "_mean_" + str(window)
        ] = (
            features[column]
            .rolling(
                window=window,
                min_periods=1
            )
            .mean()
        )

        temp[
            column + "_std_" + str(window)
        ] = (
            features[column]
            .rolling(
                window=window,
                min_periods=1
            )
            .std()
            .fillna(0)
        )

    rolling_parts.append(
        pd.DataFrame(
            temp,
            index=features.index
        )
    )


features = pd.concat(
    [
        features,
        rolling_parts[0],
        rolling_parts[1],
        rolling_parts[2]
    ],
    axis=1
)

print(
    "Total features:",
    len(features.columns) - 1
)


# ============================================================
# 6. LOAD TARGETS
# ============================================================

print("\n[5] Loading velocity targets...")

target = pd.read_csv(
    TARGET_FILE
)

target["timestamp"] = pd.to_datetime(
    target["timestamp"]
)

target = target.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# 7. MERGE
# ============================================================

print("[6] Matching sensor and target data...")

features = features.sort_values(
    "timestamp"
)

data = pd.merge_asof(
    target[
        [
            "timestamp",
            "time_s",
            "vn_mps",
            "ve_mps"
        ]
    ].sort_values("timestamp"),

    features,

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        "50ms"
    )
)


# ============================================================
# 8. CLEAN
# ============================================================

print("[7] Cleaning data...")

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna(
    subset=[
        "vn_mps",
        "ve_mps"
    ]
)

sensor_columns = [
    column
    for column in features.columns
    if column != "timestamp"
]

data = data.dropna(
    subset=sensor_columns
)

print(
    "Usable rows:",
    len(data)
)


# ============================================================
# 9. SENSOR GROUP DEFINITIONS
# ============================================================

groups = {

    "ALL": [
        "acc_",
        "linear_acc_",
        "gravity_",
        "gyro_",
        "mag_"
    ],

    "ACC_ONLY": [
        "acc_",
        "linear_acc_"
    ],

    "ACC_GYRO": [
        "acc_",
        "linear_acc_",
        "gyro_"
    ],

    "ACC_GYRO_GRAVITY": [
        "acc_",
        "linear_acc_",
        "gravity_",
        "gyro_"
    ],

    "ACC_GYRO_MAG": [
        "acc_",
        "linear_acc_",
        "gyro_",
        "mag_"
    ],

    "GYRO_MAG": [
        "gyro_",
        "mag_"
    ],

    "NO_MAG": [
        "acc_",
        "linear_acc_",
        "gravity_",
        "gyro_"
    ]
}


# ============================================================
# 10. CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(
    len(data) * 0.70
)

train = data.iloc[
    :split_index
]

test = data.iloc[
    split_index:
]

print("\nTraining rows:", len(train))
print("Testing rows:", len(test))


# ============================================================
# 11. RUN MODELS
# ============================================================

results = []

for model_name in groups:

    print("\n")
    print("=" * 70)
    print("MODEL:", model_name)
    print("=" * 70)

    prefixes = groups[model_name]

    selected = []

    for column in sensor_columns:

        for prefix in prefixes:

            if column.startswith(prefix):

                selected.append(column)
                break

    print(
        "Selected features:",
        len(selected)
    )

    # ----------------------------------------
    # INPUT
    # ----------------------------------------

    X_train = train[selected]
    X_test = test[selected]

    y_train_n = train["vn_mps"]
    y_test_n = test["vn_mps"]

    y_train_e = train["ve_mps"]
    y_test_e = test["ve_mps"]


    # ----------------------------------------
    # NORTH MODEL
    # ----------------------------------------

    print("Training North model...")

    north_model = RandomForestRegressor(
        n_estimators=75,
        max_depth=18,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    north_model.fit(
        X_train,
        y_train_n
    )

    north_prediction = north_model.predict(
        X_test
    )


    # ----------------------------------------
    # EAST MODEL
    # ----------------------------------------

    print("Training East model...")

    east_model = RandomForestRegressor(
        n_estimators=75,
        max_depth=18,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    east_model.fit(
        X_train,
        y_train_e
    )

    east_prediction = east_model.predict(
        X_test
    )


    # ----------------------------------------
    # METRICS
    # ----------------------------------------

    north_mae = mean_absolute_error(
        y_test_n,
        north_prediction
    )

    north_rmse = np.sqrt(
        mean_squared_error(
            y_test_n,
            north_prediction
        )
    )

    north_r2 = r2_score(
        y_test_n,
        north_prediction
    )


    east_mae = mean_absolute_error(
        y_test_e,
        east_prediction
    )

    east_rmse = np.sqrt(
        mean_squared_error(
            y_test_e,
            east_prediction
        )
    )

    east_r2 = r2_score(
        y_test_e,
        east_prediction
    )


    print("")
    print("North MAE :", round(north_mae, 4))
    print("North RMSE:", round(north_rmse, 4))
    print("North R2  :", round(north_r2, 4))

    print("East MAE  :", round(east_mae, 4))
    print("East RMSE :", round(east_rmse, 4))
    print("East R2   :", round(east_r2, 4))


    results.append({
        "model": model_name,
        "feature_count": len(selected),
        "north_mae": north_mae,
        "north_rmse": north_rmse,
        "north_r2": north_r2,
        "east_mae": east_mae,
        "east_rmse": east_rmse,
        "east_r2": east_r2
    })


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SENSOR ABLATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n")
print("=" * 70)
print("V10.9 COMPLETE")
print("=" * 70)