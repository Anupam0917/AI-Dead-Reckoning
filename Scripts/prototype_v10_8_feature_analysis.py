import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PATHS
# =========================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")

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
    DATA,
    "prototype_v9_velocity_targets.csv"
)

OUTPUT_FILE = os.path.join(
    DATA,
    "prototype_v10_8_feature_importance.csv"
)

GROUP_FILE = os.path.join(
    DATA,
    "prototype_v10_8_group_comparison.csv"
)

PLOT_FILE = os.path.join(
    DATA,
    "prototype_v10_8_feature_importance.png"
)


print("=" * 75)
print("V10.8 FEATURE ANALYSIS")
print("=" * 75)


# =========================================================
# 1. LOAD RAW SENSOR DATA
# =========================================================

print("\n[1] Loading raw sensor data...")

raw = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

raw.columns = (
    raw.columns
    .str.strip()
)

print(
    "Raw rows:",
    len(raw)
)

print(
    "Raw columns:",
    len(raw.columns)
)


# =========================================================
# 2. TIMESTAMP
# =========================================================

raw["timestamp"] = pd.to_datetime(
    raw["DATE (YYYY-MO-DD HH-MI-SS_SSS)"],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)


# =========================================================
# 3. FIND MAGNETOMETER COLUMNS
# =========================================================

mag_columns = {}

for col in raw.columns:

    upper = col.upper()

    if "MAGNETIC FIELD" in upper:

        if " X " in upper:
            mag_columns["mag_x"] = col

        elif " Y " in upper:
            mag_columns["mag_y"] = col

        elif " Z " in upper:
            mag_columns["mag_z"] = col


print("\nMagnetometer columns:")

for k, v in mag_columns.items():
    print(
        f"  {k}: {v}"
    )


# =========================================================
# 4. SENSOR COLUMN MAP
# =========================================================

sensor_columns = {

    "acc_x":
        "ACCELEROMETER X (m/s²)",

    "acc_y":
        "ACCELEROMETER Y (m/s²)",

    "acc_z":
        "ACCELEROMETER Z (m/s²)",

    "gravity_x":
        "GRAVITY X (m/s²)",

    "gravity_y":
        "GRAVITY Y (m/s²)",

    "gravity_z":
        "GRAVITY Z (m/s²)",

    "gyro_yaw":
        "GYROSCOPE Yaw (rad/s)",

    "gyro_pitch":
        "GYROSCOPE Pitch (rad/s)",

    "gyro_roll":
        "GYROSCOPE Roll (rad/s)"
}


sensor_columns.update(
    mag_columns
)


# =========================================================
# 5. CHECK COLUMNS
# =========================================================

print("\nChecking sensor columns...")

for name, col in sensor_columns.items():

    if col in raw.columns:
        print(
            f"  OK: {name}"
        )
    else:
        print(
            f"  MISSING: {name} -> {col}"
        )


# =========================================================
# 6. NUMERIC CONVERSION
# =========================================================

for col in sensor_columns.values():

    if col in raw.columns:

        raw[col] = pd.to_numeric(
            raw[col],
            errors="coerce"
        )


# =========================================================
# 7. CREATE SENSOR FEATURES
# =========================================================

print("\n[2] Creating sensor features...")

features = pd.DataFrame()

features["timestamp"] = raw["timestamp"]

features["acc_x"] = raw[
    sensor_columns["acc_x"]
]

features["acc_y"] = raw[
    sensor_columns["acc_y"]
]

features["acc_z"] = raw[
    sensor_columns["acc_z"]
]

features["gravity_x"] = raw[
    sensor_columns["gravity_x"]
]

features["gravity_y"] = raw[
    sensor_columns["gravity_y"]
]

features["gravity_z"] = raw[
    sensor_columns["gravity_z"]
]

features["gyro_yaw"] = raw[
    sensor_columns["gyro_yaw"]
]

features["gyro_pitch"] = raw[
    sensor_columns["gyro_pitch"]
]

features["gyro_roll"] = raw[
    sensor_columns["gyro_roll"]
]


if "mag_x" in mag_columns:

    features["mag_x"] = raw[
        mag_columns["mag_x"]
    ]

    features["mag_y"] = raw[
        mag_columns["mag_y"]
    ]

    features["mag_z"] = raw[
        mag_columns["mag_z"]
    ]


# =========================================================
# 8. DERIVED FEATURES
# =========================================================

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


if (
    "mag_x" in features.columns and
    "mag_y" in features.columns and
    "mag_z" in features.columns
):

    features["mag_magnitude"] = np.sqrt(
        features["mag_x"] ** 2 +
        features["mag_y"] ** 2 +
        features["mag_z"] ** 2
    )


# Linear acceleration

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


# =========================================================
# 9. ROLLING FEATURES
# =========================================================

print("[3] Creating temporal features...")

ROLLING_WINDOWS = [5, 10, 20]

base_signal_columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "linear_acc_x",
    "linear_acc_y",
    "linear_acc_z",
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll"
]

if "mag_x" in features.columns:

    base_signal_columns.extend([
        "mag_x",
        "mag_y",
        "mag_z"
    ])


for window in ROLLING_WINDOWS:

    for col in base_signal_columns:

        features[
            f"{col}_mean_{window}"
        ] = (
            features[col]
            .rolling(window, min_periods=1)
            .mean()
        )

        features[
            f"{col}_std_{window}"
        ] = (
            features[col]
            .rolling(window, min_periods=1)
            .std()
            .fillna(0)
        )


# =========================================================
# 10. LOAD TARGET
# =========================================================

print("\n[4] Loading velocity targets...")

target = pd.read_csv(
    TARGET_FILE
)

target["timestamp"] = pd.to_datetime(
    target["timestamp"]
)

target = target.sort_values(
    "timestamp"
).reset_index(drop=True)


# =========================================================
# 11. MERGE
# =========================================================

print("\n[5] Matching sensor features with targets...")

data = pd.merge_asof(
    target[
        [
            "timestamp",
            "time_s",
            "vn_mps",
            "ve_mps"
        ]
    ].sort_values("timestamp"),

    features.sort_values("timestamp"),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta("50ms")
)


print(
    "Merged rows:",
    len(data)
)


# =========================================================
# 12. SELECT MODEL DATA
# =========================================================

exclude = [
    "timestamp",
    "time_s",
    "vn_mps",
    "ve_mps"
]

feature_names = [
    c for c in data.columns
    if c not in exclude
]


data = data.dropna(
    subset=["vn_mps", "ve_mps"]
)

data[feature_names] = data[
    feature_names
].replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna(
    subset=feature_names
)

print(
    "Usable rows:",
    len(data)
)

print(
    "Features:",
    len(feature_names)
)


# =========================================================
# 13. CHRONOLOGICAL TRAIN / TEST
# =========================================================

split = int(
    len(data) * 0.70
)

train = data.iloc[:split]
test = data.iloc[split:]

X_train = train[
    feature_names
]

X_test = test[
    feature_names
]

y_train_n = train["vn_mps"]
y_test_n = test["vn_mps"]

y_train_e = train["ve_mps"]
y_test_e = test["ve_mps"]


# =========================================================
# 14. RANDOM FOREST
# =========================================================

print("\n[6] Training Random Forest...")

model_n = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model_e = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)


model_n.fit(
    X_train,
    y_train_n
)

model_e.fit(
    X_train,
    y_train_e
)


# =========================================================
# 15. PREDICTIONS
# =========================================================

pred_n = model_n.predict(
    X_test
)

pred_e = model_e.predict(
    X_test
)


# =========================================================
# 16. MODEL METRICS
# =========================================================

print("\n" + "=" * 75)
print("MODEL PERFORMANCE")
print("=" * 75)


print("\nNORTH")

print(
    "MAE:",
    f"{mean_absolute_error(y_test_n, pred_n):.4f}"
)

print(
    "RMSE:",
    f"{np.sqrt(mean_squared_error(y_test_n, pred_n)):.4f}"
)

print(
    "R2:",
    f"{r2_score(y_test_n, pred_n):.4f}"
)


print("\nEAST")

print(
    "MAE:",
    f"{mean_absolute_error(y_test_e, pred_e):.4f}"
)

print(
    "RMSE:",
    f"{np.sqrt(mean_squared_error(y_test_e, pred_e)):.4f}"
)

print(
    "R2:",
    f"{r2_score(y_test_e, pred_e):.4f}"
)


# =========================================================
# 17. FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({

    "feature": feature_names,

    "north_importance":
        model_n.feature_importances_,

    "east_importance":
        model_e.feature_importances_

})


importance["mean_importance"] = (
    importance["north_importance"] +
    importance["east_importance"]
) / 2


importance = importance.sort_values(
    "mean_importance",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 75)
print("TOP 30 FEATURES")
print("=" * 75)

print(
    importance.head(30).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =========================================================
# 18. SENSOR GROUPS
# =========================================================

def get_group(feature):

    if feature.startswith("acc_"):
        return "Accelerometer"

    if feature.startswith("linear_acc"):
        return "Linear Acceleration"

    if feature.startswith("gravity"):
        return "Gravity"

    if feature.startswith("gyro"):
        return "Gyroscope"

    if feature.startswith("mag"):
        return "Magnetometer"

    return "Other"


importance["group"] = (
    importance["feature"]
    .apply(get_group)
)


group_summary = (
    importance
    .groupby("group")
    .agg(
        feature_count=("feature", "count"),
        north_total=("north_importance", "sum"),
        east_total=("east_importance", "sum"),
        mean_importance=("mean_importance", "mean")
    )
    .sort_values(
        "east_total",
        ascending=False
    )
)


print("\n" + "=" * 75)
print("SENSOR GROUP IMPORTANCE")
print("=" * 75)

print(
    group_summary.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


# =========================================================
# 19. EAST VS NORTH IMPORTANCE
# =========================================================

importance["east_minus_north"] = (
    importance["east_importance"] -
    importance["north_importance"]
)

east_specific = importance.sort_values(
    "east_minus_north",
    ascending=False
)


print("\n" + "=" * 75)
print("FEATURES MORE IMPORTANT FOR EAST THAN NORTH")
print("=" * 75)

print(
    east_specific[
        [
            "feature",
            "north_importance",
            "east_importance",
            "east_minus_north"
        ]
    ].head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =========================================================
# 20. SAVE
# =========================================================

importance.to_csv(
    OUTPUT_FILE,
    index=False
)

group_summary.to_csv(
    GROUP_FILE
)

print("\nSaved:")
print(OUTPUT_FILE)

print(GROUP_FILE)


# =========================================================
# 21. FEATURE IMPORTANCE PLOT
# =========================================================

top = importance.head(20).iloc[::-1]

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top["feature"],
    top["east_importance"]
)

plt.xlabel(
    "Random Forest importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "V10.8 Top Features for East Velocity"
)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=200
)

plt.close()


print(PLOT_FILE)


# =========================================================
# 22. COMPLETE
# =========================================================

print("\n" + "=" * 75)
print("V10.8 COMPLETE")
print("=" * 75)