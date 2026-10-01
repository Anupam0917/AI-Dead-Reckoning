import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


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
    "prototype_v10_10_outage_ablation.csv"
)


# ============================================================
# SETTINGS
# ============================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

TRAIN_RATIO = 0.70


print("=" * 75)
print("V10.10 GNSS OUTAGE SENSOR ABLATION")
print("=" * 75)


# ============================================================
# 1. LOAD RAW
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

print(
    "Rows:",
    len(raw)
)


# ============================================================
# 2. MAGNETOMETER
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


# ============================================================
# 3. SENSOR DATA
# ============================================================

print("[2] Preparing sensors...")


def num(column):

    return pd.to_numeric(
        raw[column],
        errors="coerce"
    )


features = pd.DataFrame({

    "timestamp":
        raw["timestamp"],

    "acc_x":
        num(
            "ACCELEROMETER X (m/s²)"
        ),

    "acc_y":
        num(
            "ACCELEROMETER Y (m/s²)"
        ),

    "acc_z":
        num(
            "ACCELEROMETER Z (m/s²)"
        ),

    "gravity_x":
        num(
            "GRAVITY X (m/s²)"
        ),

    "gravity_y":
        num(
            "GRAVITY Y (m/s²)"
        ),

    "gravity_z":
        num(
            "GRAVITY Z (m/s²)"
        ),

    "gyro_yaw":
        num(
            "GYROSCOPE Yaw (rad/s)"
        ),

    "gyro_pitch":
        num(
            "GYROSCOPE Pitch (rad/s)"
        ),

    "gyro_roll":
        num(
            "GYROSCOPE Roll (rad/s)"
        ),

    "mag_x":
        num(mag_x),

    "mag_y":
        num(mag_y),

    "mag_z":
        num(mag_z)
})


# ============================================================
# 4. DERIVED
# ============================================================

print("[3] Creating derived features...")


features["acc_magnitude"] = np.sqrt(
    features["acc_x"] ** 2 +
    features["acc_y"] ** 2 +
    features["acc_z"] ** 2
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

features["gravity_magnitude"] = np.sqrt(
    features["gravity_x"] ** 2 +
    features["gravity_y"] ** 2 +
    features["gravity_z"] ** 2
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


parts = []


for window in [5, 10, 20]:

    temp = {}

    for column in rolling_columns:

        temp[
            column + "_mean_" + str(window)
        ] = (
            features[column]
            .rolling(
                window,
                min_periods=1
            )
            .mean()
        )

        temp[
            column + "_std_" + str(window)
        ] = (
            features[column]
            .rolling(
                window,
                min_periods=1
            )
            .std()
            .fillna(0)
        )

    parts.append(
        pd.DataFrame(
            temp,
            index=features.index
        )
    )


features = pd.concat(
    [
        features,
        parts[0],
        parts[1],
        parts[2]
    ],
    axis=1
)


# ============================================================
# 6. TARGETS
# ============================================================

print("[5] Loading velocity targets...")

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

print("[6] Matching sensors with targets...")

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

    tolerance=pd.Timedelta(
        "50ms"
    )
)


# ============================================================
# 8. CLEAN
# ============================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

sensor_columns = [
    column
    for column in features.columns
    if column != "timestamp"
]

data = data.dropna(
    subset=[
        "vn_mps",
        "ve_mps"
    ]
)

data = data.dropna(
    subset=sensor_columns
)

print(
    "Usable rows:",
    len(data)
)


# ============================================================
# 9. TRAIN / TEST
# ============================================================

split = int(
    len(data) * TRAIN_RATIO
)

train = data.iloc[
    :split
]

print(
    "Training rows:",
    len(train)
)


# ============================================================
# 10. MODEL GROUPS
# ============================================================

groups = {

    "ALL": [
        "acc_",
        "linear_acc_",
        "gravity_",
        "gyro_",
        "mag_"
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
# 11. TRAIN MODELS
# ============================================================

models = {}


for name in groups:

    print("\n" + "=" * 75)
    print(
        "Training:",
        name
    )
    print("=" * 75)


    prefixes = groups[name]

    selected = []

    for column in sensor_columns:

        for prefix in prefixes:

            if column.startswith(prefix):

                selected.append(column)
                break


    print(
        "Features:",
        len(selected)
    )


    north_model = RandomForestRegressor(

        n_estimators=100,

        max_depth=20,

        min_samples_leaf=3,

        random_state=42,

        n_jobs=-1

    )


    east_model = RandomForestRegressor(

        n_estimators=100,

        max_depth=20,

        min_samples_leaf=3,

        random_state=42,

        n_jobs=-1

    )


    north_model.fit(
        train[selected],
        train["vn_mps"]
    )

    east_model.fit(
        train[selected],
        train["ve_mps"]
    )


    models[name] = (
        selected,
        north_model,
        east_model
    )


# ============================================================
# 12. OUTAGE DATA
# ============================================================

outage = data[
    (data["time_s"] >= OUTAGE_START) &
    (data["time_s"] <= OUTAGE_END)
].copy()

outage = outage.reset_index(
    drop=True
)


print("\n")
print("=" * 75)
print("GNSS OUTAGE")
print("=" * 75)

print(
    "Start:",
    OUTAGE_START,
    "seconds"
)

print(
    "End:",
    OUTAGE_END,
    "seconds"
)

print(
    "Samples:",
    len(outage)
)


# ============================================================
# 13. REFERENCE TRAJECTORY
# ============================================================

dt = np.diff(
    outage["time_s"].values
)

dt = np.clip(
    dt,
    0.001,
    0.5
)


true_vn = outage[
    "vn_mps"
].values

true_ve = outage[
    "ve_mps"
].values


true_n = np.zeros(
    len(outage)
)

true_e = np.zeros(
    len(outage)
)


for i in range(
    1,
    len(outage)
):

    true_n[i] = (
        true_n[i - 1]
        +
        true_vn[i - 1] *
        dt[i - 1]
    )

    true_e[i] = (
        true_e[i - 1]
        +
        true_ve[i - 1] *
        dt[i - 1]
    )


# ============================================================
# 14. EVALUATE MODELS
# ============================================================

results = []


for name in models:

    print("\n")
    print("=" * 75)
    print(
        "Evaluating:",
        name
    )
    print("=" * 75)


    selected, north_model, east_model = (
        models[name]
    )


    pred_vn = north_model.predict(
        outage[selected]
    )

    pred_ve = east_model.predict(
        outage[selected]
    )


    # --------------------------------------------------------
    # VELOCITY ERROR
    # --------------------------------------------------------

    north_mae = mean_absolute_error(
        true_vn,
        pred_vn
    )

    east_mae = mean_absolute_error(
        true_ve,
        pred_ve
    )


    # --------------------------------------------------------
    # INTEGRATE AI VELOCITY
    # --------------------------------------------------------

    nav_n = np.zeros(
        len(outage)
    )

    nav_e = np.zeros(
        len(outage)
    )


    for i in range(
        1,
        len(outage)
    ):

        nav_n[i] = (
            nav_n[i - 1]
            +
            pred_vn[i - 1] *
            dt[i - 1]
        )

        nav_e[i] = (
            nav_e[i - 1]
            +
            pred_ve[i - 1] *
            dt[i - 1]
        )


    # --------------------------------------------------------
    # POSITION ERROR
    # --------------------------------------------------------

    position_error = np.sqrt(
        (nav_n - true_n) ** 2 +
        (nav_e - true_e) ** 2
    )


    final_error = position_error[-1]

    mean_error = np.mean(
        position_error
    )

    max_error = np.max(
        position_error
    )


    # --------------------------------------------------------
    # MODEL PATH
    # --------------------------------------------------------

    model_distance = np.sum(
        np.sqrt(
            np.diff(nav_n) ** 2 +
            np.diff(nav_e) ** 2
        )
    )


    true_distance = np.sum(
        np.sqrt(
            np.diff(true_n) ** 2 +
            np.diff(true_e) ** 2
        )
    )


    if true_distance > 0:

        drift_percent = (
            abs(
                model_distance -
                true_distance
            )
            /
            true_distance
            *
            100
        )

    else:

        drift_percent = np.nan


    print(
        "North velocity MAE:",
        round(north_mae, 4)
    )

    print(
        "East velocity MAE:",
        round(east_mae, 4)
    )

    print(
        "Mean position error:",
        round(mean_error, 3),
        "m"
    )

    print(
        "Final position error:",
        round(final_error, 3),
        "m"
    )

    print(
        "Maximum position error:",
        round(max_error, 3),
        "m"
    )

    print(
        "Reference distance:",
        round(true_distance, 3),
        "m"
    )

    print(
        "Model distance:",
        round(model_distance, 3),
        "m"
    )

    print(
        "Drift:",
        round(drift_percent, 3),
        "%"
    )


    results.append({

        "model":
            name,

        "outage_start_s":
            OUTAGE_START,

        "outage_duration_s":
            OUTAGE_END - OUTAGE_START,

        "samples":
            len(outage),

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

        "model_distance_m":
            model_distance,

        "drift_percent":
            drift_percent

    })


# ============================================================
# 15. SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 16. FINAL
# ============================================================

print("\n")
print("=" * 75)
print("FINAL V10.10 OUTAGE RESULTS")
print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(
    OUTPUT_FILE
)

print("\n")
print("=" * 75)
print("V10.10 COMPLETE")
print("=" * 75)