import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# V10.20
# RF Ensemble Uncertainty Validation
# ============================================================

RAW_FILE = (
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/M (Driver B)/S-M.csv"
)

TARGET_FILE = "data/processed/prototype_v9_velocity_targets.csv"

OUTPUT_FILE = (
    "data/processed/prototype_v10_20_uncertainty_validation.csv"
)

RANDOM_SEEDS = [17, 42, 91]

TRAIN_RATIO = 0.70

WINDOWS = [5, 10, 20]


print("=" * 80)
print("V10.20 RF ENSEMBLE UNCERTAINTY VALIDATION")
print("=" * 80)


# ============================================================
# Helper functions
# ============================================================

def find_column(columns, keywords):
    """
    Find a column using semantic keyword matching.
    """
    for col in columns:
        name = col.lower()
        if all(k.lower() in name for k in keywords):
            return col
    return None


def build_features(df):
    """
    Build temporal GYRO + MAG features.

    We deliberately keep the feature family close to the
    successful V10.x RF experiments.
    """

    feature_data = pd.DataFrame(index=df.index)

    gyro_cols = [
        "GYROSCOPE Yaw (rad/s)",
        "GYROSCOPE Pitch (rad/s)",
        "GYROSCOPE Roll (rad/s)"
    ]

    mag_x = find_column(df.columns, ["MAGNETIC", "FIELD", "X"])
    mag_y = find_column(df.columns, ["MAGNETIC", "FIELD", "Y"])
    mag_z = find_column(df.columns, ["MAGNETIC", "FIELD", "Z"])

    print("\nMagnetometer:")
    print("X:", mag_x)
    print("Y:", mag_y)
    print("Z:", mag_z)

    sensor_cols = gyro_cols + [mag_x, mag_y, mag_z]

    for col in sensor_cols:
        feature_data[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Raw sensor values
    # --------------------------------------------------------

    for col in sensor_cols:
        safe_name = (
            col.replace(" ", "_")
               .replace("/", "_")
               .replace("(", "")
               .replace(")", "")
               .replace("²", "2")
               .replace("μ", "u")
               .replace("Î", "I")
               .replace("°", "deg")
        )

        feature_data[f"raw_{safe_name}"] = feature_data[col]

    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    for col in sensor_cols:

        safe_name = (
            col.replace(" ", "_")
               .replace("/", "_")
               .replace("(", "")
               .replace(")", "")
               .replace("²", "2")
               .replace("μ", "u")
               .replace("Î", "I")
               .replace("°", "deg")
        )

        series = feature_data[col]

        for window in WINDOWS:

            feature_data[
                f"{safe_name}_mean_{window}"
            ] = series.rolling(
                window,
                min_periods=1
            ).mean()

            feature_data[
                f"{safe_name}_std_{window}"
            ] = series.rolling(
                window,
                min_periods=1
            ).std().fillna(0)

    # --------------------------------------------------------
    # Gyroscope magnitude
    # --------------------------------------------------------

    gx = feature_data[gyro_cols[0]]
    gy = feature_data[gyro_cols[1]]
    gz = feature_data[gyro_cols[2]]

    feature_data["gyro_magnitude"] = np.sqrt(
        gx ** 2 +
        gy ** 2 +
        gz ** 2
    )

    # --------------------------------------------------------
    # Magnetometer magnitude
    # --------------------------------------------------------

    mx = feature_data[mag_x]
    my = feature_data[mag_y]
    mz = feature_data[mag_z]

    feature_data["mag_magnitude"] = np.sqrt(
        mx ** 2 +
        my ** 2 +
        mz ** 2
    )

    for window in WINDOWS:

        feature_data[
            f"gyro_magnitude_mean_{window}"
        ] = feature_data[
            "gyro_magnitude"
        ].rolling(
            window,
            min_periods=1
        ).mean()

        feature_data[
            f"mag_magnitude_mean_{window}"
        ] = feature_data[
            "mag_magnitude"
        ].rolling(
            window,
            min_periods=1
        ).mean()

    feature_data = feature_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    feature_data = feature_data.ffill().bfill().fillna(0)

    # Keep only generated numeric features.
    return feature_data


# ============================================================
# 1. Load raw dataset
# ============================================================

print("\n[1] Loading raw dataset...")

df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

df.columns = df.columns.str.strip()

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. Load velocity targets
# ============================================================

print("\n[2] Loading velocity targets...")

targets = pd.read_csv(TARGET_FILE)

print("Target rows:", len(targets))


# ============================================================
# 3. Synchronize
# ============================================================

print("\n[3] Synchronizing data...")

if len(df) != len(targets):
    raise ValueError(
        f"Row mismatch: raw={len(df)}, targets={len(targets)}"
    )

data = df.copy()

data["timestamp"] = pd.to_datetime(
    targets["timestamp"]
)

data["vn_true"] = targets["vn_mps"].astype(float)
data["ve_true"] = targets["ve_mps"].astype(float)

data = data.reset_index(drop=True)

print("Synchronized rows:", len(data))


# ============================================================
# 4. Build features
# ============================================================

print("\n[4] Building features...")

X = build_features(data)

y_north = data["vn_true"].to_numpy()
y_east = data["ve_true"].to_numpy()

X_values = X.to_numpy(dtype=np.float32)

n = len(data)

train_end = int(n * TRAIN_RATIO)

print("Feature count:", X_values.shape[1])
print("Train rows:", train_end)
print("Test rows:", n - train_end)

print(
    "Training:",
    data["timestamp"].iloc[0],
    "to",
    data["timestamp"].iloc[train_end - 1]
)

print(
    "Testing:",
    data["timestamp"].iloc[train_end],
    "to",
    data["timestamp"].iloc[-1]
)


# ============================================================
# 5. Train multiple RF models
# ============================================================

north_predictions = []
east_predictions = []

print("\n[5] Training RF ensemble...")

for seed in RANDOM_SEEDS:

    print("\nSeed:", seed)

    print("  Training North...")

    model_n = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1
    )

    model_n.fit(
        X_values[:train_end],
        y_north[:train_end]
    )

    print("  Training East...")

    model_e = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1
    )

    model_e.fit(
        X_values[:train_end],
        y_east[:train_end]
    )

    print("  Predicting test region...")

    pred_n = model_n.predict(
        X_values[train_end:]
    )

    pred_e = model_e.predict(
        X_values[train_end:]
    )

    north_predictions.append(pred_n)
    east_predictions.append(pred_e)


# ============================================================
# 6. Calculate ensemble mean + uncertainty
# ============================================================

print("\n[6] Calculating prediction mean and uncertainty...")

north_predictions = np.vstack(north_predictions)
east_predictions = np.vstack(east_predictions)

vn_ai = north_predictions.mean(axis=0)
ve_ai = east_predictions.mean(axis=0)

vn_std = north_predictions.std(
    axis=0,
    ddof=1
)

ve_std = east_predictions.std(
    axis=0,
    ddof=1
)

vn_true_test = y_north[train_end:]
ve_true_test = y_east[train_end:]


# ============================================================
# 7. Metrics
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


north_mae = mean_absolute_error(
    vn_true_test,
    vn_ai
)

north_rmse = rmse(
    vn_true_test,
    vn_ai
)

north_r2 = r2_score(
    vn_true_test,
    vn_ai
)

east_mae = mean_absolute_error(
    ve_true_test,
    ve_ai
)

east_rmse = rmse(
    ve_true_test,
    ve_ai
)

east_r2 = r2_score(
    ve_true_test,
    ve_ai
)

true_speed = np.sqrt(
    vn_true_test ** 2 +
    ve_true_test ** 2
)

ai_speed = np.sqrt(
    vn_ai ** 2 +
    ve_ai ** 2
)

speed_mae = mean_absolute_error(
    true_speed,
    ai_speed
)


# ============================================================
# 8. Combined uncertainty
# ============================================================

uncertainty = np.sqrt(
    vn_std ** 2 +
    ve_std ** 2
)

velocity_error = np.sqrt(
    (vn_ai - vn_true_test) ** 2 +
    (ve_ai - ve_true_test) ** 2
)


# ============================================================
# 9. Correlation between uncertainty and actual error
# ============================================================

if (
    np.std(uncertainty) > 0
    and np.std(velocity_error) > 0
):

    uncertainty_error_corr = np.corrcoef(
        uncertainty,
        velocity_error
    )[0, 1]

else:

    uncertainty_error_corr = np.nan


# ============================================================
# 10. Low/high uncertainty analysis
# ============================================================

low_threshold = np.quantile(
    uncertainty,
    0.25
)

high_threshold = np.quantile(
    uncertainty,
    0.75
)

low_mask = uncertainty <= low_threshold
high_mask = uncertainty >= high_threshold

low_error = velocity_error[low_mask].mean()
high_error = velocity_error[high_mask].mean()

low_uncertainty_mean = uncertainty[low_mask].mean()
high_uncertainty_mean = uncertainty[high_mask].mean()


# ============================================================
# 11. Print results
# ============================================================

print("\n")
print("=" * 80)
print("V10.20 TEST RESULTS")
print("=" * 80)

print(
    f"North MAE : {north_mae:.4f} m/s"
)

print(
    f"North RMSE: {north_rmse:.4f} m/s"
)

print(
    f"North R²  : {north_r2:.4f}"
)

print()

print(
    f"East MAE  : {east_mae:.4f} m/s"
)

print(
    f"East RMSE : {east_rmse:.4f} m/s"
)

print(
    f"East R²   : {east_r2:.4f}"
)

print()

print(
    f"Speed MAE : {speed_mae:.4f} m/s "
    f"({speed_mae * 3.6:.4f} km/h)"
)

print("\n")
print("UNCERTAINTY")
print("-" * 80)

print(
    f"Mean uncertainty : "
    f"{uncertainty.mean():.4f} m/s"
)

print(
    f"Median uncertainty: "
    f"{np.median(uncertainty):.4f} m/s"
)

print(
    f"P95 uncertainty  : "
    f"{np.percentile(uncertainty, 95):.4f} m/s"
)

print()

print(
    f"Uncertainty/error correlation: "
    f"{uncertainty_error_corr:.4f}"
)

print()

print(
    f"Low uncertainty threshold : "
    f"{low_threshold:.4f} m/s"
)

print(
    f"Low uncertainty mean error: "
    f"{low_error:.4f} m/s"
)

print()

print(
    f"High uncertainty threshold : "
    f"{high_threshold:.4f} m/s"
)

print(
    f"High uncertainty mean error: "
    f"{high_error:.4f} m/s"
)


# ============================================================
# 12. Save detailed results
# ============================================================

print("\n[7] Saving results...")

test_timestamps = data["timestamp"].iloc[
    train_end:
].reset_index(drop=True)

output = pd.DataFrame({
    "timestamp": test_timestamps,

    "vn_true_mps": vn_true_test,
    "ve_true_mps": ve_true_test,

    "vn_ai_mps": vn_ai,
    "ve_ai_mps": ve_ai,

    "vn_std_mps": vn_std,
    "ve_std_mps": ve_std,

    "combined_uncertainty_mps": uncertainty,

    "velocity_error_mps": velocity_error,

    "true_speed_mps": true_speed,
    "ai_speed_mps": ai_speed
})

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "Saved:",
    os.path.abspath(OUTPUT_FILE)
)


# ============================================================
# 13. Uncertainty bins
# ============================================================

print("\n")
print("=" * 80)
print("UNCERTAINTY BIN ANALYSIS")
print("=" * 80)

output["uncertainty_bin"] = pd.qcut(
    output["combined_uncertainty_mps"],
    q=5,
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ],
    duplicates="drop"
)

bin_summary = (
    output
    .groupby(
        "uncertainty_bin",
        observed=True
    )
    .agg(
        samples=("velocity_error_mps", "size"),
        mean_uncertainty=(
            "combined_uncertainty_mps",
            "mean"
        ),
        mean_velocity_error=(
            "velocity_error_mps",
            "mean"
        ),
        median_velocity_error=(
            "velocity_error_mps",
            "median"
        )
    )
    .reset_index()
)

print(bin_summary.to_string(index=False))


# ============================================================
# 14. Final
# ============================================================

print("\n")
print("=" * 80)
print("V10.20 COMPLETE")
print("=" * 80)

print(
    "The experiment measures whether RF ensemble spread "
    "is actually informative about prediction error."
)

print(
    "Do not use the uncertainty as a navigation weight yet."
)

print(
    "First inspect the correlation and uncertainty bins."
)