import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

OUTPUT_DIR = Path("outputs")
PROCESSED_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

# GNSS outage used for prototype demonstration
OUTAGE_DURATION_SECONDS = 60

# We keep the outage inside the final 30% of the dataset.
TEST_START_RATIO = 0.70

# Minimum vehicle speed for selecting a meaningful driving outage.
MIN_MOVING_SPEED_KMH = 5.0


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 80)
    print("SIH PROTOTYPE V1")
    print("AI SPEED ESTIMATION + GNSS OUTAGE SIMULATION")
    print("=" * 80)

    print("\nLoading IO-VNBD dataset...")

    df = pd.read_csv(
        DATA_PATH,
        encoding="cp1252"
    )

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    date_column = (
        "DATE_YYYY-MO-DD_HH-MI-SS_SSS"
    )

    df["timestamp"] = pd.to_datetime(
        df[date_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    df["time_s"] = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    return df


# ============================================================
# SENSOR FEATURES
# ============================================================

def create_features(df):

    print("\nCreating IMU features...")

    features = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # Accelerometer
    # --------------------------------------------------------

    ax = df[
        "ACCELEROMETER_X_m_s²"
    ].astype(float)

    ay = df[
        "ACCELEROMETER_Y_m_s²"
    ].astype(float)

    az = df[
        "ACCELEROMETER_Z_m_s²"
    ].astype(float)

    # --------------------------------------------------------
    # Gyroscope
    # --------------------------------------------------------

    gyro_yaw = df[
        "GYROSCOPE_Yaw_rad_s"
    ].astype(float)

    gyro_pitch = df[
        "GYROSCOPE_Pitch_rad_s"
    ].astype(float)

    gyro_roll = df[
        "GYROSCOPE_Roll_rad_s"
    ].astype(float)

    # --------------------------------------------------------
    # Magnetometer
    # --------------------------------------------------------

    mx = df[
        "MAGNETIC_FIELD_X_Î¼T"
    ].astype(float)

    my = df[
        "MAGNETIC_FIELD_Y_Î¼T"
    ].astype(float)

    mz = df[
        "MAGNETIC_FIELD_Z_Î¼T"
    ].astype(float)

    # --------------------------------------------------------
    # Basic magnitudes
    # --------------------------------------------------------

    features["acc_x"] = ax
    features["acc_y"] = ay
    features["acc_z"] = az

    features["acc_mag"] = np.sqrt(
        ax**2 +
        ay**2 +
        az**2
    )

    features["gyro_yaw"] = gyro_yaw
    features["gyro_pitch"] = gyro_pitch
    features["gyro_roll"] = gyro_roll

    features["gyro_mag"] = np.sqrt(
        gyro_yaw**2 +
        gyro_pitch**2 +
        gyro_roll**2
    )

    features["mag_x"] = mx
    features["mag_y"] = my
    features["mag_z"] = mz

    features["mag_mag"] = np.sqrt(
        mx**2 +
        my**2 +
        mz**2
    )

    # --------------------------------------------------------
    # Gravity-free acceleration magnitude
    # --------------------------------------------------------

    gravity_mag = np.sqrt(
        df["GRAVITY_X_m_s²"].astype(float)**2 +
        df["GRAVITY_Y_m_s²"].astype(float)**2 +
        df["GRAVITY_Z_m_s²"].astype(float)**2
    )

    features["acc_gravity_difference"] = (
        features["acc_mag"] -
        gravity_mag
    )

    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    features["acc_mag_std"] = (
        features["acc_mag"]
        .rolling(10, min_periods=1)
        .std()
        .fillna(0)
    )

    features["gyro_mag_std"] = (
        features["gyro_mag"]
        .rolling(10, min_periods=1)
        .std()
        .fillna(0)
    )

    # --------------------------------------------------------
    # Clean invalid values
    # --------------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    features = features.interpolate(
        limit_direction="both"
    )

    features = features.fillna(0)

    return features


# ============================================================
# TARGET
# ============================================================

def get_target(df):

    speed = df[
        "GPS_SPEED_Kmh"
    ].astype(float)

    speed = speed.interpolate(
        limit_direction="both"
    )

    return speed.to_numpy()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(
    features,
    target,
    df
):

    split_index = int(
        len(df) *
        TEST_START_RATIO
    )

    X_train = features.iloc[
        :split_index
    ]

    y_train = target[
        :split_index
    ]

    X_test = features.iloc[
        split_index:
    ]

    y_test = target[
        split_index:
    ]

    print("\nDataset split:")

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    print(
        f"Training period  : "
        f"0 - "
        f"{df['time_s'].iloc[split_index]:.1f}s"
    )

    print(
        f"Testing period   : "
        f"{df['time_s'].iloc[split_index]:.1f}s - "
        f"{df['time_s'].iloc[-1]:.1f}s"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        split_index
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\nTraining AI speed model...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "AI model training complete."
    )

    return model


# ============================================================
# SELECT OUTAGE
# ============================================================

def select_outage(
    df,
    split_index
):

    print("\nSelecting GNSS outage...")

    test_start = split_index

    duration = (
        df["time_s"].iloc[-1]
        -
        df["time_s"].iloc[split_index]
    )

    # Search for a moving section
    # inside the test region.

    window = OUTAGE_DURATION_SECONDS

    step = 10

    candidate = None

    for start_time in np.arange(
        df["time_s"].iloc[test_start],
        df["time_s"].iloc[-1] - window,
        step
    ):

        end_time = (
            start_time +
            window
        )

        mask = (
            (df["time_s"] >= start_time)
            &
            (df["time_s"] <= end_time)
        )

        speeds = df.loc[
            mask,
            "GPS_SPEED_Kmh"
        ].astype(float)

        moving_ratio = np.mean(
            speeds > MIN_MOVING_SPEED_KMH
        )

        if moving_ratio > 0.70:

            candidate = (
                start_time,
                end_time
            )

            break

    # Fallback
    if candidate is None:

        candidate = (
            df["time_s"].iloc[test_start] + 10,
            df["time_s"].iloc[test_start]
            + 10
            + window
        )

    start_time, end_time = candidate

    print(
        f"GNSS outage:"
    )

    print(
        f"  Start : {start_time:.2f}s"
    )

    print(
        f"  End   : {end_time:.2f}s"
    )

    print(
        f"  Duration : "
        f"{end_time - start_time:.2f}s"
    )

    return start_time, end_time


# ============================================================
# EVALUATE AI MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\nEvaluating AI speed model...")

    prediction = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    print(
        f"\nAI Speed MAE  : {mae:.4f} km/h"
    )

    print(
        f"AI Speed RMSE : {rmse:.4f} km/h"
    )

    return prediction


# ============================================================
# CREATE OUTAGE PREDICTION
# ============================================================

def create_outage_prediction(
    df,
    features,
    model,
    start_time,
    end_time
):

    print("\nRunning GNSS-denied inference...")

    predicted_speed = model.predict(
        features
    )

    actual_speed = (
        df["GPS_SPEED_Kmh"]
        .astype(float)
        .to_numpy()
    )

    outage_mask = (
        (df["time_s"] >= start_time)
        &
        (df["time_s"] <= end_time)
    )

    # During the simulated outage,
    # GNSS speed is considered unavailable.
    simulated_speed = actual_speed.copy()

    simulated_speed[outage_mask] = (
        predicted_speed[outage_mask]
    )

    return (
        predicted_speed,
        simulated_speed,
        outage_mask
    )


# ============================================================
# PLOT AI SPEED
# ============================================================

def plot_speed(
    df,
    actual_speed,
    predicted_speed,
    start_time,
    end_time
):

    print("\nGenerating speed validation plot...")

    time = df[
        "time_s"
    ].to_numpy()

    plt.figure(
        figsize=(16, 7)
    )

    plt.plot(
        time,
        actual_speed,
        label="GNSS reference speed",
        linewidth=1
    )

    plt.plot(
        time,
        predicted_speed,
        label="AI predicted speed",
        linewidth=1
    )

    plt.axvspan(
        start_time,
        end_time,
        alpha=0.25,
        label="Simulated GNSS outage"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Speed (km/h)"
    )

    plt.title(
        "AI Vehicle Speed Estimation "
        "with Simulated GNSS Outage"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUTPUT_DIR /
        "prototype_ai_speed.png"
    )

    plt.savefig(
        path,
        dpi=180
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    df,
    predicted_speed,
    simulated_speed,
    outage_mask
):

    output = pd.DataFrame()

    output["timestamp"] = (
        df["timestamp"]
    )

    output["time_s"] = (
        df["time_s"]
    )

    output["gps_speed_kmh"] = (
        df["GPS_SPEED_Kmh"]
    )

    output["ai_speed_kmh"] = (
        predicted_speed
    )

    output["navigation_speed_kmh"] = (
        simulated_speed
    )

    output["gnss_available"] = (
        ~outage_mask
    )

    path = (
        PROCESSED_DIR /
        "prototype_ai_speed_results.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print(
        f"Saved results: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    features = create_features(
        df
    )

    target = get_target(
        df
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        split_index
    ) = split_data(
        features,
        target,
        df
    )

    model = train_model(
        X_train,
        y_train
    )

    test_prediction = evaluate_model(
        model,
        X_test,
        y_test
    )

    start_time, end_time = (
        select_outage(
            df,
            split_index
        )
    )

    (
        predicted_speed,
        simulated_speed,
        outage_mask
    ) = create_outage_prediction(
        df,
        features,
        model,
        start_time,
        end_time
    )

    actual_speed = (
        df["GPS_SPEED_Kmh"]
        .astype(float)
        .to_numpy()
    )

    plot_speed(
        df,
        actual_speed,
        predicted_speed,
        start_time,
        end_time
    )

    save_results(
        df,
        predicted_speed,
        simulated_speed,
        outage_mask
    )

    print("\n" + "=" * 80)
    print("PROTOTYPE V1 COMPLETE")
    print("=" * 80)

    print(
        "\nAI model : Random Forest"
    )

    print(
        "Inputs  : Accelerometer + "
        "Gyroscope + Magnetometer"
    )

    print(
        "Target  : GNSS vehicle speed"
    )

    print(
        "Mode    : AI speed estimation "
        "during simulated GNSS outage"
    )

    print(
        "\nNext:"
    )

    print(
        "AI speed -> heading -> "
        "dead reckoning trajectory -> "
        "EKF fusion"
    )


if __name__ == "__main__":
    main()