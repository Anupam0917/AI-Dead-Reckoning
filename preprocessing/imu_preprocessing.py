import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

GYRO_OUTLIER_THRESHOLD = 0.433104
LARGE_GAP_THRESHOLD = 0.2


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path=DATA_PATH):

    print("=" * 60)
    print("IMU PREPROCESSING")
    print("=" * 60)

    print("\nLoading dataset...")
    print(f"File: {file_path}")

    df = pd.read_csv(
        file_path,
        encoding="cp1252"
    )

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    return df


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df


# ============================================================
# PARSE TIMESTAMP
# ============================================================

def parse_timestamp(df):

    df = df.copy()

    date_column = "DATE_YYYY-MO-DD_HH-MI-SS_SSS"

    df["timestamp"] = pd.to_datetime(
        df[date_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    invalid = df["timestamp"].isna().sum()

    print("\nTimestamp validation:")
    print(f"Invalid timestamps : {invalid}")

    if invalid > 0:
        raise ValueError(
            "Invalid timestamps found in dataset."
        )

    return df


# ============================================================
# CREATE REAL TIME AXIS
# ============================================================

def create_time_axis(df):

    df = df.copy()

    first_timestamp = df["timestamp"].iloc[0]

    df["time_seconds"] = (
        df["timestamp"] - first_timestamp
    ).dt.total_seconds()

    df["dt"] = df["time_seconds"].diff()

    return df


# ============================================================
# EXTRACT IMU
# ============================================================

def extract_imu(df):

    accel_columns = [
        "ACCELEROMETER_X_m_s²",
        "ACCELEROMETER_Y_m_s²",
        "ACCELEROMETER_Z_m_s²"
    ]

    gyro_columns = [
        "GYROSCOPE_Yaw_rad_s",
        "GYROSCOPE_Pitch_rad_s",
        "GYROSCOPE_Roll_rad_s"
    ]

    gravity_columns = [
        "GRAVITY_X_m_s²",
        "GRAVITY_Y_m_s²",
        "GRAVITY_Z_m_s²"
    ]

    acceleration = df[accel_columns].to_numpy(dtype=float)

    gyroscope = df[gyro_columns].to_numpy(dtype=float)

    gravity = df[gravity_columns].to_numpy(dtype=float)

    return acceleration, gyroscope, gravity


# ============================================================
# SENSOR MAGNITUDES
# ============================================================

def calculate_magnitudes(acceleration, gyroscope, gravity):

    acceleration_magnitude = np.linalg.norm(
        acceleration,
        axis=1
    )

    gyroscope_magnitude = np.linalg.norm(
        gyroscope,
        axis=1
    )

    gravity_magnitude = np.linalg.norm(
        gravity,
        axis=1
    )

    return (
        acceleration_magnitude,
        gyroscope_magnitude,
        gravity_magnitude
    )


# ============================================================
# DETECT GYRO OUTLIERS
# ============================================================

def detect_gyro_outliers(gyroscope):

    gyro_magnitude = np.linalg.norm(
        gyroscope,
        axis=1
    )

    outlier_mask = (
        gyro_magnitude > GYRO_OUTLIER_THRESHOLD
    )

    return outlier_mask


# ============================================================
# CLEAN GYROSCOPE
# ============================================================

def clean_gyroscope(gyroscope, outlier_mask):

    gyro_clean = gyroscope.copy()

    for axis in range(3):

        series = pd.Series(
            gyro_clean[:, axis]
        )

        # Mark outliers as missing
        series[outlier_mask] = np.nan

        # Interpolate through bad samples
        series = series.interpolate(
            method="linear",
            limit_direction="both"
        )

        gyro_clean[:, axis] = series.to_numpy()

    return gyro_clean


# ============================================================
# GRAVITY COMPENSATION
# ============================================================

def calculate_linear_acceleration(
    acceleration,
    gravity
):

    # Accelerometer contains:
    #
    # acceleration = linear acceleration + gravity
    #
    # Therefore:
    #
    # linear acceleration = acceleration - gravity

    linear_acceleration = (
        acceleration - gravity
    )

    return linear_acceleration


# ============================================================
# DETECT LARGE TIMESTAMP GAPS
# ============================================================

def detect_time_gaps(df):

    dt = df["dt"].to_numpy()

    gap_mask = dt > LARGE_GAP_THRESHOLD

    gap_indices = np.where(gap_mask)[0]

    return gap_indices


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess():

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. CLEAN COLUMN NAMES
    # --------------------------------------------------------

    df = clean_column_names(df)

    # --------------------------------------------------------
    # 3. TIMESTAMP
    # --------------------------------------------------------

    df = parse_timestamp(df)

    # --------------------------------------------------------
    # 4. REAL TIME AXIS
    # --------------------------------------------------------

    df = create_time_axis(df)

    # --------------------------------------------------------
    # 5. EXTRACT IMU
    # --------------------------------------------------------

    acceleration, gyroscope, gravity = extract_imu(df)

    print("\nIMU shapes:")
    print(f"Acceleration : {acceleration.shape}")
    print(f"Gyroscope    : {gyroscope.shape}")
    print(f"Gravity      : {gravity.shape}")

    # --------------------------------------------------------
    # 6. MAGNITUDES
    # --------------------------------------------------------

    (
        acceleration_magnitude,
        gyroscope_magnitude,
        gravity_magnitude
    ) = calculate_magnitudes(
        acceleration,
        gyroscope,
        gravity
    )

    print("\nRaw sensor magnitudes:")

    print(
        f"Acceleration median : "
        f"{np.median(acceleration_magnitude):.4f} m/s²"
    )

    print(
        f"Acceleration maximum: "
        f"{np.max(acceleration_magnitude):.4f} m/s²"
    )

    print(
        f"Gyroscope median    : "
        f"{np.median(gyroscope_magnitude):.4f} rad/s"
    )

    print(
        f"Gyroscope maximum   : "
        f"{np.max(gyroscope_magnitude):.4f} rad/s"
    )

    print(
        f"Gravity median      : "
        f"{np.median(gravity_magnitude):.4f} m/s²"
    )

    # --------------------------------------------------------
    # 7. GYRO OUTLIERS
    # --------------------------------------------------------

    gyro_outlier_mask = detect_gyro_outliers(
        gyroscope
    )

    print("\nGyroscope outliers:")
    print(
        f"Outliers : "
        f"{np.sum(gyro_outlier_mask)}"
    )

    print(
        f"Clean    : "
        f"{len(gyro_outlier_mask) - np.sum(gyro_outlier_mask)}"
    )

    # --------------------------------------------------------
    # 8. CLEAN GYRO
    # --------------------------------------------------------

    gyroscope_clean = clean_gyroscope(
        gyroscope,
        gyro_outlier_mask
    )

    # --------------------------------------------------------
    # 9. GRAVITY COMPENSATION
    # --------------------------------------------------------

    linear_acceleration = (
        calculate_linear_acceleration(
            acceleration,
            gravity
        )
    )

    # --------------------------------------------------------
    # 10. TIME GAPS
    # --------------------------------------------------------

    gap_indices = detect_time_gaps(df)

    print("\nTimestamp gaps > 0.2 seconds:")

    print(
        f"Number of gaps: "
        f"{len(gap_indices)}"
    )

    for index in gap_indices:

        print(
            f"Row {index}: "
            f"{df['dt'].iloc[index]:.3f} seconds"
        )

    # --------------------------------------------------------
    # 11. CLEAN SENSOR CHECK
    # --------------------------------------------------------

    clean_gyro_magnitude = np.linalg.norm(
        gyroscope_clean,
        axis=1
    )

    linear_acceleration_magnitude = np.linalg.norm(
        linear_acceleration,
        axis=1
    )

    print("\nAfter preprocessing:")

    print(
        f"Clean gyro maximum : "
        f"{np.max(clean_gyro_magnitude):.4f} rad/s"
    )

    print(
        f"Linear acceleration median : "
        f"{np.median(linear_acceleration_magnitude):.4f} m/s²"
    )

    print(
        f"Linear acceleration maximum : "
        f"{np.max(linear_acceleration_magnitude):.4f} m/s²"
    )

    # --------------------------------------------------------
    # 12. RETURN EVERYTHING
    # --------------------------------------------------------

    return {
        "dataframe": df,
        "acceleration": acceleration,
        "gyroscope": gyroscope,
        "gravity": gravity,
        "gyroscope_clean": gyroscope_clean,
        "linear_acceleration": linear_acceleration,
        "gyro_outlier_mask": gyro_outlier_mask,
        "time_seconds": df["time_seconds"].to_numpy(),
        "dt": df["dt"].to_numpy(),
        "gap_indices": gap_indices
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    result = preprocess()

    print("\n" + "=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)

    print(
        "IMU preprocessing completed successfully."
    )