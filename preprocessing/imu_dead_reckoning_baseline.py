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


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path=DATA_PATH):

    print("=" * 60)
    print("IMU DEAD-RECKONING BASELINE")
    print("=" * 60)

    print("Loading dataset...")
    print(f"File: {file_path}")

    df = pd.read_csv(
        file_path,
        encoding="cp1252"
    )

    print("\nDataset loaded successfully.")
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
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # STEP 1: LOAD DATASET
    # ========================================================

    df = load_dataset()

    df = clean_column_names(df)

    print("\nCleaned columns:")

    for column in df.columns:
        print(" -", column)


    # ========================================================
    # STEP 2: REAL TIMESTAMP
    # ========================================================

    print("\n" + "=" * 60)
    print("TIMESTAMP PROCESSING")
    print("=" * 60)

    # Dataset format:
    #
    # 2019-09-07 09:13:29:506
    #
    # Note that milliseconds use ':' instead of '.'

    df["timestamp"] = pd.to_datetime(
        df["DATE_YYYY-MO-DD_HH-MI-SS_SSS"],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    # Check timestamp conversion

    timestamp_nan_count = (
        df["timestamp"].isna().sum()
    )

    print(
        f"Invalid timestamps : "
        f"{timestamp_nan_count}"
    )

    print(
        f"Start : "
        f"{df['timestamp'].iloc[0]}"
    )

    print(
        f"End   : "
        f"{df['timestamp'].iloc[-1]}"
    )

    # Seconds relative to first real timestamp

    df["timestamp_seconds"] = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds()


    # ========================================================
    # STEP 3: DATASET TIME
    # ========================================================

    df["time_seconds"] = (
        pd.to_numeric(
            df["TIME_SINCE_START_ms"],
            errors="coerce"
        ) / 1000.0
    )

    time = (
        df["time_seconds"]
        .to_numpy(dtype=float)
    )

    timestamp_time = (
        df["timestamp_seconds"]
        .to_numpy(dtype=float)
    )


    # ========================================================
    # STEP 4: EXTRACT ACCELEROMETER
    # ========================================================

    acceleration_columns = [
        "ACCELEROMETER_X_m_s²",
        "ACCELEROMETER_Y_m_s²",
        "ACCELEROMETER_Z_m_s²"
    ]

    acceleration = df[
        acceleration_columns
    ].to_numpy(dtype=float)


    # ========================================================
    # STEP 5: EXTRACT GYROSCOPE
    # ========================================================

    gyroscope_columns = [
        "GYROSCOPE_Yaw_rad_s",
        "GYROSCOPE_Pitch_rad_s",
        "GYROSCOPE_Roll_rad_s"
    ]

    gyroscope = df[
        gyroscope_columns
    ].to_numpy(dtype=float)


    # ========================================================
    # STEP 6: EXTRACT GRAVITY
    # ========================================================

    gravity_columns = [
        "GRAVITY_X_m_s²",
        "GRAVITY_Y_m_s²",
        "GRAVITY_Z_m_s²"
    ]

    gravity = df[
        gravity_columns
    ].to_numpy(dtype=float)


    # ========================================================
    # STEP 7: DATA SHAPES
    # ========================================================

    print("\n" + "=" * 60)
    print("IMU DATA EXTRACTION")
    print("=" * 60)

    print(
        f"Acceleration shape : "
        f"{acceleration.shape}"
    )

    print(
        f"Gyroscope shape    : "
        f"{gyroscope.shape}"
    )

    print(
        f"Gravity shape      : "
        f"{gravity.shape}"
    )


    # ========================================================
    # STEP 8: DATASET TIME ANALYSIS
    # ========================================================

    dt = np.diff(time)

    print("\n" + "=" * 60)
    print("DATASET TIME INTERVAL ANALYSIS")
    print("=" * 60)

    print(
        f"Start time : "
        f"{time[0]:.3f} seconds"
    )

    print(
        f"End time   : "
        f"{time[-1]:.3f} seconds"
    )

    print(
        f"Median dt  : "
        f"{np.median(dt):.4f} seconds"
    )

    print(
        f"Mean dt    : "
        f"{np.mean(dt):.4f} seconds"
    )

    print(
        f"Minimum dt : "
        f"{np.min(dt):.4f} seconds"
    )

    print(
        f"Maximum dt : "
        f"{np.max(dt):.4f} seconds"
    )


    # ========================================================
    # STEP 9: REAL TIMESTAMP ANALYSIS
    # ========================================================

    timestamp_dt = np.diff(
        timestamp_time
    )

    print("\n" + "=" * 60)
    print("REAL TIMESTAMP VALIDATION")
    print("=" * 60)

    print(
        f"Timestamp start : "
        f"{df['timestamp'].iloc[0]}"
    )

    print(
        f"Timestamp end   : "
        f"{df['timestamp'].iloc[-1]}"
    )

    print(
        f"Median timestamp dt : "
        f"{np.median(timestamp_dt):.4f} seconds"
    )

    print(
        f"Mean timestamp dt   : "
        f"{np.mean(timestamp_dt):.4f} seconds"
    )

    print(
        f"Minimum timestamp dt : "
        f"{np.min(timestamp_dt):.4f} seconds"
    )

    print(
        f"Maximum timestamp dt : "
        f"{np.max(timestamp_dt):.4f} seconds"
    )


    # ========================================================
    # STEP 10: NEGATIVE TIME JUMPS
    # ========================================================

    negative_indices = np.where(
        dt < 0
    )[0]

    print("\n" + "=" * 60)
    print("NEGATIVE TIME JUMPS")
    print("=" * 60)

    print(
        f"Number of negative jumps: "
        f"{len(negative_indices)}"
    )

    if len(negative_indices) > 0:

        for index in negative_indices:

            print("\nProblem detected:")

            print(
                f"Row             : "
                f"{index}"
            )

            print(
                f"Previous time   : "
                f"{time[index]:.3f} s"
            )

            print(
                f"Next time       : "
                f"{time[index + 1]:.3f} s"
            )

            print(
                f"dt              : "
                f"{dt[index]:.3f} s"
            )

            print(
                f"Previous timestamp : "
                f"{df['timestamp'].iloc[index]}"
            )

            print(
                f"Next timestamp     : "
                f"{df['timestamp'].iloc[index + 1]}"
            )

    else:

        print(
            "No negative time jumps detected."
        )


    # ========================================================
    # STEP 11: LARGE DATASET TIME GAPS
    # ========================================================

    large_gap_indices = np.where(
        dt > 0.2
    )[0]

    print("\n" + "=" * 60)
    print("DATASET TIME GAPS > 0.2 SECONDS")
    print("=" * 60)

    print(
        f"Number of large gaps: "
        f"{len(large_gap_indices)}"
    )

    if len(large_gap_indices) > 0:

        for index in large_gap_indices:

            print(
                f"Row {index} -> "
                f"time = {time[index]:.3f}s, "
                f"dt = {dt[index]:.3f}s"
            )

    else:

        print(
            "No large gaps detected."
        )


    # ========================================================
    # STEP 12: REAL TIMESTAMP GAPS
    # ========================================================

    timestamp_gap_indices = np.where(
        timestamp_dt > 0.2
    )[0]

    print("\n" + "=" * 60)
    print("REAL TIMESTAMP GAPS > 0.2 SECONDS")
    print("=" * 60)

    print(
        f"Number of timestamp gaps: "
        f"{len(timestamp_gap_indices)}"
    )

    if len(timestamp_gap_indices) > 0:

        for index in timestamp_gap_indices:

            print(
                f"Row {index}"
            )

            print(
                f"Timestamp : "
                f"{df['timestamp'].iloc[index]}"
            )

            print(
                f"Next      : "
                f"{df['timestamp'].iloc[index + 1]}"
            )

            print(
                f"dt        : "
                f"{timestamp_dt[index]:.3f}s"
            )

            print()


    # ========================================================
    # STEP 13: FIRST SENSOR SAMPLES
    # ========================================================

    print("=" * 60)
    print("FIRST SENSOR SAMPLES")
    print("=" * 60)

    print("\nAcceleration:")
    print(acceleration[0])

    print("\nGyroscope:")
    print(gyroscope[0])

    print("\nGravity:")
    print(gravity[0])


    # ========================================================
    # STEP 14: MISSING VALUE CHECK
    # ========================================================

    print("\n" + "=" * 60)
    print("MISSING VALUE CHECK")
    print("=" * 60)

    print(
        "Acceleration NaN:",
        np.isnan(acceleration).any()
    )

    print(
        "Gyroscope NaN   :",
        np.isnan(gyroscope).any()
    )

    print(
        "Gravity NaN     :",
        np.isnan(gravity).any()
    )

    print(
        "Dataset time NaN:",
        np.isnan(time).any()
    )

    print(
        "Timestamp NaT   :",
        df["timestamp"].isna().any()
    )


    # ========================================================
    # STEP 15: SENSOR MAGNITUDE CHECK
    # ========================================================

    acceleration_magnitude = np.linalg.norm(
        acceleration,
        axis=1
    )

    gyro_magnitude = np.linalg.norm(
        gyroscope,
        axis=1
    )

    gravity_magnitude = np.linalg.norm(
        gravity,
        axis=1
    )

    print("\n" + "=" * 60)
    print("SENSOR MAGNITUDE CHECK")
    print("=" * 60)

    print("\nAcceleration magnitude:")

    print(
        f"Median : "
        f"{np.median(acceleration_magnitude):.4f} m/s²"
    )

    print(
        f"Maximum: "
        f"{np.max(acceleration_magnitude):.4f} m/s²"
    )

    print("\nGyroscope magnitude:")

    print(
        f"Median : "
        f"{np.median(gyro_magnitude):.4f} rad/s"
    )

    print(
        f"Maximum: "
        f"{np.max(gyro_magnitude):.4f} rad/s"
    )

    print("\nGravity magnitude:")

    print(
        f"Median : "
        f"{np.median(gravity_magnitude):.4f} m/s²"
    )

    print(
        f"Minimum: "
        f"{np.min(gravity_magnitude):.4f} m/s²"
    )

    print(
        f"Maximum: "
        f"{np.max(gravity_magnitude):.4f} m/s²"
    )


    # ========================================================
    # STEP 16: COMPARE THE TWO TIME SOURCES
    # ========================================================

    time_difference = (
        timestamp_time - time
    )

    print("\n" + "=" * 60)
    print("TIME SOURCE COMPARISON")
    print("=" * 60)

    print(
        f"Initial difference : "
        f"{time_difference[0]:.6f} seconds"
    )

    print(
        f"Final difference   : "
        f"{time_difference[-1]:.6f} seconds"
    )

    print(
        f"Maximum absolute difference : "
        f"{np.max(np.abs(time_difference)):.6f} seconds"
    )


    # ========================================================
    # STEP 17: FINAL STATUS
    # ========================================================

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)

    print(
        "IMU data successfully extracted."
    )

    print(
        "Timestamp parsing and time consistency "
        "have been checked."
    )