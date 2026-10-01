# ============================================================
# sensor_bias_calibration.py
#
# STEP 5.8
#
# Stationary sensor bias calibration
#
# IMPORTANT:
#   - NO velocity integration
#   - NO position integration
#   - NO trajectory generation
#
# We estimate:
#   1. Gyroscope bias
#   2. Accelerometer bias
#   3. Magnetometer reference
#
# Stationary window:
#   3133.60 s -> 3141.70 s
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

STATIONARY_START = 3133.60
STATIONARY_END = 3141.70


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("STEP 5.8")
    print("STATIONARY SENSOR BIAS CALIBRATION")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_PATH,
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
        .str.replace(
            " ",
            "_"
        )
        .str.replace(
            "(",
            "",
            regex=False
        )
        .str.replace(
            ")",
            "",
            regex=False
        )
        .str.replace(
            "/",
            "_",
            regex=False
        )
    )

    return df


# ============================================================
# TIMESTAMP
# ============================================================

def prepare_timestamp(df):

    date_column = "DATE_YYYY-MO-DD_HH-MI-SS_SSS"

    df["timestamp"] = pd.to_datetime(
        df[date_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    if df["timestamp"].isna().any():

        raise ValueError(
            "Invalid timestamps found."
        )

    df["time_seconds"] = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    return df


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, prefix):

    matches = [
        column
        for column in df.columns
        if column.startswith(prefix)
    ]

    if not matches:

        raise KeyError(
            f"Column not found: {prefix}"
        )

    return matches[0]


# ============================================================
# EXTRACT SENSOR DATA
# ============================================================

def extract_sensors(df):

    accel_columns = [
        find_column(
            df,
            "ACCELEROMETER_X"
        ),
        find_column(
            df,
            "ACCELEROMETER_Y"
        ),
        find_column(
            df,
            "ACCELEROMETER_Z"
        )
    ]

    gravity_columns = [
        find_column(
            df,
            "GRAVITY_X"
        ),
        find_column(
            df,
            "GRAVITY_Y"
        ),
        find_column(
            df,
            "GRAVITY_Z"
        )
    ]

    gyro_columns = [
        find_column(
            df,
            "GYROSCOPE_Roll"
        ),
        find_column(
            df,
            "GYROSCOPE_Pitch"
        ),
        find_column(
            df,
            "GYROSCOPE_Yaw"
        )
    ]

    magnetometer_columns = [
        find_column(
            df,
            "MAGNETIC_FIELD_X"
        ),
        find_column(
            df,
            "MAGNETIC_FIELD_Y"
        ),
        find_column(
            df,
            "MAGNETIC_FIELD_Z"
        )
    ]

    acceleration = (
        df[accel_columns]
        .to_numpy(
            dtype=float
        )
    )

    gravity = (
        df[gravity_columns]
        .to_numpy(
            dtype=float
        )
    )

    gyro = (
        df[gyro_columns]
        .to_numpy(
            dtype=float
        )
    )

    magnetometer = (
        df[magnetometer_columns]
        .to_numpy(
            dtype=float
        )
    )

    time = (
        df["time_seconds"]
        .to_numpy(
            dtype=float
        )
    )

    return (
        time,
        acceleration,
        gravity,
        gyro,
        magnetometer
    )


# ============================================================
# SELECT STATIONARY WINDOW
# ============================================================

def select_stationary_window(
    time,
    acceleration,
    gravity,
    gyro,
    magnetometer
):

    mask = (
        (time >= STATIONARY_START)
        &
        (time <= STATIONARY_END)
    )

    if np.sum(mask) == 0:

        raise ValueError(
            "Stationary window contains no samples."
        )

    return (
        time[mask],
        acceleration[mask],
        gravity[mask],
        gyro[mask],
        magnetometer[mask]
    )


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    name,
    data
):

    mean = np.mean(
        data,
        axis=0
    )

    median = np.median(
        data,
        axis=0
    )

    std = np.std(
        data,
        axis=0
    )

    minimum = np.min(
        data,
        axis=0
    )

    maximum = np.max(
        data,
        axis=0
    )

    print(
        f"\n{name}"
    )

    print(
        "Mean:"
    )

    print(
        mean
    )

    print(
        "Median:"
    )

    print(
        median
    )

    print(
        "Std:"
    )

    print(
        std
    )

    print(
        "Minimum:"
    )

    print(
        minimum
    )

    print(
        "Maximum:"
    )

    print(
        maximum
    )

    return (
        mean,
        median,
        std
    )


# ============================================================
# GYROSCOPE BIAS
# ============================================================

def calculate_gyro_bias(
    gyro
):

    bias = np.mean(
        gyro,
        axis=0
    )

    return bias


# ============================================================
# ACCELEROMETER BIAS
#
# At rest:
#
# accelerometer ≈ gravity
#
# Therefore:
#
# accelerometer_bias =
#       mean(accelerometer)
#       - mean(gravity)
# ============================================================

def calculate_accel_bias(
    acceleration,
    gravity
):

    accel_mean = np.mean(
        acceleration,
        axis=0
    )

    gravity_mean = np.mean(
        gravity,
        axis=0
    )

    bias = (
        accel_mean
        -
        gravity_mean
    )

    return bias


# ============================================================
# MAGNETOMETER REFERENCE
# ============================================================

def calculate_magnetometer_reference(
    magnetometer
):

    mean = np.mean(
        magnetometer,
        axis=0
    )

    magnitude = np.linalg.norm(
        magnetometer,
        axis=1
    )

    return (
        mean,
        magnitude
    )


# ============================================================
# SENSOR NOISE
# ============================================================

def calculate_noise_statistics(
    gyro,
    acceleration,
    magnetometer
):

    gyro_std = np.std(
        gyro,
        axis=0
    )

    accel_std = np.std(
        acceleration,
        axis=0
    )

    magnetometer_std = np.std(
        magnetometer,
        axis=0
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATIONARY SENSOR NOISE"
    )

    print(
        "=" * 70
    )

    print(
        "\nGyroscope standard deviation:"
    )

    print(
        gyro_std
    )

    print(
        "\nAccelerometer standard deviation:"
    )

    print(
        accel_std
    )

    print(
        "\nMagnetometer standard deviation:"
    )

    print(
        magnetometer_std
    )


# ============================================================
# BIAS CORRECTION CHECK
# ============================================================

def verify_gyro_correction(
    gyro,
    gyro_bias
):

    corrected = (
        gyro
        -
        gyro_bias
    )

    corrected_mean = np.mean(
        corrected,
        axis=0
    )

    corrected_std = np.std(
        corrected,
        axis=0
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GYROSCOPE BIAS CORRECTION"
    )

    print(
        "=" * 70
    )

    print(
        "\nOriginal gyro mean:"
    )

    print(
        np.mean(
            gyro,
            axis=0
        )
    )

    print(
        "\nEstimated gyro bias:"
    )

    print(
        gyro_bias
    )

    print(
        "\nCorrected gyro mean:"
    )

    print(
        corrected_mean
    )

    print(
        "\nCorrected gyro std:"
    )

    print(
        corrected_std
    )

    return corrected


# ============================================================
# ACCELEROMETER CORRECTION CHECK
# ============================================================

def verify_accel_correction(
    acceleration,
    accel_bias,
    gravity
):

    corrected = (
        acceleration
        -
        accel_bias
    )

    corrected_mean = np.mean(
        corrected,
        axis=0
    )

    gravity_mean = np.mean(
        gravity,
        axis=0
    )

    difference = (
        corrected_mean
        -
        gravity_mean
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ACCELEROMETER BIAS CORRECTION"
    )

    print(
        "=" * 70
    )

    print(
        "\nOriginal accelerometer mean:"
    )

    print(
        np.mean(
            acceleration,
            axis=0
        )
    )

    print(
        "\nEstimated accelerometer bias:"
    )

    print(
        accel_bias
    )

    print(
        "\nCorrected accelerometer mean:"
    )

    print(
        corrected_mean
    )

    print(
        "\nMean gravity:"
    )

    print(
        gravity_mean
    )

    print(
        "\nCorrected acceleration - gravity:"
    )

    print(
        difference
    )

    return corrected


# ============================================================
# PLOT
# ============================================================

def plot_calibration(
    time,
    gyro,
    acceleration,
    magnetometer,
    gyro_bias,
    accel_bias
):

    relative_time = (
        time
        -
        time[0]
    )

    corrected_gyro = (
        gyro
        -
        gyro_bias
    )

    corrected_accel = (
        acceleration
        -
        accel_bias
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        sharex=True
    )

    # --------------------------------------------------------
    # Gyroscope
    # --------------------------------------------------------

    axes[0].plot(
        relative_time,
        gyro[:, 0],
        label="Gyro X"
    )

    axes[0].plot(
        relative_time,
        gyro[:, 1],
        label="Gyro Y"
    )

    axes[0].plot(
        relative_time,
        gyro[:, 2],
        label="Gyro Z"
    )

    axes[0].axhline(
        0.0,
        linestyle="--"
    )

    axes[0].set_ylabel(
        "rad/s"
    )

    axes[0].set_title(
        "Stationary Gyroscope"
    )

    axes[0].legend()

    axes[0].grid(True)

    # --------------------------------------------------------
    # Corrected gyroscope
    # --------------------------------------------------------

    axes[1].plot(
        relative_time,
        corrected_gyro[:, 0],
        label="Corrected Gyro X"
    )

    axes[1].plot(
        relative_time,
        corrected_gyro[:, 1],
        label="Corrected Gyro Y"
    )

    axes[1].plot(
        relative_time,
        corrected_gyro[:, 2],
        label="Corrected Gyro Z"
    )

    axes[1].axhline(
        0.0,
        linestyle="--"
    )

    axes[1].set_ylabel(
        "rad/s"
    )

    axes[1].set_title(
        "Bias-Corrected Gyroscope"
    )

    axes[1].legend()

    axes[1].grid(True)

    # --------------------------------------------------------
    # Corrected acceleration
    # --------------------------------------------------------

    axes[2].plot(
        relative_time,
        corrected_accel[:, 0],
        label="Accel X"
    )

    axes[2].plot(
        relative_time,
        corrected_accel[:, 1],
        label="Accel Y"
    )

    axes[2].plot(
        relative_time,
        corrected_accel[:, 2],
        label="Accel Z"
    )

    axes[2].set_xlabel(
        "Time since stationary window start (s)"
    )

    axes[2].set_ylabel(
        "m/s²"
    )

    axes[2].set_title(
        "Bias-Corrected Accelerometer"
    )

    axes[2].legend()

    axes[2].grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    df = clean_column_names(
        df
    )

    df = prepare_timestamp(
        df
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    (
        time,
        acceleration,
        gravity,
        gyro,
        magnetometer
    ) = extract_sensors(
        df
    )

    # --------------------------------------------------------
    # Select stationary window
    # --------------------------------------------------------

    (
        window_time,
        window_acceleration,
        window_gravity,
        window_gyro,
        window_magnetometer
    ) = select_stationary_window(
        time,
        acceleration,
        gravity,
        gyro,
        magnetometer
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SELECTED STATIONARY WINDOW"
    )

    print(
        "=" * 70
    )

    print(
        f"\nStart : "
        f"{window_time[0]:.3f} s"
    )

    print(
        f"End   : "
        f"{window_time[-1]:.3f} s"
    )

    print(
        f"Duration : "
        f"{window_time[-1] - window_time[0]:.3f} s"
    )

    print(
        f"Samples : "
        f"{len(window_time)}"
    )

    # --------------------------------------------------------
    # Raw statistics
    # --------------------------------------------------------

    print_statistics(
        "GYROSCOPE",
        window_gyro
    )

    print_statistics(
        "ACCELEROMETER",
        window_acceleration
    )

    print_statistics(
        "GRAVITY SENSOR",
        window_gravity
    )

    print_statistics(
        "MAGNETOMETER",
        window_magnetometer
    )

    # --------------------------------------------------------
    # Gyro bias
    # --------------------------------------------------------

    gyro_bias = calculate_gyro_bias(
        window_gyro
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ESTIMATED GYROSCOPE BIAS"
    )

    print(
        "=" * 70
    )

    print(
        "\nBias [rad/s]:"
    )

    print(
        gyro_bias
    )

    print(
        "\nBias magnitude:"
    )

    print(
        np.linalg.norm(
            gyro_bias
        )
    )

    # --------------------------------------------------------
    # Accelerometer bias
    # --------------------------------------------------------

    accel_bias = calculate_accel_bias(
        window_acceleration,
        window_gravity
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ESTIMATED ACCELEROMETER BIAS"
    )

    print(
        "=" * 70
    )

    print(
        "\nBias [m/s²]:"
    )

    print(
        accel_bias
    )

    print(
        "\nBias magnitude:"
    )

    print(
        np.linalg.norm(
            accel_bias
        )
    )

    # --------------------------------------------------------
    # Magnetometer
    # --------------------------------------------------------

    (
        magnetometer_reference,
        magnetometer_magnitude
    ) = calculate_magnetometer_reference(
        window_magnetometer
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MAGNETOMETER REFERENCE"
    )

    print(
        "=" * 70
    )

    print(
        "\nMean magnetic field:"
    )

    print(
        magnetometer_reference
    )

    print(
        "\nMagnetic magnitude:"
    )

    print(
        f"Mean   : "
        f"{np.mean(magnetometer_magnitude):.4f} µT"
    )

    print(
        f"Std    : "
        f"{np.std(magnetometer_magnitude):.4f} µT"
    )

    print(
        f"Minimum: "
        f"{np.min(magnetometer_magnitude):.4f} µT"
    )

    print(
        f"Maximum: "
        f"{np.max(magnetometer_magnitude):.4f} µT"
    )

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------

    calculate_noise_statistics(
        window_gyro,
        window_acceleration,
        window_magnetometer
    )

    # --------------------------------------------------------
    # Correct gyro
    # --------------------------------------------------------

    corrected_gyro = verify_gyro_correction(
        window_gyro,
        gyro_bias
    )

    # --------------------------------------------------------
    # Correct accelerometer
    # --------------------------------------------------------

    corrected_acceleration = (
        verify_accel_correction(
            window_acceleration,
            accel_bias,
            window_gravity
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    print(
        "\nGenerating calibration plot..."
    )

    plot_calibration(
        window_time,
        window_gyro,
        window_acceleration,
        window_magnetometer,
        gyro_bias,
        accel_bias
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 5.8 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSensor bias calibration completed."

    )

    print(
        "\nVelocity integration : NOT performed"
    )

    print(
        "Position integration : NOT performed"
    )

    print(
        "\nThese bias estimates will later be used by:"
    )

    print(
        "1. Quaternion orientation"
    )

    print(
        "2. IMU propagation"
    )

    print(
        "3. EKF / IEKF"
    )

    print(
        "4. AI reliability model"
    )