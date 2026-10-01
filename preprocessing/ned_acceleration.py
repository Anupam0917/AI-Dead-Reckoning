import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.transform import Rotation


# ============================================================
# PATHS
# ============================================================

RAW_DATA_PATH = Path(
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

ORIENTATION_PATH = Path(
    "data/processed/dynamic_complementary_quaternion.csv"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "ned_acceleration.csv"

PLOT_DIR = Path("outputs")
PLOT_PATH = PLOT_DIR / "ned_acceleration_validation.png"


# ============================================================
# CONSTANTS
# ============================================================

GRAVITY = 9.80665

# Stationary accelerometer bias obtained previously
ACCEL_BIAS = np.array([
    -0.00645854,
     0.00997073,
     0.03871098
])


# ============================================================
# DATA LOADING
# ============================================================

def load_raw_dataset():
    print("=" * 70)
    print("STEP 5.12")
    print("NED ACCELERATION PROCESSING")
    print("=" * 70)

    print("\nLoading raw dataset...")

    df = pd.read_csv(
        RAW_DATA_PATH,
        encoding="cp1252"
    )

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    return df


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
# TIMESTAMP
# ============================================================

def create_timestamp(df):

    date_column = "DATE_YYYY-MO-DD_HH-MI-SS_SSS"

    if date_column not in df.columns:
        raise KeyError(
            f"Timestamp column not found: {date_column}"
        )

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
            "Invalid timestamps detected."
        )

    print(f"Start : {df['timestamp'].iloc[0]}")
    print(f"End   : {df['timestamp'].iloc[-1]}")

    return df


# ============================================================
# FIND ORIENTATION COLUMNS
# ============================================================

def find_orientation_columns(orientation_df):

    columns = orientation_df.columns.tolist()

    print("\nOrientation columns:")
    print(columns)

    # --------------------------------------------------------
    # Try quaternion columns first
    # --------------------------------------------------------

    quaternion_sets = [
        ["qw", "qx", "qy", "qz"],
        ["q_w", "q_x", "q_y", "q_z"],
        ["quaternion_w", "quaternion_x",
         "quaternion_y", "quaternion_z"],
        ["quat_w", "quat_x", "quat_y", "quat_z"],
    ]

    for candidate in quaternion_sets:

        if all(col in columns for col in candidate):
            print("\nQuaternion columns detected:")
            print(candidate)

            return {
                "type": "quaternion",
                "columns": candidate
            }

    # --------------------------------------------------------
    # Case-insensitive quaternion detection
    # --------------------------------------------------------

    lower_map = {
        col.lower(): col
        for col in columns
    }

    possible_sets = [
        ["qw", "qx", "qy", "qz"],
        ["q_w", "q_x", "q_y", "q_z"],
    ]

    for candidate in possible_sets:

        if all(name in lower_map for name in candidate):

            actual = [
                lower_map[name]
                for name in candidate
            ]

            print("\nQuaternion columns detected:")
            print(actual)

            return {
                "type": "quaternion",
                "columns": actual
            }

    raise KeyError(
        "\nCould not find quaternion columns in "
        "dynamic_complementary_quaternion.csv.\n"
        "Please print the CSV columns."
    )


# ============================================================
# LOAD ORIENTATION
# ============================================================

def load_orientation():

    print("\nLoading quaternion orientation...")

    if not ORIENTATION_PATH.exists():

        raise FileNotFoundError(
            f"\nOrientation file not found:\n"
            f"{ORIENTATION_PATH}\n\n"
            "Run dynamic_complementary_quaternion.py first."
        )

    orientation_df = pd.read_csv(
        ORIENTATION_PATH
    )

    print(
        f"Orientation rows : "
        f"{len(orientation_df)}"
    )

    info = find_orientation_columns(
        orientation_df
    )

    quaternion_columns = info["columns"]

    q = orientation_df[
        quaternion_columns
    ].to_numpy(dtype=float)

    return orientation_df, q


# ============================================================
# ACCELEROMETER
# ============================================================

def get_accelerometer(df):

    required = [
        "ACCELEROMETER_X_m_s²",
        "ACCELEROMETER_Y_m_s²",
        "ACCELEROMETER_Z_m_s²"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise KeyError(
            f"Missing accelerometer columns: {missing}"
        )

    accel = df[
        required
    ].to_numpy(dtype=float)

    return accel


# ============================================================
# GRAVITY SENSOR
# ============================================================

def get_gravity(df):

    required = [
        "GRAVITY_X_m_s²",
        "GRAVITY_Y_m_s²",
        "GRAVITY_Z_m_s²"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise KeyError(
            f"Missing gravity columns: {missing}"
        )

    gravity = df[
        required
    ].to_numpy(dtype=float)

    return gravity


# ============================================================
# GPS SPEED
# ============================================================

def get_gps_speed(df):

    column = "GPS_SPEED_Kmh"

    if column in df.columns:

        return df[column].to_numpy(
            dtype=float
        )

    return np.full(
        len(df),
        np.nan
    )


# ============================================================
# QUATERNION VALIDATION
# ============================================================

def validate_quaternions(q):

    print("\nQuaternion validation...")

    norms = np.linalg.norm(
        q,
        axis=1
    )

    print(
        f"Quaternion norm mean : "
        f"{np.mean(norms):.6f}"
    )

    print(
        f"Quaternion norm std  : "
        f"{np.std(norms):.6f}"
    )

    print(
        f"Quaternion norm min  : "
        f"{np.min(norms):.6f}"
    )

    print(
        f"Quaternion norm max  : "
        f"{np.max(norms):.6f}"
    )

    bad = (
        ~np.isfinite(norms)
        |
        (norms < 0.5)
        |
        (norms > 1.5)
    )

    print(
        f"Invalid quaternions  : "
        f"{np.sum(bad)}"
    )

    if np.any(bad):

        print(
            "Repairing invalid quaternion samples..."
        )

        valid_indices = np.where(~bad)[0]

        if len(valid_indices) == 0:
            raise ValueError(
                "No valid quaternions available."
            )

        for i in np.where(bad)[0]:

            nearest = valid_indices[
                np.argmin(
                    np.abs(
                        valid_indices - i
                    )
                )
            ]

            q[i] = q[nearest]

    # Normalize
    q_norm = np.linalg.norm(
        q,
        axis=1,
        keepdims=True
    )

    q = q / q_norm

    return q


# ============================================================
# BODY-FRAME LINEAR ACCELERATION
# ============================================================

def calculate_body_linear_acceleration(
    accel,
    gravity
):

    print("\nCalculating body-frame linear acceleration...")

    # Accelerometer includes gravity.
    #
    # Linear acceleration:
    #
    #       a_linear = a_accel - gravity
    #
    # This is performed before rotating to NED.

    corrected_accel = (
        accel - ACCEL_BIAS
    )

    linear_accel = (
        corrected_accel - gravity
    )

    return corrected_accel, linear_accel


# ============================================================
# BODY → NED
# ============================================================

def rotate_body_to_ned(
    linear_accel,
    q
):

    print("\nRotating acceleration:")
    print("BODY frame -> NED frame")

    ned_accel = np.zeros_like(
        linear_accel
    )

    for i in range(len(linear_accel)):

        qw, qx, qy, qz = q[i]

        # scipy Rotation expects:
        #
        # [x, y, z, w]
        #
        scipy_quaternion = np.array([
            qx,
            qy,
            qz,
            qw
        ])

        rotation = Rotation.from_quat(
            scipy_quaternion
        )

        ned_accel[i] = rotation.apply(
            linear_accel[i]
        )

    return ned_accel


# ============================================================
# ACCELERATION STATISTICS
# ============================================================

def print_statistics(
    body_linear,
    ned_accel,
    gps_speed
):

    print("\n" + "=" * 70)
    print("ACCELERATION STATISTICS")
    print("=" * 70)

    body_mag = np.linalg.norm(
        body_linear,
        axis=1
    )

    ned_horizontal = np.linalg.norm(
        ned_accel[:, :2],
        axis=1
    )

    ned_mag = np.linalg.norm(
        ned_accel,
        axis=1
    )

    print("\nBODY-FRAME LINEAR ACCELERATION")

    print(
        f"Mean magnitude   : "
        f"{np.mean(body_mag):.4f} m/s²"
    )

    print(
        f"Median magnitude : "
        f"{np.median(body_mag):.4f} m/s²"
    )

    print(
        f"Maximum magnitude: "
        f"{np.max(body_mag):.4f} m/s²"
    )

    print("\nNED ACCELERATION")

    print(
        f"North mean : "
        f"{np.mean(ned_accel[:, 0]):.4f}"
    )

    print(
        f"East mean  : "
        f"{np.mean(ned_accel[:, 1]):.4f}"
    )

    print(
        f"Down mean  : "
        f"{np.mean(ned_accel[:, 2]):.4f}"
    )

    print(
        f"Horizontal acceleration median : "
        f"{np.median(ned_horizontal):.4f} m/s²"
    )

    print(
        f"Horizontal acceleration max : "
        f"{np.max(ned_horizontal):.4f} m/s²"
    )

    print(
        f"NED acceleration max magnitude : "
        f"{np.max(ned_mag):.4f} m/s²"
    )

    if np.any(np.isfinite(gps_speed)):

        moving = gps_speed > 0.5

        if np.any(moving):

            print("\nMOVING SAMPLES")

            print(
                f"Samples : "
                f"{np.sum(moving)}"
            )

            print(
                f"GPS speed mean : "
                f"{np.mean(gps_speed[moving]):.2f} km/h"
            )

            print(
                f"Horizontal acceleration mean : "
                f"{np.mean(ned_horizontal[moving]):.4f} m/s²"
            )


# ============================================================
# STATIONARY VALIDATION
# ============================================================

def stationary_validation(
    ned_accel,
    gps_speed
):

    print("\n" + "=" * 70)
    print("STATIONARY NED ACCELERATION VALIDATION")
    print("=" * 70)

    stationary = (
        np.isfinite(gps_speed)
        &
        (gps_speed <= 0.5)
    )

    count = np.sum(stationary)

    print(
        f"\nStationary samples : {count}"
    )

    if count == 0:

        print(
            "No stationary samples found."
        )

        return

    stationary_ned = ned_accel[
        stationary
    ]

    horizontal = np.linalg.norm(
        stationary_ned[:, :2],
        axis=1
    )

    print("\nStationary NED means:")

    print(
        f"North : "
        f"{np.mean(stationary_ned[:, 0]):.6f}"
    )

    print(
        f"East  : "
        f"{np.mean(stationary_ned[:, 1]):.6f}"
    )

    print(
        f"Down  : "
        f"{np.mean(stationary_ned[:, 2]):.6f}"
    )

    print("\nStationary NED standard deviation:")

    print(
        f"North : "
        f"{np.std(stationary_ned[:, 0]):.6f}"
    )

    print(
        f"East  : "
        f"{np.std(stationary_ned[:, 1]):.6f}"
    )

    print(
        f"Down  : "
        f"{np.std(stationary_ned[:, 2]):.6f}"
    )

    print("\nHorizontal acceleration:")

    print(
        f"Mean : "
        f"{np.mean(horizontal):.6f} m/s²"
    )

    print(
        f"Median : "
        f"{np.median(horizontal):.6f} m/s²"
    )

    print(
        f"Max : "
        f"{np.max(horizontal):.6f} m/s²"
    )


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_results(
    df,
    q,
    corrected_accel,
    body_linear,
    ned_accel,
    gps_speed
):

    print("\nSaving processed acceleration...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = pd.DataFrame()

    output["timestamp"] = df[
        "timestamp"
    ]

    output["qw"] = q[:, 0]
    output["qx"] = q[:, 1]
    output["qy"] = q[:, 2]
    output["qz"] = q[:, 3]

    output["accel_x_corrected"] = (
        corrected_accel[:, 0]
    )

    output["accel_y_corrected"] = (
        corrected_accel[:, 1]
    )

    output["accel_z_corrected"] = (
        corrected_accel[:, 2]
    )

    output["linear_accel_body_x"] = (
        body_linear[:, 0]
    )

    output["linear_accel_body_y"] = (
        body_linear[:, 1]
    )

    output["linear_accel_body_z"] = (
        body_linear[:, 2]
    )

    output["accel_north"] = (
        ned_accel[:, 0]
    )

    output["accel_east"] = (
        ned_accel[:, 1]
    )

    output["accel_down"] = (
        ned_accel[:, 2]
    )

    output["horizontal_accel"] = (
        np.linalg.norm(
            ned_accel[:, :2],
            axis=1
        )
    )

    output["gps_speed_kmh"] = gps_speed

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print(
        f"Rows saved : {len(output)}"
    )


# ============================================================
# PLOT
# ============================================================

def generate_plot(
    df,
    ned_accel,
    gps_speed
):

    print("\nGenerating validation plot...")

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    time_seconds = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds().to_numpy()

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 12),
        sharex=True
    )

    # --------------------------------------------------------
    # North
    # --------------------------------------------------------

    axes[0].plot(
        time_seconds,
        ned_accel[:, 0],
        label="North acceleration"
    )

    axes[0].axhline(
        0,
        linestyle="--"
    )

    axes[0].set_ylabel(
        "m/s²"
    )

    axes[0].set_title(
        "NED North Acceleration"
    )

    axes[0].grid(
        True,
        alpha=0.3
    )

    axes[0].legend()

    # --------------------------------------------------------
    # East
    # --------------------------------------------------------

    axes[1].plot(
        time_seconds,
        ned_accel[:, 1],
        label="East acceleration"
    )

    axes[1].axhline(
        0,
        linestyle="--"
    )

    axes[1].set_ylabel(
        "m/s²"
    )

    axes[1].set_title(
        "NED East Acceleration"
    )

    axes[1].grid(
        True,
        alpha=0.3
    )

    axes[1].legend()

    # --------------------------------------------------------
    # Down
    # --------------------------------------------------------

    axes[2].plot(
        time_seconds,
        ned_accel[:, 2],
        label="Down acceleration"
    )

    axes[2].axhline(
        0,
        linestyle="--"
    )

    axes[2].set_ylabel(
        "m/s²"
    )

    axes[2].set_title(
        "NED Down Acceleration"
    )

    axes[2].grid(
        True,
        alpha=0.3
    )

    axes[2].legend()

    # --------------------------------------------------------
    # Horizontal + GPS
    # --------------------------------------------------------

    horizontal = np.linalg.norm(
        ned_accel[:, :2],
        axis=1
    )

    axes[3].plot(
        time_seconds,
        horizontal,
        label="Horizontal acceleration"
    )

    axes[3].set_ylabel(
        "m/s²"
    )

    axes[3].set_xlabel(
        "Time (seconds)"
    )

    axes[3].set_title(
        "Horizontal Acceleration vs GPS Speed"
    )

    ax2 = axes[3].twinx()

    ax2.plot(
        time_seconds,
        gps_speed,
        linestyle="--",
        label="GPS speed"
    )

    ax2.set_ylabel(
        "GPS speed (km/h)"
    )

    axes[3].grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_PATH,
        dpi=150
    )

    plt.show()

    print(
        f"Plot saved to: {PLOT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load raw data
    # --------------------------------------------------------

    df = load_raw_dataset()

    df = clean_column_names(
        df
    )

    df = create_timestamp(
        df
    )

    # --------------------------------------------------------
    # 2. Load orientation
    # --------------------------------------------------------

    orientation_df, q = (
        load_orientation()
    )

    # --------------------------------------------------------
    # 3. Check row alignment
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ROW ALIGNMENT VALIDATION")
    print("=" * 70)

    print(
        f"Raw dataset rows : {len(df)}"
    )

    print(
        f"Orientation rows : {len(orientation_df)}"
    )

    if len(df) != len(orientation_df):

        raise ValueError(
            "\nRaw dataset and orientation dataset "
            "have different numbers of rows."
        )

    print(
        "Row alignment : OK"
    )

    # --------------------------------------------------------
    # 4. Validate quaternion
    # --------------------------------------------------------

    q = validate_quaternions(
        q
    )

    # --------------------------------------------------------
    # 5. Sensors
    # --------------------------------------------------------

    accel = get_accelerometer(
        df
    )

    gravity = get_gravity(
        df
    )

    gps_speed = get_gps_speed(
        df
    )

    print("\nSensor shapes:")

    print(
        f"Accelerometer : {accel.shape}"
    )

    print(
        f"Gravity       : {gravity.shape}"
    )

    # --------------------------------------------------------
    # 6. Body-frame linear acceleration
    # --------------------------------------------------------

    (
        corrected_accel,
        body_linear
    ) = calculate_body_linear_acceleration(
        accel,
        gravity
    )

    # --------------------------------------------------------
    # 7. Body → NED
    # --------------------------------------------------------

    ned_accel = rotate_body_to_ned(
        body_linear,
        q
    )

    # --------------------------------------------------------
    # 8. Statistics
    # --------------------------------------------------------

    print_statistics(
        body_linear,
        ned_accel,
        gps_speed
    )

    # --------------------------------------------------------
    # 9. Stationary validation
    # --------------------------------------------------------

    stationary_validation(
        ned_accel,
        gps_speed
    )

    # --------------------------------------------------------
    # 10. Save
    # --------------------------------------------------------

    save_results(
        df,
        q,
        corrected_accel,
        body_linear,
        ned_accel,
        gps_speed
    )

    # --------------------------------------------------------
    # 11. Plot
    # --------------------------------------------------------

    generate_plot(
        df,
        ned_accel,
        gps_speed
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 5.12 COMPLETE")
    print("=" * 70)

    print(
        "\nNED acceleration processing completed."
    )

    print(
        "Velocity integration : NOT performed"
    )

    print(
        "Position integration : NOT performed"
    )

    print(
        "\nNext:"
    )

    print(
        "Validate NED acceleration."
    )

    print(
        "Only after validation will we proceed "
        "to velocity integration."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()