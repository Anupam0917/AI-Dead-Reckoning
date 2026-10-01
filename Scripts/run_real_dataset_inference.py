from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# BACKEND IMPORTS
# ============================================================

from backend.model import NavigationModel
from backend.preprocessing import SensorPreprocessor


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
    / "M (Driver B)"
    / "S-M.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "navigation_model.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "production_real_sensor_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

ENCODING = "cp1252"

# V10.18 Temporal GRU expects 20 consecutive samples
SEQUENCE_LENGTH = 20

# IO-VNBD smartphone sensor is approximately 10 Hz
DEFAULT_DT = 0.1


# ============================================================
# GYROSCOPE COLUMN MAPPING
# ============================================================

GYRO_COLUMN_MAP = {
    "GYROSCOPE Yaw (rad/s)": "gyro_yaw",
    "GYROSCOPE Pitch (rad/s)": "gyro_pitch",
    "GYROSCOPE Roll (rad/s)": "gyro_roll",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_axis_column(columns, sensor_name, axis):
    """
    Find a sensor column by sensor name + axis.

    This intentionally ignores the unit portion of the column
    name because the original CSV contains encoding variations
    such as:

        ÎμT
        Î¼T

    Example:

        MAGNETIC FIELD X (Î¼T)

    is detected simply using:

        MAGNETIC FIELD + X
    """

    prefix = f"{sensor_name} {axis}".upper()

    matches = [
        column
        for column in columns
        if str(column).strip().upper().startswith(prefix)
    ]

    if len(matches) == 0:
        raise ValueError(
            f"Could not find column for "
            f"{sensor_name} {axis}."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple columns found for "
            f"{sensor_name} {axis}: {matches}"
        )

    return matches[0]


def print_separator():
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print("REAL IO-VNBD SENSOR INFERENCE")
    print_separator()

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    print("\nDataset:")
    print(DATA_PATH)

    print("\nModel:")
    print(MODEL_PATH)

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\nLoading IO-VNBD dataset...")

    df = pd.read_csv(
        DATA_PATH,
        encoding=ENCODING
    )

    # Clean whitespace around column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    print(
        f"Loaded {len(df):,} rows "
        f"and {len(df.columns)} columns."
    )

    # ========================================================
    # DISPLAY DATASET COLUMNS
    # ========================================================

    print("\nDataset columns:")

    for index, column in enumerate(df.columns):
        print(f"  {index:02d}: {column}")

    # ========================================================
    # REQUIRED GPS + TIME COLUMNS
    # ========================================================

    required_base_columns = [
        "DATE (YYYY-MO-DD HH-MI-SS_SSS)",
        "GPS LATITUDE (degrees)",
        "GPS LONGITUDE (degrees)",
    ]

    missing_base = [
        column
        for column in required_base_columns
        if column not in df.columns
    ]

    if missing_base:

        print("\nMissing required columns:")

        for column in missing_base:
            print("  ", repr(column))

        raise ValueError(
            "Required GPS/time columns are missing."
        )

    # ========================================================
    # GYROSCOPE COLUMNS
    # ========================================================

    missing_gyro = [
        column
        for column in GYRO_COLUMN_MAP
        if column not in df.columns
    ]

    if missing_gyro:

        print("\nMissing gyroscope columns:")

        for column in missing_gyro:
            print("  ", repr(column))

        raise ValueError(
            "Required gyroscope columns are missing."
        )

    # ========================================================
    # MAGNETOMETER COLUMNS
    # ========================================================

    # Do NOT hardcode the μT part.
    #
    # The CSV has encoding/mojibake variations in this portion.
    #
    # Instead we detect:
    #
    # MAGNETIC FIELD X
    # MAGNETIC FIELD Y
    # MAGNETIC FIELD Z

    mag_x_column = find_axis_column(
        df.columns,
        "MAGNETIC FIELD",
        "X"
    )

    mag_y_column = find_axis_column(
        df.columns,
        "MAGNETIC FIELD",
        "Y"
    )

    mag_z_column = find_axis_column(
        df.columns,
        "MAGNETIC FIELD",
        "Z"
    )

    print("\nDetected magnetometer columns:")

    print(
        f"  mag_x -> {repr(mag_x_column)}"
    )

    print(
        f"  mag_y -> {repr(mag_y_column)}"
    )

    print(
        f"  mag_z -> {repr(mag_z_column)}"
    )

    # ========================================================
    # FINAL COLUMN MAP
    # ========================================================

    column_map = {

        # Gyroscope
        "GYROSCOPE Yaw (rad/s)": "gyro_yaw",
        "GYROSCOPE Pitch (rad/s)": "gyro_pitch",
        "GYROSCOPE Roll (rad/s)": "gyro_roll",

        # Magnetometer
        mag_x_column: "mag_x",
        mag_y_column: "mag_y",
        mag_z_column: "mag_z",
    }

    # ========================================================
    # LOAD PRODUCTION MODEL
    # ========================================================

    print("\nLoading production model...")

    model = NavigationModel(
        MODEL_PATH
    )

    preprocessor = SensorPreprocessor(
        model
    )

    print("\nProduction model information:")

    print(
        f"  Sequence length : "
        f"{model.sequence_length}"
    )

    print(
        f"  Input features  : "
        f"{model.input_size}"
    )

    print(
        f"  Hidden size     : "
        f"{model.hidden_size}"
    )

    print(
        f"  GRU layers      : "
        f"{model.num_layers}"
    )

    print("\nModel feature order:")

    for index, feature in enumerate(
        model.feature_columns
    ):
        print(
            f"  {index}: {feature}"
        )

    # ========================================================
    # VERIFY MODEL FEATURES
    # ========================================================

    expected_features = [
        "gyro_yaw",
        "gyro_pitch",
        "gyro_roll",
        "mag_x",
        "mag_y",
        "mag_z",
    ]

    if model.feature_columns != expected_features:

        raise ValueError(
            "\nModel feature order does not match "
            "the expected V10.18 production order.\n"
            f"Expected: {expected_features}\n"
            f"Found:    {model.feature_columns}"
        )

    print(
        "\nModel feature order verified successfully."
    )

    # ========================================================
    # PREPARE TIMESTAMP
    # ========================================================

    print("\nParsing timestamps...")

    df["timestamp"] = pd.to_datetime(
        df["DATE (YYYY-MO-DD HH-MI-SS_SSS)"],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    invalid_timestamps = (
        df["timestamp"].isna().sum()
    )

    if invalid_timestamps > 0:

        print(
            f"Removing {invalid_timestamps:,} "
            f"rows with invalid timestamps."
        )

        df = df.dropna(
            subset=["timestamp"]
        )

    df = df.reset_index(
        drop=True
    )

    # ========================================================
    # PREPARE SENSOR COLUMNS
    # ========================================================

    print("\nPreparing sensor data...")

    for source_column, target_column in column_map.items():

        df[target_column] = pd.to_numeric(
            df[source_column],
            errors="coerce"
        )

    sensor_columns = [
        "gyro_yaw",
        "gyro_pitch",
        "gyro_roll",
        "mag_x",
        "mag_y",
        "mag_z",
    ]

    # ========================================================
    # SENSOR VALIDITY
    # ========================================================

    invalid_sensor_rows = (
        ~df[sensor_columns]
        .notna()
        .all(axis=1)
    )

    invalid_count = int(
        invalid_sensor_rows.sum()
    )

    print(
        f"Rows with missing sensor values: "
        f"{invalid_count:,}"
    )

    if invalid_count > 0:

        df = df.loc[
            ~invalid_sensor_rows
        ].reset_index(
            drop=True
        )

    print(
        f"Rows available for inference: "
        f"{len(df):,}"
    )

    # ========================================================
    # TIME
    # ========================================================

    df["time_s"] = (
        df["timestamp"] -
        df["timestamp"].iloc[0]
    ).dt.total_seconds()

    # ========================================================
    # SAMPLING INTERVAL
    # ========================================================

    dt_values = (
        df["time_s"]
        .diff()
        .dropna()
    )

    if len(dt_values) > 0:

        median_dt = float(
            dt_values.median()
        )

        mean_dt = float(
            dt_values.mean()
        )

        max_dt = float(
            dt_values.max()
        )

        print(
            f"\nMedian sensor interval : "
            f"{median_dt:.4f} s"
        )

        print(
            f"Mean sensor interval   : "
            f"{mean_dt:.4f} s"
        )

        print(
            f"Maximum time gap       : "
            f"{max_dt:.4f} s"
        )

        if median_dt > 0:

            print(
                f"Approximate frequency  : "
                f"{1.0 / median_dt:.2f} Hz"
            )

    else:

        median_dt = DEFAULT_DT

        print(
            "\nCould not calculate sensor "
            "sampling interval."
        )

    # ========================================================
    # ALLOCATE PREDICTION ARRAYS
    # ========================================================

    total_rows = len(df)

    prediction_vn = np.full(
        total_rows,
        np.nan,
        dtype=np.float32
    )

    prediction_ve = np.full(
        total_rows,
        np.nan,
        dtype=np.float32
    )

    prediction_speed = np.full(
        total_rows,
        np.nan,
        dtype=np.float32
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    print_separator()
    print("RUNNING TEMPORAL GRU INFERENCE")
    print_separator()

    print(
        f"\nTotal rows: {total_rows:,}"
    )

    print(
        f"Window size: {SEQUENCE_LENGTH} samples"
    )

    print(
        "\nEach prediction uses:"
    )

    print(
        "  gyro_yaw"
    )

    print(
        "  gyro_pitch"
    )

    print(
        "  gyro_roll"
    )

    print(
        "  mag_x"
    )

    print(
        "  mag_y"
    )

    print(
        "  mag_z"
    )

    print("\nStarting inference...\n")

    # ========================================================
    # SLIDING WINDOW
    # ========================================================

    for end_index in range(
        SEQUENCE_LENGTH - 1,
        total_rows
    ):

        start_index = (
            end_index -
            SEQUENCE_LENGTH +
            1
        )

        # ----------------------------------------------------
        # Get 20 consecutive rows
        # ----------------------------------------------------

        window = df.iloc[
            start_index:end_index + 1
        ]

        # ----------------------------------------------------
        # Build model input
        # ----------------------------------------------------

        samples = []

        for _, row in window.iterrows():

            samples.append(
                {
                    "gyro_yaw": float(
                        row["gyro_yaw"]
                    ),

                    "gyro_pitch": float(
                        row["gyro_pitch"]
                    ),

                    "gyro_roll": float(
                        row["gyro_roll"]
                    ),

                    "mag_x": float(
                        row["mag_x"]
                    ),

                    "mag_y": float(
                        row["mag_y"]
                    ),

                    "mag_z": float(
                        row["mag_z"]
                    ),
                }
            )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        sequence = preprocessor.prepare(
            samples
        )

        # ----------------------------------------------------
        # AI inference
        # ----------------------------------------------------

        prediction = model.predict(
            sequence
        )

        # ----------------------------------------------------
        # Store prediction
        # ----------------------------------------------------

        prediction_vn[end_index] = (
            prediction["vn_mps"]
        )

        prediction_ve[end_index] = (
            prediction["ve_mps"]
        )

        prediction_speed[end_index] = (
            prediction["speed_mps"]
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            end_index % 5000 == 0
            or end_index == total_rows - 1
        ):

            progress = (
                end_index /
                max(total_rows - 1, 1)
            ) * 100

            print(
                f"  Progress: "
                f"{progress:6.2f}% "
                f"({end_index:,}/{total_rows:,})"
            )

    # ========================================================
    # ADD AI PREDICTIONS
    # ========================================================

    df["vn_ai_mps"] = (
        prediction_vn
    )

    df["ve_ai_mps"] = (
        prediction_ve
    )

    df["ai_speed_mps"] = (
        prediction_speed
    )

    df["ai_speed_kmh"] = (
        df["ai_speed_mps"] * 3.6
    )

    # ========================================================
    # GPS REFERENCE DATA
    # ========================================================

    df["latitude"] = pd.to_numeric(
        df["GPS LATITUDE (degrees)"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["GPS LONGITUDE (degrees)"],
        errors="coerce"
    )

    # GPS speed may contain missing values
    if "GPS SPEED (Kmh)" in df.columns:

        df["gps_speed_kmh"] = pd.to_numeric(
            df["GPS SPEED (Kmh)"],
            errors="coerce"
        )

    else:

        df["gps_speed_kmh"] = np.nan

    # ========================================================
    # AI SPEED STATISTICS
    # ========================================================

    valid_ai_speed = (
        df["ai_speed_kmh"]
        .dropna()
    )

    if len(valid_ai_speed) > 0:

        print("\nAI speed statistics:")

        print(
            f"  Mean   : "
            f"{valid_ai_speed.mean():.3f} km/h"
        )

        print(
            f"  Median : "
            f"{valid_ai_speed.median():.3f} km/h"
        )

        print(
            f"  Min    : "
            f"{valid_ai_speed.min():.3f} km/h"
        )

        print(
            f"  Max    : "
            f"{valid_ai_speed.max():.3f} km/h"
        )

    # ========================================================
    # OUTPUT COLUMNS
    # ========================================================

    output_columns = [
        "timestamp",
        "time_s",

        "latitude",
        "longitude",

        "gps_speed_kmh",

        "gyro_yaw",
        "gyro_pitch",
        "gyro_roll",

        "mag_x",
        "mag_y",
        "mag_z",

        "vn_ai_mps",
        "ve_ai_mps",

        "ai_speed_mps",
        "ai_speed_kmh",
    ]

    output_df = df[
        output_columns
    ].copy()

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    valid_predictions = int(
        output_df["vn_ai_mps"]
        .notna()
        .sum()
    )

    print_separator()
    print("REAL SENSOR INFERENCE COMPLETE")
    print_separator()

    print(
        f"\nTotal rows: "
        f"{len(output_df):,}"
    )

    print(
        f"Valid AI predictions: "
        f"{valid_predictions:,}"
    )

    print(
        f"Rows without prediction: "
        f"{len(output_df) - valid_predictions:,}"
    )

    print("\nOutput saved to:")

    print(
        OUTPUT_PATH
    )

    # ========================================================
    # FIRST FEW PREDICTIONS
    # ========================================================

    print("\nFirst 5 AI predictions:")

    preview = (
        output_df[
            [
                "timestamp",
                "vn_ai_mps",
                "ve_ai_mps",
                "ai_speed_kmh",
            ]
        ]
        .dropna()
        .head(5)
    )

    if len(preview) > 0:

        print(
            preview.to_string(
                index=False
            )
        )

    else:

        print(
            "No valid predictions found."
        )

    print("\nOutput file:")

    print(
        OUTPUT_PATH
    )

    print_separator()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()