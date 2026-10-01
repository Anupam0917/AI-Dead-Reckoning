import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# V10.18
# TEMPORAL GRU VELOCITY ESTIMATOR
#
# Input:
#   20 samples ~= 2 seconds
#   Gyro XYZ + Magnetometer XYZ
#
# Output:
#   North velocity
#   East velocity
#
# Important:
#   Chronological split
#   Training-only normalization
#   No GNSS input to the model
# ============================================================


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

MODEL_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_18_temporal_gru.pt"
)

PREDICTION_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_18_temporal_gru.csv"
)

OUTAGE_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_18_gru_outage_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20

TRAIN_RATIO = 0.70

BATCH_SIZE = 256

EPOCHS = 20

LEARNING_RATE = 0.001

HIDDEN_SIZE = 64

NUM_LAYERS = 2

DROPOUT = 0.2

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("V10.18 TEMPORAL GRU VELOCITY ESTIMATOR")
print("=" * 80)

print(
    "\nDevice:",
    DEVICE
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH,
    "samples"
)

print(
    "Approximate temporal window:",
    SEQUENCE_LENGTH * 0.1,
    "seconds"
)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("\n[1] Loading raw dataset...")

raw = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

raw.columns = raw.columns.str.strip()

print(
    "Rows:",
    len(raw)
)


# ============================================================
# 2. PARSE TIMESTAMP
# ============================================================

raw["timestamp"] = pd.to_datetime(

    raw[
        "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
    ],

    format="%Y-%m-%d %H:%M:%S:%f",

    errors="coerce"

)


# ============================================================
# 3. FIND MAGNETOMETER
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


if (
    mag_x is None
    or mag_y is None
    or mag_z is None
):

    raise RuntimeError(
        "Could not detect magnetometer columns."
    )


print("\nMagnetometer:")

print(
    "X:",
    mag_x
)

print(
    "Y:",
    mag_y
)

print(
    "Z:",
    mag_z
)


# ============================================================
# 4. NUMERIC HELPER
# ============================================================

def numeric(column):

    return pd.to_numeric(
        raw[column],
        errors="coerce"
    )


# ============================================================
# 5. SENSOR DATA
# ============================================================

print(
    "\n[2] Preparing sensor data..."
)


sensor = pd.DataFrame({

    "timestamp":
        raw["timestamp"],

    "gyro_yaw":
        numeric(
            "GYROSCOPE Yaw (rad/s)"
        ),

    "gyro_pitch":
        numeric(
            "GYROSCOPE Pitch (rad/s)"
        ),

    "gyro_roll":
        numeric(
            "GYROSCOPE Roll (rad/s)"
        ),

    "mag_x":
        numeric(mag_x),

    "mag_y":
        numeric(mag_y),

    "mag_z":
        numeric(mag_z)

})


# ============================================================
# 6. LOAD VELOCITY TARGETS
# ============================================================

print(
    "\n[3] Loading velocity targets..."
)


target = pd.read_csv(
    TARGET_FILE
)


target["timestamp"] = pd.to_datetime(
    target["timestamp"]
)


target = target.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# 7. SYNCHRONIZE
# ============================================================

print(
    "\n[4] Synchronizing sensors and targets..."
)


data = pd.merge_asof(

    target[
        [
            "timestamp",
            "time_s",
            "vn_mps",
            "ve_mps"
        ]
    ].sort_values(
        "timestamp"
    ),

    sensor.sort_values(
        "timestamp"
    ),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta(
        "50ms"
    )

)


# ============================================================
# 8. CLEAN SENSOR DATA
# ============================================================

feature_columns = [

    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",

    "mag_x",
    "mag_y",
    "mag_z"

]


for column in feature_columns:

    data[column] = (
        data[column]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .interpolate(
            limit_direction="both"
        )
        .fillna(0)
    )


data["vn_mps"] = (
    data["vn_mps"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .interpolate(
        limit_direction="both"
    )
    .fillna(0)
)


data["ve_mps"] = (
    data["ve_mps"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .interpolate(
        limit_direction="both"
    )
    .fillna(0)
)


data = data.reset_index(
    drop=True
)


# ============================================================
# 9. CHRONOLOGICAL SPLIT
# ============================================================

n = len(data)

train_end = int(
    n * TRAIN_RATIO
)

print(
    "\nTotal samples:",
    n
)

print(
    "Training samples:",
    train_end
)

print(
    "Testing samples:",
    n - train_end
)

print(
    "Training time:",
    round(
        data["time_s"].iloc[0],
        2
    ),
    "to",
    round(
        data["time_s"].iloc[train_end - 1],
        2
    )
)

print(
    "Testing time:",
    round(
        data["time_s"].iloc[train_end],
        2
    ),
    "to",
    round(
        data["time_s"].iloc[-1],
        2
    )
)


# ============================================================
# 10. RAW ARRAYS
# ============================================================

sensor_array = data[
    feature_columns
].to_numpy(
    dtype=np.float32
)


target_array = data[
    [
        "vn_mps",
        "ve_mps"
    ]
].to_numpy(
    dtype=np.float32
)


# ============================================================
# 11. TRAIN-ONLY NORMALIZATION
# ============================================================

print(
    "\n[5] Calculating training normalization..."
)


train_sensor = sensor_array[
    :train_end
]


feature_mean = train_sensor.mean(
    axis=0
)

feature_std = train_sensor.std(
    axis=0
)


feature_std[
    feature_std < 1e-6
] = 1.0


sensor_normalized = (

    sensor_array
    -
    feature_mean

) / feature_std


# ============================================================
# 12. CREATE SEQUENCES
#
# Each target at index i uses:
#
#   i-19 ... i
#
# This prevents future information from entering
# the prediction.
# ============================================================

print(
    "\n[6] Creating temporal sequences..."
)


def create_sequences(
    start_index,
    end_index
):

    X_list = []
    y_list = []
    index_list = []


    first_index = max(

        start_index,

        SEQUENCE_LENGTH - 1

    )


    for i in range(
        first_index,
        end_index
    ):

        sequence_start = (
            i
            -
            SEQUENCE_LENGTH
            +
            1
        )


        # Never allow a training sequence to
        # contain test data.

        if sequence_start < start_index:

            continue


        sequence = sensor_normalized[
            sequence_start:
            i + 1
        ]


        X_list.append(
            sequence
        )


        y_list.append(
            target_array[i]
        )


        index_list.append(
            i
        )


    return (

        np.asarray(
            X_list,
            dtype=np.float32
        ),

        np.asarray(
            y_list,
            dtype=np.float32
        ),

        np.asarray(
            index_list,
            dtype=np.int64
        )

    )


X_train, y_train, train_indices = create_sequences(
    0,
    train_end
)


X_test, y_test, test_indices = create_sequences(
    train_end,
    n
)


print(
    "\nTraining sequence shape:",
    X_train.shape
)

print(
    "Training target shape:",
    y_train.shape
)

print(
    "Testing sequence shape:",
    X_test.shape
)

print(
    "Testing target shape:",
    y_test.shape
)


# ============================================================
# 13. PYTORCH DATASETS
# ============================================================

train_dataset = TensorDataset(

    torch.from_numpy(
        X_train
    ),

    torch.from_numpy(
        y_train
    )

)


test_dataset = TensorDataset(

    torch.from_numpy(
        X_test
    ),

    torch.from_numpy(
        y_test
    )

)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


# ============================================================
# 14. GRU MODEL
# ============================================================

class TemporalGRU(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout
    ):

        super().__init__()


        self.gru = nn.GRU(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )

        )


        self.head = nn.Sequential(

            nn.Linear(
                hidden_size,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                2
            )

        )


    def forward(
        self,
        x
    ):

        output, hidden = self.gru(
            x
        )


        # Last temporal state

        last_state = output[:, -1, :]


        return self.head(
            last_state
        )


# ============================================================
# 15. CREATE MODEL
# ============================================================

model = TemporalGRU(

    input_size=6,

    hidden_size=HIDDEN_SIZE,

    num_layers=NUM_LAYERS,

    dropout=DROPOUT

).to(
    DEVICE
)


print(
    "\n[7] GRU architecture:"
)

print(
    model
)


# ============================================================
# 16. OPTIMIZER
# ============================================================

criterion = nn.MSELoss()


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


# ============================================================
# 17. TRAINING
# ============================================================

print(
    "\n[8] Training GRU..."
)


for epoch in range(
    EPOCHS
):

    model.train()

    running_loss = 0.0

    sample_count = 0


    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            DEVICE
        )

        batch_y = batch_y.to(
            DEVICE
        )


        optimizer.zero_grad()


        prediction = model(
            batch_x
        )


        loss = criterion(
            prediction,
            batch_y
        )


        loss.backward()


        optimizer.step()


        batch_size = (
            batch_x.shape[0]
        )


        running_loss += (
            loss.item()
            *
            batch_size
        )


        sample_count += (
            batch_size
        )


    epoch_loss = (
        running_loss
        /
        sample_count
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Train MSE: {epoch_loss:.6f}"
    )


# ============================================================
# 18. SAVE MODEL
# ============================================================

torch.save(

    {
        "model_state_dict":
            model.state_dict(),

        "feature_mean":
            feature_mean,

        "feature_std":
            feature_std,

        "sequence_length":
            SEQUENCE_LENGTH,

        "feature_columns":
            feature_columns,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS

    },

    MODEL_FILE

)


print(
    "\nModel saved:"
)

print(
    MODEL_FILE
)


# ============================================================
# 19. TEST PREDICTIONS
# ============================================================

print(
    "\n[9] Generating test predictions..."
)


model.eval()


predictions = []


with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(
            DEVICE
        )

        prediction = model(
            batch_x
        )

        predictions.append(
            prediction.cpu().numpy()
        )


predictions = np.concatenate(
    predictions,
    axis=0
)


# ============================================================
# 20. TEST METRICS
# ============================================================

true_north = y_test[:, 0]

true_east = y_test[:, 1]

pred_north = predictions[:, 0]

pred_east = predictions[:, 1]


north_mae = mean_absolute_error(
    true_north,
    pred_north
)

north_rmse = np.sqrt(
    mean_squared_error(
        true_north,
        pred_north
    )
)

north_r2 = r2_score(
    true_north,
    pred_north
)


east_mae = mean_absolute_error(
    true_east,
    pred_east
)

east_rmse = np.sqrt(
    mean_squared_error(
        true_east,
        pred_east
    )
)

east_r2 = r2_score(
    true_east,
    pred_east
)


true_speed = np.sqrt(

    true_north ** 2
    +
    true_east ** 2

)


pred_speed = np.sqrt(

    pred_north ** 2
    +
    pred_east ** 2

)


speed_mae = mean_absolute_error(
    true_speed,
    pred_speed
)

speed_rmse = np.sqrt(
    mean_squared_error(
        true_speed,
        pred_speed
    )
)


# ============================================================
# 21. PRINT TEST METRICS
# ============================================================

print("\n")
print("=" * 80)
print("V10.18 TEST METRICS")
print("=" * 80)


print(
    "\nNorth velocity:"
)

print(
    "MAE:",
    round(
        north_mae,
        4
    ),
    "m/s"
)

print(
    "RMSE:",
    round(
        north_rmse,
        4
    ),
    "m/s"
)

print(
    "R²:",
    round(
        north_r2,
        4
    )
)


print(
    "\nEast velocity:"
)

print(
    "MAE:",
    round(
        east_mae,
        4
    ),
    "m/s"
)

print(
    "RMSE:",
    round(
        east_rmse,
        4
    ),
    "m/s"
)

print(
    "R²:",
    round(
        east_r2,
        4
    )
)


print(
    "\nSpeed:"
)

print(
    "MAE:",
    round(
        speed_mae,
        4
    ),
    "m/s"
)

print(
    "MAE:",
    round(
        speed_mae * 3.6,
        4
    ),
    "km/h"
)

print(
    "RMSE:",
    round(
        speed_rmse,
        4
    ),
    "m/s"
)


# ============================================================
# 22. BUILD FULL PREDICTION TABLE
# ============================================================

prediction_data = data.iloc[
    test_indices
].copy()


prediction_data[
    "vn_ai_gru"
] = pred_north


prediction_data[
    "ve_ai_gru"
] = pred_east


prediction_data[
    "north_error"
] = (

    pred_north
    -
    true_north

)


prediction_data[
    "east_error"
] = (

    pred_east
    -
    true_east

)


prediction_data[
    "speed_true_mps"
] = true_speed


prediction_data[
    "speed_ai_mps"
] = pred_speed


prediction_data[
    "speed_error_mps"
] = (

    pred_speed
    -
    true_speed

)


prediction_data[
    "speed_true_kmh"
] = (
    true_speed
    *
    3.6
)


prediction_data[
    "speed_ai_kmh"
] = (
    pred_speed
    *
    3.6
)


prediction_data[
    [
        "timestamp",
        "time_s",
        "vn_mps",
        "ve_mps",
        "vn_ai_gru",
        "ve_ai_gru",
        "north_error",
        "east_error",
        "speed_true_mps",
        "speed_ai_mps",
        "speed_error_mps",
        "speed_true_kmh",
        "speed_ai_kmh"
    ]
].to_csv(

    PREDICTION_FILE,

    index=False

)


print(
    "\nPredictions saved:"
)

print(
    PREDICTION_FILE
)


# ============================================================
# 23. OUTAGE EVALUATION
# ============================================================

print(
    "\n[10] Evaluating GNSS-denied windows..."
)


OUTAGES = [

    (7500.0, 30.0),
    (7500.0, 60.0),
    (7500.0, 120.0),

    (8500.0, 30.0),
    (8500.0, 60.0),
    (8500.0, 120.0),

    (9500.0, 30.0),
    (9500.0, 60.0),
    (9500.0, 120.0),

    (10000.0, 30.0),
    (10000.0, 60.0),
    (10000.0, 120.0)

]


outage_results = []


for start, duration in OUTAGES:

    end = (
        start
        +
        duration
    )


    subset = prediction_data[
        (
            prediction_data["time_s"]
            >= start
        )
        &
        (
            prediction_data["time_s"]
            <= end
        )
    ].copy()


    if len(subset) < 10:

        print(
            f"Skipping {start}/{duration}: "
            "insufficient test samples."
        )

        continue


    time = subset[
        "time_s"
    ].to_numpy(
        dtype=float
    )


    true_vn_window = subset[
        "vn_mps"
    ].to_numpy(
        dtype=float
    )


    true_ve_window = subset[
        "ve_mps"
    ].to_numpy(
        dtype=float
    )


    ai_vn_window = subset[
        "vn_ai_gru"
    ].to_numpy(
        dtype=float
    )


    ai_ve_window = subset[
        "ve_ai_gru"
    ].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # Velocity MAE
    # --------------------------------------------------------

    window_north_mae = np.mean(
        np.abs(
            ai_vn_window
            -
            true_vn_window
        )
    )


    window_east_mae = np.mean(
        np.abs(
            ai_ve_window
            -
            true_ve_window
        )
    )


    # --------------------------------------------------------
    # Integrate velocity
    # --------------------------------------------------------

    true_north_pos = np.zeros(
        len(subset)
    )

    true_east_pos = np.zeros(
        len(subset)
    )

    ai_north_pos = np.zeros(
        len(subset)
    )

    ai_east_pos = np.zeros(
        len(subset)
    )


    for i in range(
        1,
        len(subset)
    ):

        dt = np.clip(

            time[i]
            -
            time[i - 1],

            0.001,

            0.5

        )


        true_north_pos[i] = (

            true_north_pos[i - 1]

            +

            true_vn_window[i - 1]
            *
            dt

        )


        true_east_pos[i] = (

            true_east_pos[i - 1]

            +

            true_ve_window[i - 1]
            *
            dt

        )


        ai_north_pos[i] = (

            ai_north_pos[i - 1]

            +

            ai_vn_window[i - 1]
            *
            dt

        )


        ai_east_pos[i] = (

            ai_east_pos[i - 1]

            +

            ai_ve_window[i - 1]
            *
            dt

        )


    position_error = np.sqrt(

        (
            ai_north_pos
            -
            true_north_pos
        ) ** 2

        +

        (
            ai_east_pos
            -
            true_east_pos
        ) ** 2

    )


    mean_position_error = np.mean(
        position_error
    )


    final_position_error = (
        position_error[-1]
    )


    max_position_error = np.max(
        position_error
    )


    true_distance = np.sum(

        np.sqrt(

            np.diff(
                true_north_pos
            ) ** 2

            +

            np.diff(
                true_east_pos
            ) ** 2

        )

    )


    ai_distance = np.sum(

        np.sqrt(

            np.diff(
                ai_north_pos
            ) ** 2

            +

            np.diff(
                ai_east_pos
            ) ** 2

        )

    )


    if true_distance > 0:

        drift_percent = (

            abs(
                ai_distance
                -
                true_distance
            )

            /
            true_distance
            *
            100

        )

    else:

        drift_percent = np.nan


    result = {

        "start_s":
            start,

        "duration_s":
            duration,

        "samples":
            len(subset),

        "north_velocity_mae_mps":
            window_north_mae,

        "east_velocity_mae_mps":
            window_east_mae,

        "mean_position_error_m":
            mean_position_error,

        "final_position_error_m":
            final_position_error,

        "max_position_error_m":
            max_position_error,

        "reference_distance_m":
            true_distance,

        "gru_distance_m":
            ai_distance,

        "drift_percent":
            drift_percent

    }


    outage_results.append(
        result
    )


    print(
        f"\n{start:.0f}s / "
        f"{duration:.0f}s"
    )

    print(
        "North MAE:",
        round(
            window_north_mae,
            4
        ),
        "m/s"
    )

    print(
        "East MAE:",
        round(
            window_east_mae,
            4
        ),
        "m/s"
    )

    print(
        "Mean position error:",
        round(
            mean_position_error,
            3
        ),
        "m"
    )

    print(
        "Final position error:",
        round(
            final_position_error,
            3
        ),
        "m"
    )

    print(
        "Drift:",
        round(
            drift_percent,
            3
        ),
        "%"
    )


# ============================================================
# 24. SAVE OUTAGE RESULTS
# ============================================================

outage_dataframe = pd.DataFrame(
    outage_results
)


outage_dataframe.to_csv(
    OUTAGE_FILE,
    index=False
)


# ============================================================
# 25. OUTAGE SUMMARY
# ============================================================

if len(outage_dataframe) > 0:

    print("\n")
    print("=" * 80)
    print("V10.18 MULTI-OUTAGE SUMMARY")
    print("=" * 80)


    print(
        "\nValid outage windows:",
        len(outage_dataframe)
    )


    print(
        "\nMean final position error:",
        round(
            outage_dataframe[
                "final_position_error_m"
            ].mean(),
            3
        ),
        "m"
    )


    print(
        "Median final position error:",
        round(
            outage_dataframe[
                "final_position_error_m"
            ].median(),
            3
        ),
        "m"
    )


    print(
        "Maximum final position error:",
        round(
            outage_dataframe[
                "final_position_error_m"
            ].max(),
            3
        ),
        "m"
    )


    print(
        "\nMean position error:",
        round(
            outage_dataframe[
                "mean_position_error_m"
            ].mean(),
            3
        ),
        "m"
    )


    print(
        "Mean drift:",
        round(
            outage_dataframe[
                "drift_percent"
            ].mean(),
            3
        ),
        "%"
    )


    print(
        "Median drift:",
        round(
            outage_dataframe[
                "drift_percent"
            ].median(),
            3
        ),
        "%"
    )


    print(
        "\n"
        + "-" * 100
    )


    print(
        outage_dataframe.round(
            3
        ).to_string(
            index=False
        )
    )


print(
    "\nOutage results saved:"
)

print(
    OUTAGE_FILE
)


# ============================================================
# 26. FINAL
# ============================================================

print("\n")
print("=" * 80)
print("V10.18 COMPLETE")
print("=" * 80)

print(
    "\nNow compare V10.18 against V10.16."
)

print(
    "Do not modify the GRU based on one result."
)

print(
    "The multi-outage benchmark decides whether "
    "the temporal model actually helps."
)