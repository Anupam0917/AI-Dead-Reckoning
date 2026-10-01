import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


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
    "prototype_v10_11_temporal_cnn.csv"
)

MODEL_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_11_temporal_cnn.pt"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW = 20
TRAIN_RATIO = 0.70

BATCH_SIZE = 256
EPOCHS = 15

LEARNING_RATE = 0.001

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


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 75)
print("V10.11 TEMPORAL CNN VELOCITY MODEL")
print("=" * 75)

print("\nDevice:", device)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("\n[1] Loading raw sensor data...")

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
# 2. FIND MAGNETOMETER
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


print("\nMagnetometer columns:")
print("X:", mag_x)
print("Y:", mag_y)
print("Z:", mag_z)


# ============================================================
# 3. PREPARE SENSOR DATA
# ============================================================

print("\n[2] Preparing sensor data...")


def numeric(column):

    return pd.to_numeric(
        raw[column],
        errors="coerce"
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
# 4. LOAD VELOCITY TARGETS
# ============================================================

print("\n[3] Loading velocity targets...")

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
# 5. MERGE
# ============================================================

print("\n[4] Matching sensors with targets...")

data = pd.merge_asof(

    target[
        [
            "timestamp",
            "time_s",
            "vn_mps",
            "ve_mps"
        ]
    ].sort_values("timestamp"),

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
# 6. CLEAN
# ============================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna(
    subset=[
        "gyro_yaw",
        "gyro_pitch",
        "gyro_roll",
        "mag_x",
        "mag_y",
        "mag_z",
        "vn_mps",
        "ve_mps"
    ]
).reset_index(
    drop=True
)


print(
    "Usable rows:",
    len(data)
)


# ============================================================
# 7. SENSOR MATRIX
# ============================================================

sensor_columns = [

    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",

    "mag_x",
    "mag_y",
    "mag_z"

]


X = data[
    sensor_columns
].values.astype(
    np.float32
)

Y = data[
    [
        "vn_mps",
        "ve_mps"
    ]
].values.astype(
    np.float32
)


# ============================================================
# 8. NORMALIZATION
# ============================================================

split_index = int(
    len(data) * TRAIN_RATIO
)


print(
    "\nTraining rows:",
    split_index
)

print(
    "Testing rows:",
    len(data) - split_index
)


# Normalize using TRAINING data only

train_sensor = X[
    :split_index
]

train_target = Y[
    :split_index
]


sensor_mean = np.mean(
    train_sensor,
    axis=0
)

sensor_std = np.std(
    train_sensor,
    axis=0
)

sensor_std[
    sensor_std < 1e-6
] = 1.0


target_mean = np.mean(
    train_target,
    axis=0
)

target_std = np.std(
    train_target,
    axis=0
)

target_std[
    target_std < 1e-6
] = 1.0


X_normalized = (
    X - sensor_mean
) / sensor_std


Y_normalized = (
    Y - target_mean
) / target_std


# ============================================================
# 9. BUILD SEQUENCES
# ============================================================

print("\n[5] Building temporal sequences...")

sequence_X = []
sequence_Y = []
sequence_indices = []


for i in range(
    WINDOW - 1,
    len(data)
):

    start = i - WINDOW + 1

    sequence_X.append(
        X_normalized[
            start:i + 1
        ]
    )

    sequence_Y.append(
        Y_normalized[i]
    )

    sequence_indices.append(
        i
    )


sequence_X = np.asarray(
    sequence_X,
    dtype=np.float32
)

sequence_Y = np.asarray(
    sequence_Y,
    dtype=np.float32
)

sequence_indices = np.asarray(
    sequence_indices
)


print(
    "Sequence shape:",
    sequence_X.shape
)

print(
    "Target shape:",
    sequence_Y.shape
)


# ============================================================
# 10. CHRONOLOGICAL SPLIT
# ============================================================

train_mask = (
    sequence_indices
    < split_index
)

test_mask = (
    sequence_indices
    >= split_index
)


X_train = sequence_X[
    train_mask
]

Y_train = sequence_Y[
    train_mask
]

X_test = sequence_X[
    test_mask
]

Y_test = sequence_Y[
    test_mask
]


print(
    "\nTraining sequences:",
    len(X_train)
)

print(
    "Testing sequences:",
    len(X_test)
)


# ============================================================
# 11. DATASET
# ============================================================

class VelocityDataset(
    Dataset
):

    def __init__(
        self,
        X,
        Y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.Y = torch.tensor(
            Y,
            dtype=torch.float32
        )


    def __len__(
        self
    ):

        return len(self.X)


    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.Y[index]
        )


train_dataset = VelocityDataset(
    X_train,
    Y_train
)

test_dataset = VelocityDataset(
    X_test,
    Y_test
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 12. TEMPORAL CNN
# ============================================================

class TemporalCNN(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Conv1d(
                in_channels=6,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(
                1
            )

        )


        self.head = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                2
            )

        )


    def forward(
        self,
        x
    ):

        # Input:
        # batch x window x sensors

        # Conv1D requires:
        # batch x sensors x window

        x = x.permute(
            0,
            2,
            1
        )

        x = self.network(
            x
        )

        x = self.head(
            x
        )

        return x


model = TemporalCNN().to(
    device
)


print("\nModel:")
print(model)


# ============================================================
# 13. LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 14. TRAINING
# ============================================================

print("\n")
print("=" * 75)
print("TRAINING")
print("=" * 75)


for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0


    for batch_X, batch_Y in train_loader:

        batch_X = batch_X.to(
            device
        )

        batch_Y = batch_Y.to(
            device
        )


        optimizer.zero_grad()


        prediction = model(
            batch_X
        )


        loss = criterion(
            prediction,
            batch_Y
        )


        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
            *
            len(batch_X)
        )


    epoch_loss = (
        total_loss
        /
        len(train_dataset)
    )


    print(
        "Epoch",
        epoch + 1,
        "/",
        EPOCHS,
        "| Loss:",
        round(
            epoch_loss,
            6
        )
    )


# ============================================================
# 15. SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "sensor_mean":
            sensor_mean,

        "sensor_std":
            sensor_std,

        "target_mean":
            target_mean,

        "target_std":
            target_std,

        "window":
            WINDOW,

        "sensor_columns":
            sensor_columns

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
# 16. TEST
# ============================================================

print("\n")
print("=" * 75)
print("TESTING")
print("=" * 75)


model.eval()

predictions = []
actuals = []


with torch.no_grad():

    for batch_X, batch_Y in test_loader:

        batch_X = batch_X.to(
            device
        )

        prediction = model(
            batch_X
        )

        predictions.append(
            prediction.cpu().numpy()
        )

        actuals.append(
            batch_Y.numpy()
        )


predictions = np.concatenate(
    predictions,
    axis=0
)

actuals = np.concatenate(
    actuals,
    axis=0
)


# Convert back to physical units

predictions = (
    predictions
    *
    target_std
    +
    target_mean
)

actuals = (
    actuals
    *
    target_std
    +
    target_mean
)


# ============================================================
# 17. METRICS
# ============================================================

north_error = (
    predictions[:, 0]
    -
    actuals[:, 0]
)

east_error = (
    predictions[:, 1]
    -
    actuals[:, 1]
)


north_mae = np.mean(
    np.abs(
        north_error
    )
)

north_rmse = np.sqrt(
    np.mean(
        north_error ** 2
    )
)

north_r2 = 1 - (
    np.sum(
        north_error ** 2
    )
    /
    np.sum(
        (
            actuals[:, 0]
            -
            np.mean(
                actuals[:, 0]
            )
        ) ** 2
    )
)


east_mae = np.mean(
    np.abs(
        east_error
    )
)

east_rmse = np.sqrt(
    np.mean(
        east_error ** 2
    )
)

east_r2 = 1 - (
    np.sum(
        east_error ** 2
    )
    /
    np.sum(
        (
            actuals[:, 1]
            -
            np.mean(
                actuals[:, 1]
            )
        ) ** 2
    )
)


print(
    "\nNorth MAE:",
    round(
        north_mae,
        4
    ),
    "m/s"
)

print(
    "North RMSE:",
    round(
        north_rmse,
        4
    ),
    "m/s"
)

print(
    "North R2:",
    round(
        north_r2,
        4
    )
)


print(
    "\nEast MAE:",
    round(
        east_mae,
        4
    ),
    "m/s"
)

print(
    "East RMSE:",
    round(
        east_rmse,
        4
    ),
    "m/s"
)

print(
    "East R2:",
    round(
        east_r2,
        4
    )
)


# ============================================================
# 18. SAVE PREDICTIONS
# ============================================================

test_indices = sequence_indices[
    test_mask
]

result = data.iloc[
    test_indices
].copy()


result = result[
    [
        "timestamp",
        "time_s",
        "vn_mps",
        "ve_mps"
    ]
].copy()


result[
    "vn_cnn_mps"
] = predictions[:, 0]

result[
    "ve_cnn_mps"
] = predictions[:, 1]


result[
    "vn_error_mps"
] = (
    predictions[:, 0]
    -
    actuals[:, 0]
)

result[
    "ve_error_mps"
] = (
    predictions[:, 1]
    -
    actuals[:, 1]
)


result.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nPredictions saved:"
)

print(
    OUTPUT_FILE
)


print("\n")
print("=" * 75)
print("V10.11 COMPLETE")
print("=" * 75)