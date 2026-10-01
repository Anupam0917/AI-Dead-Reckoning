import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# V10.21
# Temporal GRU with Velocity Context
# ============================================================

RAW_FILE = (
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/M (Driver B)/S-M.csv"
)

TARGET_FILE = (
    "data/processed/prototype_v9_velocity_targets.csv"
)

MODEL_FILE = (
    "data/processed/prototype_v10_21_temporal_velocity_gru.pt"
)

OUTPUT_FILE = (
    "data/processed/prototype_v10_21_temporal_velocity_gru.csv"
)

SEED = 42

SEQ_LEN = 20
TRAIN_RATIO = 0.70

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 80)
print("V10.21 TEMPORAL GRU WITH VELOCITY CONTEXT")
print("=" * 80)

print("\nDevice:", DEVICE)


# ============================================================
# Helper
# ============================================================

def find_column(columns, keywords):

    for col in columns:

        name = col.lower()

        if all(k.lower() in name for k in keywords):
            return col

    return None


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

if len(df) != len(targets):

    raise ValueError(
        f"Row mismatch: raw={len(df)}, "
        f"targets={len(targets)}"
    )


# ============================================================
# 3. Synchronize
# ============================================================

print("\n[3] Synchronizing data...")

data = df.copy()

data["timestamp"] = pd.to_datetime(
    targets["timestamp"]
)

data["vn_true"] = targets[
    "vn_mps"
].astype(float)

data["ve_true"] = targets[
    "ve_mps"
].astype(float)

data = data.reset_index(drop=True)


# ============================================================
# 4. Detect magnetometer
# ============================================================

mag_x = find_column(
    data.columns,
    ["MAGNETIC", "FIELD", "X"]
)

mag_y = find_column(
    data.columns,
    ["MAGNETIC", "FIELD", "Y"]
)

mag_z = find_column(
    data.columns,
    ["MAGNETIC", "FIELD", "Z"]
)

print("\nMagnetometer:")
print("X:", mag_x)
print("Y:", mag_y)
print("Z:", mag_z)


# ============================================================
# 5. Sensor features
# ============================================================

sensor_columns = [
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
    mag_x,
    mag_y,
    mag_z
]

sensor_data = pd.DataFrame(index=data.index)

for col in sensor_columns:

    sensor_data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )


# ============================================================
# 6. Add velocity-context features
# ============================================================

print("\n[4] Building velocity-context features...")


# Target-derived velocity is used ONLY to construct
# teacher-forcing context during training.

# Important:
# We shift the target by one sample so the current target
# is never directly included in its own input.

sensor_data["previous_vn"] = (
    data["vn_true"]
    .shift(1)
)

sensor_data["previous_ve"] = (
    data["ve_true"]
    .shift(1)
)

sensor_data["previous_speed"] = np.sqrt(
    sensor_data["previous_vn"] ** 2
    +
    sensor_data["previous_ve"] ** 2
)

sensor_data["previous_dvn"] = (
    sensor_data["previous_vn"]
    .diff()
)

sensor_data["previous_dve"] = (
    sensor_data["previous_ve"]
    .diff()
)

sensor_data["previous_dspeed"] = (
    sensor_data["previous_speed"]
    .diff()
)


# ------------------------------------------------------------
# Fill missing values
# ------------------------------------------------------------

sensor_data = sensor_data.replace(
    [np.inf, -np.inf],
    np.nan
)

sensor_data = (
    sensor_data
    .ffill()
    .bfill()
    .fillna(0)
)


# ============================================================
# 7. Normalize using training region only
# ============================================================

n = len(data)

train_end = int(
    n * TRAIN_RATIO
)

print("Train end:", train_end)

feature_values = sensor_data.to_numpy(
    dtype=np.float32
)

feature_mean = feature_values[
    :train_end
].mean(axis=0)

feature_std = feature_values[
    :train_end
].std(axis=0)

feature_std[
    feature_std < 1e-6
] = 1.0

feature_values = (
    feature_values - feature_mean
) / feature_std


# ============================================================
# 8. Build sequences
# ============================================================

print("\n[5] Building temporal sequences...")

vn = data["vn_true"].to_numpy(
    dtype=np.float32
)

ve = data["ve_true"].to_numpy(
    dtype=np.float32
)

X_sequences = []
Y_targets = []
indices = []

for i in range(
    SEQ_LEN - 1,
    n
):

    start = i - SEQ_LEN + 1
    end = i + 1

    X_sequences.append(
        feature_values[start:end]
    )

    Y_targets.append(
        [
            vn[i],
            ve[i]
        ]
    )

    indices.append(i)


X_sequences = np.asarray(
    X_sequences,
    dtype=np.float32
)

Y_targets = np.asarray(
    Y_targets,
    dtype=np.float32
)

indices = np.asarray(
    indices,
    dtype=np.int64
)

print(
    "Sequence shape:",
    X_sequences.shape
)

print(
    "Target shape:",
    Y_targets.shape
)


# ============================================================
# 9. Chronological split
# ============================================================

train_mask = indices < train_end
test_mask = indices >= train_end

X_train = X_sequences[
    train_mask
]

Y_train = Y_targets[
    train_mask
]

X_test = X_sequences[
    test_mask
]

Y_test = Y_targets[
    test_mask
]

test_indices = indices[
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
# 10. PyTorch datasets
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(Y_train)
)

test_dataset = TensorDataset(
    torch.tensor(X_test),
    torch.tensor(Y_test)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# 11. GRU model
# ============================================================

class VelocityGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.head = nn.Sequential(

            nn.Linear(
                hidden_size,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                2
            )
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last = output[:, -1, :]

        return self.head(last)


model = VelocityGRU(
    input_size=X_train.shape[2]
).to(DEVICE)


print("\n[6] Model architecture:")
print(model)


# ============================================================
# 12. Training
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("\n[7] Training...")


for epoch in range(
    EPOCHS
):

    model.train()

    running_loss = 0.0

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

        running_loss += (
            loss.item()
            *
            len(batch_x)
        )

    epoch_loss = (
        running_loss
        /
        len(train_dataset)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"Loss: {epoch_loss:.6f}"
    )


# ============================================================
# 13. Save model
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_FILE
)

print(
    "\nModel saved:",
    os.path.abspath(MODEL_FILE)
)


# ============================================================
# 14. Test prediction
# ============================================================

print("\n[8] Generating test predictions...")

model.eval()

predictions = []

with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        BATCH_SIZE
    ):

        batch = torch.tensor(
            X_test[
                start:
                start + BATCH_SIZE
            ]
        ).to(DEVICE)

        pred = model(
            batch
        )

        predictions.append(
            pred.cpu().numpy()
        )


predictions = np.vstack(
    predictions
)

vn_pred = predictions[:, 0]
ve_pred = predictions[:, 1]

vn_true = Y_test[:, 0]
ve_true = Y_test[:, 1]


# ============================================================
# 15. Metrics
# ============================================================

def rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


north_mae = mean_absolute_error(
    vn_true,
    vn_pred
)

north_rmse = rmse(
    vn_true,
    vn_pred
)

north_r2 = r2_score(
    vn_true,
    vn_pred
)

east_mae = mean_absolute_error(
    ve_true,
    ve_pred
)

east_rmse = rmse(
    ve_true,
    ve_pred
)

east_r2 = r2_score(
    ve_true,
    ve_pred
)

true_speed = np.sqrt(
    vn_true ** 2 +
    ve_true ** 2
)

pred_speed = np.sqrt(
    vn_pred ** 2 +
    ve_pred ** 2
)

speed_mae = mean_absolute_error(
    true_speed,
    pred_speed
)


# ============================================================
# 16. Print results
# ============================================================

print("\n")
print("=" * 80)
print("V10.21 TEST RESULTS")
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


# ============================================================
# 17. Save predictions
# ============================================================

test_timestamps = data[
    "timestamp"
].iloc[
    test_indices
].reset_index(drop=True)

output = pd.DataFrame({

    "timestamp":
        test_timestamps,

    "vn_true_mps":
        vn_true,

    "ve_true_mps":
        ve_true,

    "vn_ai_mps":
        vn_pred,

    "ve_ai_mps":
        ve_pred,

    "true_speed_mps":
        true_speed,

    "ai_speed_mps":
        pred_speed
})

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nPredictions saved:",
    os.path.abspath(OUTPUT_FILE)
)


# ============================================================
# 18. Final
# ============================================================

print("\n")
print("=" * 80)
print("V10.21 COMPLETE")
print("=" * 80)

print(
    "This experiment evaluates whether adding "
    "previous-velocity context improves temporal "
    "velocity prediction."
)