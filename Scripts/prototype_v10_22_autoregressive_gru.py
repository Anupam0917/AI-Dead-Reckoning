import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# V10.22
# AUTOREGRESSIVE TEMPORAL GRU
# ============================================================

RAW_FILE = (
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/M (Driver B)/S-M.csv"
)

TARGET_FILE = (
    "data/processed/prototype_v9_velocity_targets.csv"
)

MODEL_FILE = (
    "data/processed/prototype_v10_22_autoregressive_gru.pt"
)

OUTPUT_FILE = (
    "data/processed/prototype_v10_22_autoregressive_gru.csv"
)

OUTAGE_OUTPUT = (
    "data/processed/prototype_v10_22_outage_results.csv"
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

OUTAGES = [
    (7500, 30),
    (7500, 60),
    (7500, 120),

    (8500, 30),
    (8500, 60),
    (8500, 120),

    (9500, 30),
    (9500, 60),
    (9500, 120),

    (10000, 30),
    (10000, 60),
    (10000, 120),
]


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 80)
print("V10.22 AUTOREGRESSIVE TEMPORAL GRU")
print("=" * 80)

print("\nDevice:", DEVICE)


# ============================================================
# Helpers
# ============================================================

def find_column(columns, keywords):

    for col in columns:

        name = col.lower()

        if all(
            k.lower() in name
            for k in keywords
        ):
            return col

    return None


def rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


# ============================================================
# 1. Load dataset
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

targets = pd.read_csv(
    TARGET_FILE
)

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
# 5. Sensor matrix
# ============================================================

sensor_columns = [
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
    mag_x,
    mag_y,
    mag_z
]

sensor_data = pd.DataFrame(
    index=data.index
)

for col in sensor_columns:

    sensor_data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )


sensor_data = (
    sensor_data
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .ffill()
    .bfill()
    .fillna(0)
)


# ============================================================
# 6. Training feature construction
# ============================================================

print("\n[4] Building training sequences...")


# We use previous velocity only as a TRAINING context.
#
# During the actual outage benchmark below, this context
# is replaced recursively by the model's own prediction.
#
# Therefore the outage benchmark does NOT use true GNSS
# velocity inside the autoregressive loop.

vn = data["vn_true"].to_numpy(
    dtype=np.float32
)

ve = data["ve_true"].to_numpy(
    dtype=np.float32
)

speed = np.sqrt(
    vn ** 2 +
    ve ** 2
)

previous_vn = np.roll(
    vn,
    1
)

previous_ve = np.roll(
    ve,
    1
)

previous_speed = np.roll(
    speed,
    1
)

previous_dvn = np.zeros_like(vn)
previous_dve = np.zeros_like(ve)

previous_dvn[1:] = (
    vn[1:] -
    vn[:-1]
)

previous_dve[1:] = (
    ve[1:] -
    ve[:-1]
)

previous_dspeed = np.zeros_like(speed)

previous_dspeed[1:] = (
    speed[1:] -
    speed[:-1]
)

previous_vn[0] = 0
previous_ve[0] = 0
previous_speed[0] = 0


# ------------------------------------------------------------
# Combine sensors + previous velocity context
# ------------------------------------------------------------

feature_df = sensor_data.copy()

feature_df["previous_vn"] = previous_vn
feature_df["previous_ve"] = previous_ve
feature_df["previous_speed"] = previous_speed
feature_df["previous_dvn"] = previous_dvn
feature_df["previous_dve"] = previous_dve
feature_df["previous_dspeed"] = previous_dspeed

feature_df = (
    feature_df
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .ffill()
    .bfill()
    .fillna(0)
)


# ============================================================
# 7. Chronological split
# ============================================================

n = len(data)

train_end = int(
    n * TRAIN_RATIO
)

print(
    "Train end:",
    train_end
)

print(
    "Training:",
    data["timestamp"].iloc[0],
    "to",
    data["timestamp"].iloc[
        train_end - 1
    ]
)

print(
    "Testing:",
    data["timestamp"].iloc[
        train_end
    ],
    "to",
    data["timestamp"].iloc[-1]
)


# ============================================================
# 8. Normalize using training only
# ============================================================

features = feature_df.to_numpy(
    dtype=np.float32
)

feature_mean = features[
    :train_end
].mean(axis=0)

feature_std = features[
    :train_end
].std(axis=0)

feature_std[
    feature_std < 1e-6
] = 1.0

features = (
    features -
    feature_mean
) / feature_std


# ============================================================
# 9. Build sequences
# ============================================================

X = []
Y = []
indices = []

for i in range(
    SEQ_LEN - 1,
    n
):

    start = (
        i -
        SEQ_LEN +
        1
    )

    X.append(
        features[start:i + 1]
    )

    Y.append(
        [
            vn[i],
            ve[i]
        ]
    )

    indices.append(i)


X = np.asarray(
    X,
    dtype=np.float32
)

Y = np.asarray(
    Y,
    dtype=np.float32
)

indices = np.asarray(
    indices,
    dtype=np.int64
)

print(
    "\nSequence shape:",
    X.shape
)

print(
    "Target shape:",
    Y.shape
)


# ============================================================
# 10. Split
# ============================================================

train_mask = (
    indices <
    train_end
)

test_mask = (
    indices >=
    train_end
)

X_train = X[
    train_mask
]

Y_train = Y[
    train_mask
]

X_test = X[
    test_mask
]

Y_test = Y[
    test_mask
]

test_indices = indices[
    test_mask
]

print(
    "Training sequences:",
    len(X_train)
)

print(
    "Testing sequences:",
    len(X_test)
)


# ============================================================
# 11. GRU model
# ============================================================

class AutoRegressiveGRU(
    nn.Module
):

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


model = AutoRegressiveGRU(
    input_size=X_train.shape[2]
).to(DEVICE)


print("\n[5] Model:")
print(model)


# ============================================================
# 12. Train
# ============================================================

dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(Y_train)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("\n[6] Training...")


for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(
            DEVICE
        )

        batch_y = batch_y.to(
            DEVICE
        )

        optimizer.zero_grad()

        pred = model(
            batch_x
        )

        loss = criterion(
            pred,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            *
            len(batch_x)
        )

    epoch_loss = (
        total_loss /
        len(dataset)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"Loss: {epoch_loss:.6f}"
    )


# ============================================================
# 13. Save model + normalization
# ============================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "feature_mean":
        feature_mean,

    "feature_std":
        feature_std,

    "seq_len":
        SEQ_LEN,

    "input_size":
        X_train.shape[2]
}

torch.save(
    checkpoint,
    MODEL_FILE
)

print(
    "\nModel saved:",
    os.path.abspath(
        MODEL_FILE
    )
)


# ============================================================
# 14. Standard test prediction
# ============================================================

print("\n[7] Standard test evaluation...")

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

vn_test = Y_test[:, 0]
ve_test = Y_test[:, 1]

print("\nStandard test metrics:")

print(
    "North MAE:",
    f"{mean_absolute_error(vn_test, vn_pred):.4f}",
    "m/s"
)

print(
    "East MAE :",
    f"{mean_absolute_error(ve_test, ve_pred):.4f}",
    "m/s"
)

print(
    "North R2:",
    f"{r2_score(vn_test, vn_pred):.4f}"
)

print(
    "East R2:",
    f"{r2_score(ve_test, ve_pred):.4f}"
)


# ============================================================
# 15. Autoregressive prediction helper
# ============================================================

def autoregressive_predict(
    start_index,
    end_index,
    initial_vn,
    initial_ve
):

    """
    Predict velocity recursively.

    IMPORTANT:
    Once the outage begins, the true velocity is NEVER
    inserted into the prediction loop.

    The model's own previous prediction becomes the
    velocity-context input for the next prediction.
    """

    history = []

    # --------------------------------------------------------
    # Build initial sensor/history window
    # --------------------------------------------------------

    start_history = (
        start_index -
        SEQ_LEN +
        1
    )

    if start_history < 0:

        raise ValueError(
            "Not enough history."
        )

    previous_pred_vn = float(
        initial_vn
    )

    previous_pred_ve = float(
        initial_ve
    )

    previous_pred_speed = np.sqrt(
        previous_pred_vn ** 2 +
        previous_pred_ve ** 2
    )

    previous_pred_dvn = 0.0
    previous_pred_dve = 0.0
    previous_pred_dspeed = 0.0

    predictions = []

    for current_index in range(
        start_index,
        end_index + 1
    ):

        sequence_rows = []

        # ----------------------------------------------------
        # For every point in the sequence, construct
        # sensor + velocity context.
        #
        # For rows before outage:
        # use the actual target velocity.
        #
        # For the current/outage rows:
        # use recursively predicted velocity.
        # ----------------------------------------------------

        for j in range(
            current_index -
            SEQ_LEN +
            1,
            current_index + 1
        ):

            sensor_values = (
                sensor_data
                .iloc[j]
                .to_numpy(
                    dtype=np.float32
                )
            )

            if j < start_index:

                context_vn = vn[j]
                context_ve = ve[j]

                if j == 0:

                    context_dvn = 0.0
                    context_dve = 0.0
                    context_dspeed = 0.0

                else:

                    context_dvn = (
                        vn[j] -
                        vn[j - 1]
                    )

                    context_dve = (
                        ve[j] -
                        ve[j - 1]
                    )

                    old_speed = np.sqrt(
                        vn[j - 1] ** 2 +
                        ve[j - 1] ** 2
                    )

                    new_speed = np.sqrt(
                        vn[j] ** 2 +
                        ve[j] ** 2
                    )

                    context_dspeed = (
                        new_speed -
                        old_speed
                    )

            else:

                context_vn = (
                    previous_pred_vn
                )

                context_ve = (
                    previous_pred_ve
                )

                context_dvn = (
                    previous_pred_dvn
                )

                context_dve = (
                    previous_pred_dve
                )

                context_dspeed = (
                    previous_pred_dspeed
                )

            context_speed = np.sqrt(
                context_vn ** 2 +
                context_ve ** 2
            )

            row = np.concatenate(
                [
                    sensor_values,
                    np.array(
                        [
                            context_vn,
                            context_ve,
                            context_speed,
                            context_dvn,
                            context_dve,
                            context_dspeed
                        ],
                        dtype=np.float32
                    )
                ]
            )

            sequence_rows.append(
                row
            )

        sequence = np.asarray(
            sequence_rows,
            dtype=np.float32
        )

        sequence = (
            sequence -
            feature_mean
        ) / feature_std

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(
            DEVICE
        )

        with torch.no_grad():

            prediction = model(
                tensor
            ).cpu().numpy()[0]

        predicted_vn = float(
            prediction[0]
        )

        predicted_ve = float(
            prediction[1]
        )

        predicted_speed = np.sqrt(
            predicted_vn ** 2 +
            predicted_ve ** 2
        )

        predicted_dvn = (
            predicted_vn -
            previous_pred_vn
        )

        predicted_dve = (
            predicted_ve -
            previous_pred_ve
        )

        predicted_dspeed = (
            predicted_speed -
            previous_pred_speed
        )

        predictions.append(
            [
                current_index,
                predicted_vn,
                predicted_ve
            ]
        )

        previous_pred_vn = predicted_vn
        previous_pred_ve = predicted_ve
        previous_pred_speed = predicted_speed
        previous_pred_dvn = predicted_dvn
        previous_pred_dve = predicted_dve
        previous_pred_dspeed = predicted_dspeed

    return np.asarray(
        predictions
    )


# ============================================================
# 16. Outage benchmark
# ============================================================

print("\n")
print("=" * 80)
print("12-OUTAGE AUTOREGRESSIVE BENCHMARK")
print("=" * 80)


timestamps = (
    data["timestamp"]
    .to_numpy()
)

time_seconds = (
    (
        timestamps -
        timestamps[0]
    )
    /
    np.timedelta64(
        1,
        "s"
    )
).astype(float)


# ------------------------------------------------------------
# Convert time to nearest index
# ------------------------------------------------------------

def find_index(time_value):

    return int(
        np.argmin(
            np.abs(
                time_seconds -
                time_value
            )
        )
    )


results = []

all_output_rows = []


for outage_start, duration in OUTAGES:

    start_idx = find_index(
        outage_start
    )

    end_idx = find_index(
        outage_start +
        duration
    )

    print()
    print(
        f"{outage_start}s / "
        f"{duration}s"
    )

    # --------------------------------------------------------
    # Position origin at GNSS position at outage start.
    #
    # Reference displacement comes from the canonical
    # target-derived GNSS velocity integration.
    # --------------------------------------------------------

    ref_n = 0.0
    ref_e = 0.0

    nav_n = 0.0
    nav_e = 0.0

    previous_timestamp = (
        timestamps[start_idx]
    )

    # --------------------------------------------------------
    # Initial velocity is the LAST AVAILABLE GNSS velocity
    # before outage.
    # --------------------------------------------------------

    initial_vn = vn[
        start_idx - 1
    ]

    initial_ve = ve[
        start_idx - 1
    ]

    predictions = (
        autoregressive_predict(
            start_idx,
            end_idx,
            initial_vn,
            initial_ve
        )
    )

    pred_dict = {
        int(row[0]):
            (
                float(row[1]),
                float(row[2])
            )
        for row in predictions
    }

    errors = []

    model_distance = 0.0
    reference_distance = 0.0

    previous_nav_n = 0.0
    previous_nav_e = 0.0

    previous_ref_n = 0.0
    previous_ref_e = 0.0

    for current_idx in range(
        start_idx,
        end_idx + 1
    ):

        if current_idx == start_idx:

            dt = (
                (
                    timestamps[
                        current_idx
                    ]
                    -
                    timestamps[
                        current_idx - 1
                    ]
                )
                /
                np.timedelta64(
                    1,
                    "s"
                )
            )

        else:

            dt = (
                (
                    timestamps[
                        current_idx
                    ]
                    -
                    timestamps[
                        current_idx - 1
                    ]
                )
                /
                np.timedelta64(
                    1,
                    "s"
                )
            )

        dt = float(dt)

        predicted_vn, predicted_ve = (
            pred_dict[
                current_idx
            ]
        )

        true_vn = vn[
            current_idx
        ]

        true_ve = ve[
            current_idx
        ]

        # ----------------------------------------------------
        # AI navigation
        # ----------------------------------------------------

        nav_n += (
            predicted_vn *
            dt
        )

        nav_e += (
            predicted_ve *
            dt
        )

        # ----------------------------------------------------
        # Reference
        # ----------------------------------------------------

        ref_n += (
            true_vn *
            dt
        )

        ref_e += (
            true_ve *
            dt
        )

        error = np.sqrt(
            (
                nav_n -
                ref_n
            ) ** 2
            +
            (
                nav_e -
                ref_e
            ) ** 2
        )

        errors.append(
            error
        )

        step_model = np.sqrt(
            (
                nav_n -
                previous_nav_n
            ) ** 2
            +
            (
                nav_e -
                previous_nav_e
            ) ** 2
        )

        step_reference = np.sqrt(
            (
                ref_n -
                previous_ref_n
            ) ** 2
            +
            (
                ref_e -
                previous_ref_e
            ) ** 2
        )

        model_distance += step_model
        reference_distance += (
            step_reference
        )

        previous_nav_n = nav_n
        previous_nav_e = nav_e

        previous_ref_n = ref_n
        previous_ref_e = ref_e

        all_output_rows.append(
            {
                "outage_start_s":
                    outage_start,

                "duration_s":
                    duration,

                "timestamp":
                    timestamps[
                        current_idx
                    ],

                "time_s":
                    time_seconds[
                        current_idx
                    ],

                "vn_true_mps":
                    true_vn,

                "ve_true_mps":
                    true_ve,

                "vn_ai_mps":
                    predicted_vn,

                "ve_ai_mps":
                    predicted_ve,

                "north_m":
                    nav_n,

                "east_m":
                    nav_e,

                "reference_north_m":
                    ref_n,

                "reference_east_m":
                    ref_e,

                "position_error_m":
                    error
            }
        )

    final_error = errors[-1]

    mean_error = np.mean(
        errors
    )

    max_error = np.max(
        errors
    )

    drift = (
        final_error /
        reference_distance *
        100.0
        if reference_distance > 0
        else np.nan
    )

    print(
        f"Final error: "
        f"{final_error:.3f} m"
    )

    print(
        f"Mean error: "
        f"{mean_error:.3f} m"
    )

    print(
        f"Max error: "
        f"{max_error:.3f} m"
    )

    print(
        f"Reference distance: "
        f"{reference_distance:.3f} m"
    )

    print(
        f"Model distance: "
        f"{model_distance:.3f} m"
    )

    print(
        f"Drift: "
        f"{drift:.3f} %"
    )

    results.append(
        {
            "start_s":
                outage_start,

            "duration_s":
                duration,

            "samples":
                end_idx -
                start_idx +
                1,

            "mean_position_error_m":
                mean_error,

            "final_position_error_m":
                final_error,

            "max_position_error_m":
                max_error,

            "reference_distance_m":
                reference_distance,

            "model_distance_m":
                model_distance,

            "drift_percent":
                drift
        }
    )


# ============================================================
# 17. Save outage results
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTAGE_OUTPUT,
    index=False
)

prediction_df = pd.DataFrame(
    all_output_rows
)

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 18. Summary
# ============================================================

print("\n")
print("=" * 80)
print("V10.22 SUMMARY")
print("=" * 80)

print(
    "\nValid outage windows:",
    len(results_df)
)

print(
    "Mean final error:",
    f"{results_df['final_position_error_m'].mean():.3f} m"
)

print(
    "Median final error:",
    f"{results_df['final_position_error_m'].median():.3f} m"
)

print(
    "Maximum final error:",
    f"{results_df['final_position_error_m'].max():.3f} m"
)

print(
    "Mean position error:",
    f"{results_df['mean_position_error_m'].mean():.3f} m"
)

print(
    "Mean drift:",
    f"{results_df['drift_percent'].mean():.3f} %"
)

print(
    "Median drift:",
    f"{results_df['drift_percent'].median():.3f} %"
)

print("\n")
print(results_df.to_string(index=False))


print("\n")
print(
    "Outage results saved:",
    os.path.abspath(
        OUTAGE_OUTPUT
    )
)

print(
    "Detailed predictions saved:",
    os.path.abspath(
        OUTPUT_FILE
    )
)

print("\n")
print("=" * 80)
print("V10.22 COMPLETE")
print("=" * 80)

print(
    "The outage loop uses previous AI predictions, "
    "not true GNSS velocity."
)