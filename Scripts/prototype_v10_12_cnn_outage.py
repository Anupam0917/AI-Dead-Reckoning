import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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
    "prototype_v10_11_temporal_cnn.pt"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_12_cnn_outage.csv"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW = 20

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 75)
print("V10.12 TEMPORAL CNN GNSS OUTAGE TEST")
print("=" * 75)

print("\nDevice:", device)


# ============================================================
# CNN MODEL
# ============================================================

class TemporalCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                6,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
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


    def forward(self, x):

        # [batch, time, sensors]
        # -> [batch, sensors, time]

        x = x.permute(
            0,
            2,
            1
        )

        x = self.network(x)

        x = self.head(x)

        return x


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

print("\n[1] Loading trained CNN...")

# IMPORTANT:
# PyTorch 2.6+ defaults to weights_only=True.
# Our checkpoint contains NumPy arrays as well as
# model weights, so explicitly use weights_only=False.
#
# This is appropriate because this checkpoint was
# created locally by our own V10.11 training script.

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)


model = TemporalCNN().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


sensor_mean = np.asarray(
    checkpoint["sensor_mean"],
    dtype=np.float32
)

sensor_std = np.asarray(
    checkpoint["sensor_std"],
    dtype=np.float32
)

target_mean = np.asarray(
    checkpoint["target_mean"],
    dtype=np.float32
)

target_std = np.asarray(
    checkpoint["target_std"],
    dtype=np.float32
)

sensor_columns = checkpoint[
    "sensor_columns"
]


print("Model loaded successfully.")

print(
    "Sensors:",
    sensor_columns
)


# ============================================================
# 2. LOAD RAW SENSOR DATA
# ============================================================

print("\n[2] Loading raw sensor data...")

raw = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

raw.columns = raw.columns.str.strip()


# ============================================================
# 3. PARSE TIMESTAMP
# ============================================================

raw["timestamp"] = pd.to_datetime(
    raw[
        "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
    ],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)


# ============================================================
# 4. FIND MAGNETOMETER COLUMNS
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


if mag_x is None or mag_y is None or mag_z is None:

    raise ValueError(
        "Could not find magnetometer columns."
    )


print(
    "\nMagnetometer columns:"
)

print("X:", mag_x)
print("Y:", mag_y)
print("Z:", mag_z)


# ============================================================
# 5. NUMERIC CONVERSION
# ============================================================

def numeric(column):

    return pd.to_numeric(
        raw[column],
        errors="coerce"
    )


# ============================================================
# 6. CREATE SENSOR DATAFRAME
# ============================================================

print("\n[3] Preparing sensors...")


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
# 7. LOAD VELOCITY TARGETS
# ============================================================

print("\n[4] Loading velocity targets...")

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
# 8. MATCH SENSOR DATA WITH TARGET
# ============================================================

print("\n[5] Matching sensors and targets...")


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


# Remove invalid rows

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


data = data.dropna(
    subset=
    sensor_columns
    +
    [
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
# 9. EXTRACT GNSS OUTAGE
# ============================================================

print("\n[6] Extracting outage...")


outage = data[
    (data["time_s"] >= OUTAGE_START)
    &
    (data["time_s"] <= OUTAGE_END)
].copy()


outage = outage.reset_index(
    drop=True
)


if len(outage) == 0:

    raise ValueError(
        "No data found inside the requested outage."
    )


print(
    "Outage samples:",
    len(outage)
)

print(
    "Outage start:",
    outage["time_s"].iloc[0]
)

print(
    "Outage end:",
    outage["time_s"].iloc[-1]
)


# ============================================================
# 10. NORMALIZE SENSOR DATA
# ============================================================

X = outage[
    sensor_columns
].values.astype(
    np.float32
)


X = (
    X - sensor_mean
) / sensor_std


# ============================================================
# 11. BUILD 20-SAMPLE TEMPORAL WINDOWS
# ============================================================

print("\n[7] Building CNN windows...")


windows = []

valid_indices = []


for i in range(
    WINDOW - 1,
    len(X)
):

    start = (
        i - WINDOW + 1
    )

    windows.append(
        X[start:i + 1]
    )

    valid_indices.append(i)


windows = np.asarray(
    windows,
    dtype=np.float32
)


print(
    "CNN input shape:",
    windows.shape
)


# ============================================================
# 12. CNN INFERENCE
# ============================================================

print("\n[8] Running CNN prediction...")


input_tensor = torch.tensor(
    windows,
    dtype=torch.float32
).to(device)


with torch.no_grad():

    normalized_prediction = (
        model(
            input_tensor
        )
        .cpu()
        .numpy()
    )


# Convert normalized output
# back to m/s

prediction = (
    normalized_prediction
    *
    target_std
    +
    target_mean
)


# ============================================================
# 13. ALIGN TARGETS
# ============================================================

indices = np.asarray(
    valid_indices
)


true_vn = outage[
    "vn_mps"
].values[
    indices
]


true_ve = outage[
    "ve_mps"
].values[
    indices
]


times = outage[
    "time_s"
].values[
    indices
]


vn_pred = prediction[:, 0]

ve_pred = prediction[:, 1]


# ============================================================
# 14. VELOCITY METRICS
# ============================================================

vn_mae = np.mean(
    np.abs(
        vn_pred - true_vn
    )
)


ve_mae = np.mean(
    np.abs(
        ve_pred - true_ve
    )
)


vn_rmse = np.sqrt(
    np.mean(
        (vn_pred - true_vn) ** 2
    )
)


ve_rmse = np.sqrt(
    np.mean(
        (ve_pred - true_ve) ** 2
    )
)


print("\n")
print("=" * 75)
print("VELOCITY PERFORMANCE")
print("=" * 75)


print(
    "North MAE:",
    round(vn_mae, 4),
    "m/s"
)


print(
    "North RMSE:",
    round(vn_rmse, 4),
    "m/s"
)


print(
    "East MAE:",
    round(ve_mae, 4),
    "m/s"
)


print(
    "East RMSE:",
    round(ve_rmse, 4),
    "m/s"
)


# ============================================================
# 15. TIME STEP
# ============================================================

dt = np.diff(
    times
)


dt = np.clip(
    dt,
    0.001,
    0.5
)


# ============================================================
# 16. TRUE TRAJECTORY
# ============================================================

true_n = np.zeros(
    len(times)
)

true_e = np.zeros(
    len(times)
)


for i in range(
    1,
    len(times)
):

    true_n[i] = (
        true_n[i - 1]
        +
        true_vn[i - 1]
        *
        dt[i - 1]
    )

    true_e[i] = (
        true_e[i - 1]
        +
        true_ve[i - 1]
        *
        dt[i - 1]
    )


# ============================================================
# 17. CNN DEAD RECKONING
# ============================================================

cnn_n = np.zeros(
    len(times)
)

cnn_e = np.zeros(
    len(times)
)


for i in range(
    1,
    len(times)
):

    cnn_n[i] = (
        cnn_n[i - 1]
        +
        vn_pred[i - 1]
        *
        dt[i - 1]
    )

    cnn_e[i] = (
        cnn_e[i - 1]
        +
        ve_pred[i - 1]
        *
        dt[i - 1]
    )


# ============================================================
# 18. POSITION ERROR
# ============================================================

position_error = np.sqrt(

    (cnn_n - true_n) ** 2
    +
    (cnn_e - true_e) ** 2

)


mean_error = np.mean(
    position_error
)


final_error = position_error[-1]


max_error = np.max(
    position_error
)


# ============================================================
# 19. DISTANCE
# ============================================================

true_distance = np.sum(

    np.sqrt(

        np.diff(true_n) ** 2
        +
        np.diff(true_e) ** 2

    )

)


cnn_distance = np.sum(

    np.sqrt(

        np.diff(cnn_n) ** 2
        +
        np.diff(cnn_e) ** 2

    )

)


drift_percent = (

    abs(
        cnn_distance
        -
        true_distance
    )

    /

    true_distance

    *

    100

)


# ============================================================
# 20. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("CNN DEAD-RECKONING PERFORMANCE")
print("=" * 75)


print(
    "Mean position error:",
    round(
        mean_error,
        3
    ),
    "m"
)


print(
    "Final position error:",
    round(
        final_error,
        3
    ),
    "m"
)


print(
    "Maximum position error:",
    round(
        max_error,
        3
    ),
    "m"
)


print(
    "Reference distance:",
    round(
        true_distance,
        3
    ),
    "m"
)


print(
    "CNN distance:",
    round(
        cnn_distance,
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
# 21. SAVE DETAILED RESULTS
# ============================================================

result = pd.DataFrame({

    "time_s":
        times,

    "vn_true_mps":
        true_vn,

    "ve_true_mps":
        true_ve,

    "vn_cnn_mps":
        vn_pred,

    "ve_cnn_mps":
        ve_pred,

    "true_north_m":
        true_n,

    "true_east_m":
        true_e,

    "cnn_north_m":
        cnn_n,

    "cnn_east_m":
        cnn_e,

    "position_error_m":
        position_error

})


result.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(
    OUTPUT_FILE
)


print("\n")
print("=" * 75)
print("V10.12 COMPLETE")
print("=" * 75)