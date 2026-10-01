import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# V10.19
# RF + GRU VELOCITY ENSEMBLE
#
# RF:
#   GYRO + MAG temporal/statistical features
#
# GRU:
#   20-sample temporal sequence
#
# Ensemble:
#   V = alpha * RF + (1-alpha) * GRU
#
# Alpha is selected using ONLY a validation section
# inside the original training region.
#
# The 12 outage windows remain completely untouched
# during alpha selection.
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

GRU_MODEL_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_18_temporal_gru.pt"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_19_rf_gru_ensemble.csv"
)

OUTAGE_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v10_19_ensemble_outage_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20

TRAIN_RATIO = 0.70

# Validation is the final 20% of the training region.
VALIDATION_RATIO = 0.20

N_ESTIMATORS = 150

MAX_DEPTH = 20

MIN_SAMPLES_LEAF = 3

RANDOM_STATE = 42

BATCH_SIZE = 512

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Candidate ensemble weights.
# alpha = 1.0 -> RF only
# alpha = 0.0 -> GRU only
ALPHAS = np.arange(
    0.0,
    1.01,
    0.1
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("V10.19 RF + GRU ENSEMBLE")
print("=" * 80)

print(
    "\nDevice:",
    DEVICE
)

print(
    "Candidate alpha values:",
    ALPHAS
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
# 2. TIMESTAMP
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


print(
    "\nMagnetometer:"
)

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
# 6. LOAD TARGETS
# ============================================================

print(
    "\n[2] Loading velocity targets..."
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
    "\n[3] Synchronizing data..."
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
# 8. CLEAN
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
# 9. GLOBAL CHRONOLOGICAL SPLIT
# ============================================================

n = len(data)

train_end = int(
    n * TRAIN_RATIO
)

validation_start = int(
    train_end
    *
    (1.0 - VALIDATION_RATIO)
)


print(
    "\nTotal:",
    n
)

print(
    "Train end:",
    train_end
)

print(
    "Validation start:",
    validation_start
)

print(
    "\nTraining:",
    round(
        data["time_s"].iloc[0],
        2
    ),
    "to",
    round(
        data["time_s"].iloc[validation_start - 1],
        2
    )
)

print(
    "Validation:",
    round(
        data["time_s"].iloc[validation_start],
        2
    ),
    "to",
    round(
        data["time_s"].iloc[train_end - 1],
        2
    )
)

print(
    "Final test:",
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
# 10. RF FEATURES
# ============================================================

print(
    "\n[4] Building RF features..."
)


rf_features = pd.DataFrame(
    index=data.index
)


for column in feature_columns:

    rf_features[column] = data[
        column
    ]


    for window in [5, 10, 20]:

        rf_features[
            f"{column}_mean_{window}"
        ] = (

            data[column]

            .rolling(
                window,
                min_periods=1
            )

            .mean()

        )


        rf_features[
            f"{column}_std_{window}"
        ] = (

            data[column]

            .rolling(
                window,
                min_periods=1
            )

            .std()

            .fillna(0)

        )


rf_features[
    "mag_magnitude"
] = np.sqrt(

    data["mag_x"] ** 2
    +
    data["mag_y"] ** 2
    +
    data["mag_z"] ** 2

)


rf_features[
    "gyro_magnitude"
] = np.sqrt(

    data["gyro_yaw"] ** 2
    +
    data["gyro_pitch"] ** 2
    +
    data["gyro_roll"] ** 2

)


rf_features = rf_features.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)


X_rf = rf_features.to_numpy(
    dtype=np.float32
)


y_north = data[
    "vn_mps"
].to_numpy(
    dtype=np.float32
)


y_east = data[
    "ve_mps"
].to_numpy(
    dtype=np.float32
)


# ============================================================
# 11. TRAIN RF
# ============================================================

print(
    "\n[5] Training RF North..."
)

rf_north = RandomForestRegressor(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    random_state=RANDOM_STATE,

    n_jobs=-1

)

rf_north.fit(

    X_rf[:train_end],

    y_north[:train_end]

)


print(
    "RF North complete."
)


print(
    "\n[6] Training RF East..."
)

rf_east = RandomForestRegressor(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    random_state=RANDOM_STATE,

    n_jobs=-1

)

rf_east.fit(

    X_rf[:train_end],

    y_east[:train_end]

)


print(
    "RF East complete."
)


# ============================================================
# 12. RF PREDICTIONS
# ============================================================

print(
    "\n[7] Generating RF predictions..."
)

rf_vn = rf_north.predict(
    X_rf
)

rf_ve = rf_east.predict(
    X_rf
)


# ============================================================
# 13. LOAD GRU CHECKPOINT
# ============================================================

print(
    "\n[8] Loading V10.18 GRU..."
)


checkpoint = torch.load(

    GRU_MODEL_FILE,

    map_location=DEVICE,

    weights_only=False

)


gru_mean = np.asarray(
    checkpoint[
        "feature_mean"
    ],
    dtype=np.float32
)


gru_std = np.asarray(
    checkpoint[
        "feature_std"
    ],
    dtype=np.float32
)


gru_std[
    gru_std < 1e-6
] = 1.0


gru_feature_columns = checkpoint[
    "feature_columns"
]


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

        output, _ = self.gru(
            x
        )

        last_state = output[:, -1, :]

        return self.head(
            last_state
        )


# ============================================================
# 15. CREATE GRU
# ============================================================

gru = TemporalGRU(

    input_size=6,

    hidden_size=checkpoint[
        "hidden_size"
    ],

    num_layers=checkpoint[
        "num_layers"
    ],

    dropout=0.2

).to(
    DEVICE
)


gru.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


gru.eval()


# ============================================================
# 16. CREATE GRU SEQUENCES
# ============================================================

print(
    "\n[9] Generating GRU predictions..."
)


sensor_array = data[
    gru_feature_columns
].to_numpy(
    dtype=np.float32
)


sensor_normalized = (

    sensor_array
    -
    gru_mean

) / gru_std


test_indices = np.arange(
    train_end,
    n
)


gru_predictions = np.zeros(
    (
        n,
        2
    ),
    dtype=np.float32
)


# We need the previous 19 samples for
# every test prediction.

for start in range(
    train_end,
    n,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        n
    )


    batch_sequences = []


    for i in range(
        start,
        end
    ):

        sequence_start = (
            i
            -
            SEQUENCE_LENGTH
            +
            1
        )


        sequence = sensor_normalized[
            sequence_start:
            i + 1
        ]


        batch_sequences.append(
            sequence
        )


    batch_array = np.asarray(
        batch_sequences,
        dtype=np.float32
    )


    batch_tensor = torch.from_numpy(
        batch_array
    ).to(
        DEVICE
    )


    with torch.no_grad():

        batch_prediction = gru(
            batch_tensor
        )


    gru_predictions[
        start:end
    ] = batch_prediction.cpu().numpy()


gru_vn = gru_predictions[
    :,
    0
]

gru_ve = gru_predictions[
    :,
    1
]


# ============================================================
# 17. SELECT ALPHA USING VALIDATION ONLY
# ============================================================

print(
    "\n[10] Selecting ensemble weight..."
)

print(
    "Validation region:"
)

print(
    round(
        data["time_s"].iloc[validation_start],
        2
    ),
    "to",
    round(
        data["time_s"].iloc[train_end - 1],
        2
    ),
    "seconds"
)


validation_results = []


for alpha in ALPHAS:

    # RF weight = alpha
    # GRU weight = 1-alpha

    blend_n = (

        alpha * rf_vn
        +
        (1.0 - alpha) * gru_vn

    )


    blend_e = (

        alpha * rf_ve
        +
        (1.0 - alpha) * gru_ve

    )


    # Only validation portion

    true_n = y_north[
        validation_start:train_end
    ]

    true_e = y_east[
        validation_start:train_end
    ]

    pred_n = blend_n[
        validation_start:train_end
    ]

    pred_e = blend_e[
        validation_start:train_end
    ]


    north_mae = np.mean(
        np.abs(
            pred_n
            -
            true_n
        )
    )


    east_mae = np.mean(
        np.abs(
            pred_e
            -
            true_e
        )
    )


    speed_true = np.sqrt(
        true_n ** 2
        +
        true_e ** 2
    )


    speed_pred = np.sqrt(
        pred_n ** 2
        +
        pred_e ** 2
    )


    speed_mae = np.mean(
        np.abs(
            speed_pred
            -
            speed_true
        )
    )


    combined_mae = (
        north_mae
        +
        east_mae
    ) / 2.0


    validation_results.append({

        "alpha":
            alpha,

        "north_mae_mps":
            north_mae,

        "east_mae_mps":
            east_mae,

        "speed_mae_mps":
            speed_mae,

        "combined_mae_mps":
            combined_mae

    })


validation_df = pd.DataFrame(
    validation_results
)


best_row = validation_df.loc[
    validation_df[
        "combined_mae_mps"
    ].idxmin()
]


best_alpha = float(
    best_row["alpha"]
)


print(
    "\nValidation results:"
)

print(
    validation_df.round(
        5
    ).to_string(
        index=False
    )
)


print(
    "\nSelected alpha:",
    best_alpha
)

print(
    "RF weight:",
    best_alpha
)

print(
    "GRU weight:",
    1.0 - best_alpha
)


# ============================================================
# 18. FINAL ENSEMBLE PREDICTIONS
# ============================================================

ensemble_vn = (

    best_alpha * rf_vn

    +

    (1.0 - best_alpha) * gru_vn

)


ensemble_ve = (

    best_alpha * rf_ve

    +

    (1.0 - best_alpha) * gru_ve

)


# ============================================================
# 19. HELPER: VELOCITY METRICS
# ============================================================

def velocity_metrics(
    true_n,
    true_e,
    pred_n,
    pred_e
):

    north_mae = np.mean(
        np.abs(
            pred_n
            -
            true_n
        )
    )


    east_mae = np.mean(
        np.abs(
            pred_e
            -
            true_e
        )
    )


    speed_true = np.sqrt(
        true_n ** 2
        +
        true_e ** 2
    )


    speed_pred = np.sqrt(
        pred_n ** 2
        +
        pred_e ** 2
    )


    speed_mae = np.mean(
        np.abs(
            speed_pred
            -
            speed_true
        )
    )


    return (
        north_mae,
        east_mae,
        speed_mae
    )


# ============================================================
# 20. FINAL TEST METRICS
# ============================================================

test_true_n = y_north[
    train_end:
]

test_true_e = y_east[
    train_end:
]


rf_test_n = rf_vn[
    train_end:
]

rf_test_e = rf_ve[
    train_end:
]


gru_test_n = gru_vn[
    train_end:
]

gru_test_e = gru_ve[
    train_end:
]


ens_test_n = ensemble_vn[
    train_end:
]

ens_test_e = ensemble_ve[
    train_end:
]


rf_metrics = velocity_metrics(

    test_true_n,

    test_true_e,

    rf_test_n,

    rf_test_e

)


gru_metrics = velocity_metrics(

    test_true_n,

    test_true_e,

    gru_test_n,

    gru_test_e

)


ens_metrics = velocity_metrics(

    test_true_n,

    test_true_e,

    ens_test_n,

    ens_test_e

)


print("\n")
print("=" * 100)
print("FINAL HELD-OUT TEST COMPARISON")
print("=" * 100)


comparison = pd.DataFrame({

    "model": [
        "RF",
        "GRU",
        "RF+GRU"
    ],

    "north_mae_mps": [
        rf_metrics[0],
        gru_metrics[0],
        ens_metrics[0]
    ],

    "east_mae_mps": [
        rf_metrics[1],
        gru_metrics[1],
        ens_metrics[1]
    ],

    "speed_mae_mps": [
        rf_metrics[2],
        gru_metrics[2],
        ens_metrics[2]
    ],

    "speed_mae_kmh": [
        rf_metrics[2] * 3.6,
        gru_metrics[2] * 3.6,
        ens_metrics[2] * 3.6
    ]

})


print(
    comparison.round(
        5
    ).to_string(
        index=False
    )
)


# ============================================================
# 21. BUILD OUTPUT DATAFRAME
# ============================================================

output = data[
    [
        "timestamp",
        "time_s",
        "vn_mps",
        "ve_mps"
    ]
].copy()


output["vn_rf"] = rf_vn

output["ve_rf"] = rf_ve

output["vn_gru"] = gru_vn

output["ve_gru"] = gru_ve

output["vn_ensemble"] = ensemble_vn

output["ve_ensemble"] = ensemble_ve


output["ensemble_north_error"] = (

    output["vn_ensemble"]
    -
    output["vn_mps"]

)


output["ensemble_east_error"] = (

    output["ve_ensemble"]
    -
    output["ve_mps"]

)


output.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nFull ensemble predictions saved:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 22. MULTI-OUTAGE EVALUATION
# ============================================================

print(
    "\n[11] Running 12-outage benchmark..."
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


def evaluate_outage(
    start,
    duration,
    pred_n,
    pred_e
):

    end = (
        start
        +
        duration
    )


    subset = output[
        (
            output["time_s"]
            >= start
        )
        &
        (
            output["time_s"]
            <= end
        )
    ].copy()


    if len(subset) < 10:

        return None


    time = subset[
        "time_s"
    ].to_numpy(
        dtype=float
    )


    true_n = subset[
        "vn_mps"
    ].to_numpy(
        dtype=float
    )


    true_e = subset[
        "ve_mps"
    ].to_numpy(
        dtype=float
    )


    # Match predictions by timestamps

    indices = subset.index.to_numpy()

    ai_n = pred_n[
        indices
    ]

    ai_e = pred_e[
        indices
    ]


    true_n_pos = np.zeros(
        len(subset)
    )

    true_e_pos = np.zeros(
        len(subset)
    )

    ai_n_pos = np.zeros(
        len(subset)
    )

    ai_e_pos = np.zeros(
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


        true_n_pos[i] = (

            true_n_pos[i - 1]
            +
            true_n[i - 1] * dt

        )


        true_e_pos[i] = (

            true_e_pos[i - 1]
            +
            true_e[i - 1] * dt

        )


        ai_n_pos[i] = (

            ai_n_pos[i - 1]
            +
            ai_n[i - 1] * dt

        )


        ai_e_pos[i] = (

            ai_e_pos[i - 1]
            +
            ai_e[i - 1] * dt

        )


    position_error = np.sqrt(

        (
            ai_n_pos
            -
            true_n_pos
        ) ** 2

        +

        (
            ai_e_pos
            -
            true_e_pos
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


    reference_distance = np.sum(

        np.sqrt(

            np.diff(
                true_n_pos
            ) ** 2

            +

            np.diff(
                true_e_pos
            ) ** 2

        )

    )


    model_distance = np.sum(

        np.sqrt(

            np.diff(
                ai_n_pos
            ) ** 2

            +

            np.diff(
                ai_e_pos
            ) ** 2

        )

    )


    if reference_distance > 0:

        drift = (

            abs(
                model_distance
                -
                reference_distance
            )

            /
            reference_distance
            *
            100

        )

    else:

        drift = np.nan


    return {

        "start_s":
            start,

        "duration_s":
            duration,

        "samples":
            len(subset),

        "north_velocity_mae_mps":
            np.mean(
                np.abs(
                    ai_n
                    -
                    true_n
                )
            ),

        "east_velocity_mae_mps":
            np.mean(
                np.abs(
                    ai_e
                    -
                    true_e
                )
            ),

        "mean_position_error_m":
            mean_position_error,

        "final_position_error_m":
            final_position_error,

        "max_position_error_m":
            max_position_error,

        "reference_distance_m":
            reference_distance,

        "ensemble_distance_m":
            model_distance,

        "drift_percent":
            drift

    }


ensemble_results = []


for start, duration in OUTAGES:

    result = evaluate_outage(

        start,

        duration,

        ensemble_vn,

        ensemble_ve

    )


    if result is None:

        continue


    ensemble_results.append(
        result
    )


    print(
        f"\n{start:.0f}s / "
        f"{duration:.0f}s"
    )

    print(
        "Final error:",
        round(
            result[
                "final_position_error_m"
            ],
            3
        ),
        "m"
    )

    print(
        "Drift:",
        round(
            result[
                "drift_percent"
            ],
            3
        ),
        "%"
    )


# ============================================================
# 23. SAVE OUTAGE RESULTS
# ============================================================

outage_df = pd.DataFrame(
    ensemble_results
)


outage_df.to_csv(
    OUTAGE_FILE,
    index=False
)


# ============================================================
# 24. SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("V10.19 RF + GRU ENSEMBLE SUMMARY")
print("=" * 100)


print(
    "\nSelected alpha:",
    best_alpha
)

print(
    "RF weight:",
    best_alpha
)

print(
    "GRU weight:",
    1.0 - best_alpha
)


if len(outage_df) > 0:

    print(
        "\nValid outage windows:",
        len(outage_df)
    )


    print(
        "\nMean final error:",
        round(
            outage_df[
                "final_position_error_m"
            ].mean(),
            3
        ),
        "m"
    )


    print(
        "Median final error:",
        round(
            outage_df[
                "final_position_error_m"
            ].median(),
            3
        ),
        "m"
    )


    print(
        "Maximum final error:",
        round(
            outage_df[
                "final_position_error_m"
            ].max(),
            3
        ),
        "m"
    )


    print(
        "\nMean position error:",
        round(
            outage_df[
                "mean_position_error_m"
            ].mean(),
            3
        ),
        "m"
    )


    print(
        "Mean drift:",
        round(
            outage_df[
                "drift_percent"
            ].mean(),
            3
        ),
        "%"
    )


    print(
        "Median drift:",
        round(
            outage_df[
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
        outage_df.round(
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
# 25. COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("V10.19 COMPLETE")
print("=" * 80)

print(
    "\nRF, GRU and RF+GRU have now been evaluated."
)

print(
    "The 12 outage windows were not used to select alpha."
)