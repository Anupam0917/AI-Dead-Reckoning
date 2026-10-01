"""
V10.2 - Phone Frame -> Navigation Frame Diagnostic

Goal:
Determine how the smartphone sensor X/Y axes relate to
the North/East navigation frame.

We estimate a 2D rotation:

    [North]   [ cos(theta)  -sin(theta)] [Phone X]
    [East ] = [ sin(theta)   cos(theta)] [Phone Y]

The rotation is learned ONLY from the pre-outage period.

GNSS is used only to construct the supervised reference velocity.

The 8500-8560 s outage is NOT used to estimate the rotation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    ROOT
    / "data"
    / "raw"
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
    / "M (Driver B)"
    / "S-M.csv"
)

TARGET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "prototype_v9_velocity_targets.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)

OUTPUTS = (
    ROOT
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUTS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

# Only use data BEFORE the outage for calibration.

CALIBRATION_END = OUTAGE_START


# ============================================================
# LOAD RAW DATA
# ============================================================

print("=" * 80)
print("V10.2 PHONE FRAME ALIGNMENT DIAGNOSTIC")
print("=" * 80)

print("\nLoading raw dataset...")

df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252",
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print(
    "Raw rows:",
    len(df)
)


# ============================================================
# TIMESTAMP
# ============================================================

TIME_COL = (
    "DATE (YYYY-MO-DD HH-MI-SS_SSS)"
)

df["timestamp"] = pd.to_datetime(
    df[TIME_COL],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce",
)

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

df["time_s"] = (
    (
        df["timestamp"]
        -
        df["timestamp"].iloc[0]
    )
    .dt.total_seconds()
)


# ============================================================
# SENSOR COLUMNS
# ============================================================

ACC_X = "ACCELEROMETER X (m/s²)"
ACC_Y = "ACCELEROMETER Y (m/s²)"
ACC_Z = "ACCELEROMETER Z (m/s²)"

GRAV_X = "GRAVITY X (m/s²)"
GRAV_Y = "GRAVITY Y (m/s²)"
GRAV_Z = "GRAVITY Z (m/s²)"


for column in [
    ACC_X,
    ACC_Y,
    ACC_Z,
    GRAV_X,
    GRAV_Y,
    GRAV_Z,
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# LOAD VELOCITY TARGETS
# ============================================================

print(
    "\nLoading GNSS-derived velocity targets..."
)

targets = pd.read_csv(
    TARGET_FILE
)

targets["timestamp"] = pd.to_datetime(
    targets["timestamp"],
    errors="coerce",
)


targets = targets[
    [
        "timestamp",
        "vn_mps",
        "ve_mps",
    ]
].copy()


targets = targets.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# MERGE
# ============================================================

df = pd.merge_asof(
    df.sort_values(
        "timestamp"
    ),
    targets,
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(
        milliseconds=60
    ),
)


df = df.dropna(
    subset=[
        ACC_X,
        ACC_Y,
        ACC_Z,
        GRAV_X,
        GRAV_Y,
        GRAV_Z,
        "vn_mps",
        "ve_mps",
    ]
).reset_index(
    drop=True
)


print(
    "Usable rows:",
    len(df)
)


# ============================================================
# LINEAR ACCELERATION
# ============================================================

df["linear_x"] = (
    df[ACC_X]
    -
    df[GRAV_X]
)

df["linear_y"] = (
    df[ACC_Y]
    -
    df[GRAV_Y]
)

df["linear_z"] = (
    df[ACC_Z]
    -
    df[GRAV_Z]
)


# ============================================================
# CALIBRATION DATA
# ============================================================

calibration = df[
    df["time_s"]
    <
    CALIBRATION_END
].copy()


print(
    "\nCalibration rows:",
    len(calibration)
)

print(
    "Calibration interval:",
    f"0 -> {CALIBRATION_END}s"
)


# ============================================================
# FUNCTION: ROTATE PHONE FRAME
# ============================================================

def rotate_xy(
    x,
    y,
    theta,
):

    c = np.cos(theta)
    s = np.sin(theta)

    north = (
        c * x
        -
        s * y
    )

    east = (
        s * x
        +
        c * y
    )

    return north, east


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    theta,
    x,
    y,
    target_n,
    target_e,
):

    pred_n, pred_e = rotate_xy(
        x,
        y,
        theta,
    )

    true_speed = np.sqrt(
        target_n ** 2
        +
        target_e ** 2
    )

    pred_speed = np.sqrt(
        pred_n ** 2
        +
        pred_e ** 2
    )

    north_mae = np.mean(
        np.abs(
            pred_n
            -
            target_n
        )
    )

    east_mae = np.mean(
        np.abs(
            pred_e
            -
            target_e
        )
    )

    speed_mae = np.mean(
        np.abs(
            pred_speed
            -
            true_speed
        )
    )

    north_rmse = np.sqrt(
        np.mean(
            (
                pred_n
                -
                target_n
            ) ** 2
        )
    )

    east_rmse = np.sqrt(
        np.mean(
            (
                pred_e
                -
                target_e
            ) ** 2
        )
    )

    return {
        "theta": theta,
        "north_mae": north_mae,
        "east_mae": east_mae,
        "speed_mae": speed_mae,
        "north_rmse": north_rmse,
        "east_rmse": east_rmse,
    }


# ============================================================
# PREPARE DATA
# ============================================================

x = calibration[
    "linear_x"
].to_numpy()

y = calibration[
    "linear_y"
].to_numpy()

target_n = calibration[
    "vn_mps"
].to_numpy()

target_e = calibration[
    "ve_mps"
].to_numpy()


# ============================================================
# NORMALIZE ACCELERATION
# ============================================================

# Acceleration and velocity are different physical units.
#
# We therefore do NOT compare raw acceleration directly
# against velocity.
#
# Instead we compare the direction of horizontal motion.

accel_norm = np.sqrt(
    x ** 2
    +
    y ** 2
)

velocity_norm = np.sqrt(
    target_n ** 2
    +
    target_e ** 2
)


valid = (
    accel_norm > 0.20
) & (
    velocity_norm > 0.50
)


x_valid = x[valid]
y_valid = y[valid]

target_n_valid = target_n[valid]
target_e_valid = target_e[valid]


# ============================================================
# SEARCH ROTATION
# ============================================================

print(
    "\nSearching phone -> navigation rotation..."
)

angles_deg = np.arange(
    -180.0,
    180.0,
    0.5,
)


rotation_results = []


for angle_deg in angles_deg:

    theta = np.radians(
        angle_deg
    )

    pred_n, pred_e = rotate_xy(
        x_valid,
        y_valid,
        theta,
    )


    # Direction only.

    pred_norm = np.sqrt(
        pred_n ** 2
        +
        pred_e ** 2
    )

    pred_norm = np.maximum(
        pred_norm,
        1e-9,
    )


    true_norm = np.sqrt(
        target_n_valid ** 2
        +
        target_e_valid ** 2
    )

    true_norm = np.maximum(
        true_norm,
        1e-9,
    )


    # Unit-vector direction error.

    pred_n_unit = (
        pred_n
        /
        pred_norm
    )

    pred_e_unit = (
        pred_e
        /
        pred_norm
    )

    true_n_unit = (
        target_n_valid
        /
        true_norm
    )

    true_e_unit = (
        target_e_valid
        /
        true_norm
    )


    cosine = (
        pred_n_unit
        *
        true_n_unit
        +
        pred_e_unit
        *
        true_e_unit
    )


    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )


    angular_error = np.degrees(
        np.arccos(
            cosine
        )
    )


    mean_angle_error = np.mean(
        angular_error
    )


    rotation_results.append(
        {
            "angle_deg":
                angle_deg,

            "direction_error_deg":
                mean_angle_error,
        }
    )


rotation_results = pd.DataFrame(
    rotation_results
)


# ============================================================
# BEST ANGLE
# ============================================================

best_row = (
    rotation_results
    .sort_values(
        "direction_error_deg"
    )
    .iloc[0]
)


best_angle = float(
    best_row["angle_deg"]
)


best_theta = np.radians(
    best_angle
)


print(
    "\nBest rotation:"
)

print(
    f"{best_angle:.2f} degrees"
)

print(
    "Mean direction error:"
)

print(
    f"{best_row['direction_error_deg']:.3f} degrees"
)


# ============================================================
# METRICS AT BEST ROTATION
# ============================================================

best_metrics = calculate_metrics(
    best_theta,
    x_valid,
    y_valid,
    target_n_valid,
    target_e_valid,
)


print(
    "\nBest rotation metrics:"
)

print(
    f"North MAE:"
    f" {best_metrics['north_mae']:.4f}"
    f" m/s"
)

print(
    f"East MAE:"
    f" {best_metrics['east_mae']:.4f}"
    f" m/s"
)

print(
    f"North RMSE:"
    f" {best_metrics['north_rmse']:.4f}"
    f" m/s"
)

print(
    f"East RMSE:"
    f" {best_metrics['east_rmse']:.4f}"
    f" m/s"
)


# ============================================================
# TEST COMMON ROTATIONS
# ============================================================

print(
    "\nCommon rotation diagnostics:"
)

for angle in [
    0,
    45,
    90,
    135,
    180,
    -45,
    -90,
    -135,
]:

    metrics = calculate_metrics(
        np.radians(angle),
        x_valid,
        y_valid,
        target_n_valid,
        target_e_valid,
    )

    print(
        f"{angle:>5}° -> "
        f"N MAE {metrics['north_mae']:.3f}, "
        f"E MAE {metrics['east_mae']:.3f}"
    )


# ============================================================
# APPLY BEST ROTATION TO COMPLETE DATASET
# ============================================================

all_x = df[
    "linear_x"
].to_numpy()

all_y = df[
    "linear_y"
].to_numpy()


rot_n, rot_e = rotate_xy(
    all_x,
    all_y,
    best_theta,
)


df["rotated_north_accel"] = (
    rot_n
)

df["rotated_east_accel"] = (
    rot_e
)


# ============================================================
# OUTAGE DATA
# ============================================================

outage = df[
    (
        df["time_s"]
        >= OUTAGE_START
    )
    &
    (
        df["time_s"]
        <= OUTAGE_END
    )
].copy()


print(
    "\nOutage samples:",
    len(outage)
)


# ============================================================
# OUTAGE CORRELATION
# ============================================================

out_n = outage[
    "rotated_north_accel"
].to_numpy()

out_e = outage[
    "rotated_east_accel"
].to_numpy()

out_true_n = outage[
    "vn_mps"
].to_numpy()

out_true_e = outage[
    "ve_mps"
].to_numpy()


# Correlation helper.

def safe_corr(a, b):

    if (
        np.std(a) < 1e-9
        or
        np.std(b) < 1e-9
    ):
        return np.nan

    return np.corrcoef(
        a,
        b,
    )[0, 1]


print(
    "\nOutage directional correlation:"
)

print(
    f"North:"
    f" {safe_corr(out_n, out_true_n):.4f}"
)

print(
    f"East:"
    f" {safe_corr(out_e, out_true_e):.4f}"
)


# ============================================================
# SAVE ROTATION RESULTS
# ============================================================

rotation_file = (
    OUTPUT_DIR
    /
    "prototype_v10_2_rotation_search.csv"
)

rotation_results.to_csv(
    rotation_file,
    index=False,
)


# ============================================================
# SAVE DATA
# ============================================================

output_columns = [
    "timestamp",
    "time_s",

    "linear_x",
    "linear_y",

    "rotated_north_accel",
    "rotated_east_accel",

    "vn_mps",
    "ve_mps",
]


output_file = (
    OUTPUT_DIR
    /
    "prototype_v10_2_frame_alignment.csv"
)

df[
    output_columns
].to_csv(
    output_file,
    index=False,
)


# ============================================================
# ROTATION SEARCH PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    rotation_results[
        "angle_deg"
    ],
    rotation_results[
        "direction_error_deg"
    ],
)

plt.axvline(
    best_angle,
    linestyle="--",
    label=(
        f"Best = "
        f"{best_angle:.1f}°"
    ),
)

plt.xlabel(
    "Phone → Navigation rotation (degrees)"
)

plt.ylabel(
    "Mean direction error (degrees)"
)

plt.title(
    "Phone Frame Alignment Search"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

rotation_plot = (
    OUTPUTS
    /
    "prototype_v10_2_rotation_search.png"
)

plt.savefig(
    rotation_plot,
    dpi=200,
)

plt.close()


# ============================================================
# OUTAGE VECTOR PLOT
# ============================================================

plt.figure(
    figsize=(12, 8)
)

# Normalize vectors so the plot shows direction
# rather than acceleration magnitude.

sample_step = 10

out_n_norm = (
    out_n
    /
    np.maximum(
        np.sqrt(
            out_n ** 2
            +
            out_e ** 2
        ),
        1e-9,
    )
)

out_e_norm = (
    out_e
    /
    np.maximum(
        np.sqrt(
            out_n ** 2
            +
            out_e ** 2
        ),
        1e-9,
    )
)


true_norm = np.sqrt(
    out_true_n ** 2
    +
    out_true_e ** 2
)

true_norm = np.maximum(
    true_norm,
    1e-9,
)


true_n_norm = (
    out_true_n
    /
    true_norm
)

true_e_norm = (
    out_true_e
    /
    true_norm
)


t = outage[
    "time_s"
].to_numpy()


plt.plot(
    t[::sample_step],
    out_n_norm[::sample_step],
    label="Rotated phone X/Y → North",
)

plt.plot(
    t[::sample_step],
    true_n_norm[::sample_step],
    label="GNSS North direction",
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "North directional component"
)

plt.title(
    "Frame Alignment During GNSS Outage"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

vector_plot = (
    OUTPUTS
    /
    "prototype_v10_2_frame_alignment_outage.png"
)

plt.savefig(
    vector_plot,
    dpi=200,
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "best_rotation_deg":
                best_angle,

            "mean_direction_error_deg":
                float(
                    best_row[
                        "direction_error_deg"
                    ]
                ),

            "north_mae_mps":
                best_metrics[
                    "north_mae"
                ],

            "east_mae_mps":
                best_metrics[
                    "east_mae"
                ],

            "north_rmse_mps":
                best_metrics[
                    "north_rmse"
                ],

            "east_rmse_mps":
                best_metrics[
                    "east_rmse"
                ],

            "outage_north_correlation":
                safe_corr(
                    out_n,
                    out_true_n,
                ),

            "outage_east_correlation":
                safe_corr(
                    out_e,
                    out_true_e,
                ),
        }
    ]
)


summary_file = (
    OUTPUT_DIR
    /
    "prototype_v10_2_frame_alignment_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("V10.2 COMPLETE")
print("=" * 80)

print(
    "\nRotation search:"
)

print(
    rotation_file
)

print(
    "\nFrame-alignment data:"
)

print(
    output_file
)

print(
    "\nSummary:"
)

print(
    summary_file
)

print(
    "\nPlots:"
)

print(
    rotation_plot
)

print(
    vector_plot
)