import os
import itertools
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ORIGINAL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Synchronised V abd S datasets",
    "Categorised IOVNB Dataset",
    "M (Driver B)",
    "S-M.csv"
)

ORIENTATION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "dynamic_complementary_quaternion.csv"
)

STATIONARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "strict_stationary_samples.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "accelerometer_ned_convention_test.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("STEP 6.7 - ACCELEROMETER ↔ NED FRAME CONVENTION TEST")
print("=" * 80)


# ============================================================
# LOAD FILES
# ============================================================

print("\nLoading original synchronized dataset...")

original = pd.read_csv(
    ORIGINAL_FILE,
    encoding="cp1252"
)

print(
    f"Original rows: {len(original)}"
)

print("\nLoading orientation data...")

orientation = pd.read_csv(
    ORIENTATION_FILE
)

print(
    f"Orientation rows: {len(orientation)}"
)

print("\nLoading stationary samples...")

stationary = pd.read_csv(
    STATIONARY_FILE
)

print(
    f"Stationary rows: {len(stationary)}"
)


# ============================================================
# AUTOMATICALLY FIND ACCELEROMETER COLUMNS
# ============================================================

print("\nSearching for raw accelerometer columns...")

accelerometer_columns = []

for column in original.columns:

    name = str(column).upper()

    if "ACCELEROMETER" in name:

        accelerometer_columns.append(column)


print(
    "\nAccelerometer columns found:"
)

for column in accelerometer_columns:

    print(
        f"  {column}"
    )


if len(accelerometer_columns) < 3:

    raise ValueError(
        "Could not find all three raw accelerometer "
        "columns in the original dataset."
    )


# ============================================================
# SELECT X/Y/Z
# ============================================================
#
# The IO-VNBD dataset should contain:
#
# ACCELEROMETER X
# ACCELEROMETER Y
# ACCELEROMETER Z
#
# We identify them by X/Y/Z appearing after the
# ACCELEROMETER label.
# ============================================================

accel_x = None
accel_y = None
accel_z = None

for column in accelerometer_columns:

    name = str(column).upper()

    if " X " in name or name.endswith(" X"):

        accel_x = column

    elif " Y " in name or name.endswith(" Y"):

        accel_y = column

    elif " Z " in name or name.endswith(" Z"):

        accel_z = column


# ============================================================
# FALLBACK
# ============================================================

if accel_x is None or accel_y is None or accel_z is None:

    print(
        "\nAutomatic X/Y/Z detection was ambiguous."
    )

    print(
        "Using first three accelerometer columns."
    )

    accel_x = accelerometer_columns[0]
    accel_y = accelerometer_columns[1]
    accel_z = accelerometer_columns[2]


print("\nSelected accelerometer columns:")

print(
    f"X = {accel_x}"
)

print(
    f"Y = {accel_y}"
)

print(
    f"Z = {accel_z}"
)


# ============================================================
# FIND TIMESTAMP COLUMN
# ============================================================

timestamp_column = None

for column in original.columns:

    name = str(column).upper()

    if "DATE" in name and "YYYY" in name:

        timestamp_column = column
        break


if timestamp_column is None:

    raise ValueError(
        "Could not locate the timestamp column."
    )


print(
    f"\nTimestamp column:"
    f"\n{timestamp_column}"
)


# ============================================================
# CREATE TIME
# ============================================================

print("\nParsing timestamps...")

timestamps = pd.to_datetime(
    original[timestamp_column],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)

if timestamps.isna().any():

    raise ValueError(
        "Some timestamps could not be parsed."
    )


time_seconds = (
    timestamps -
    timestamps.iloc[0]
).dt.total_seconds()


# ============================================================
# BODY FRAME DATA
# ============================================================

body = pd.DataFrame({

    "time_seconds":
        time_seconds.to_numpy(),

    "body_x":
        pd.to_numeric(
            original[accel_x],
            errors="coerce"
        ).to_numpy(),

    "body_y":
        pd.to_numeric(
            original[accel_y],
            errors="coerce"
        ).to_numpy(),

    "body_z":
        pd.to_numeric(
            original[accel_z],
            errors="coerce"
        ).to_numpy()
})


# ============================================================
# REMOVE INVALID SENSOR ROWS
# ============================================================

body = body.dropna(
    subset=[
        "body_x",
        "body_y",
        "body_z"
    ]
).copy()


# ============================================================
# STRICT STATIONARY SAMPLES
# ============================================================

if "strict_stationary" not in stationary.columns:

    raise ValueError(
        "strict_stationary column missing."
    )

strict = stationary[
    stationary["strict_stationary"].astype(bool)
].copy()

strict = strict[
    ["time_seconds"]
].sort_values(
    "time_seconds"
)

print(
    f"\nStrict stationary samples:"
    f" {len(strict)}"
)


# ============================================================
# MATCH RAW ACCELEROMETER
# ============================================================

body = body.sort_values(
    "time_seconds"
)

matched = pd.merge_asof(
    strict,
    body,
    on="time_seconds",
    direction="nearest",
    tolerance=0.051
)

matched = matched.dropna(
    subset=[
        "body_x",
        "body_y",
        "body_z"
    ]
).copy()

print(
    f"Matched body-frame samples:"
    f" {len(matched)}"
)


# ============================================================
# MATCH ORIENTATION GRAVITY
# ============================================================

orientation = orientation.sort_values(
    "time_seconds"
)

matched = pd.merge_asof(
    matched.sort_values(
        "time_seconds"
    ),
    orientation[
        [
            "time_seconds",
            "ned_gravity_n",
            "ned_gravity_e",
            "ned_gravity_d"
        ]
    ],
    on="time_seconds",
    direction="nearest",
    tolerance=0.051
)

matched = matched.dropna(
    subset=[
        "ned_gravity_n",
        "ned_gravity_e",
        "ned_gravity_d"
    ]
).copy()

print(
    f"Fully matched samples:"
    f" {len(matched)}"
)


# ============================================================
# VECTORS
# ============================================================

body_vector = matched[
    [
        "body_x",
        "body_y",
        "body_z"
    ]
].to_numpy(
    dtype=float
)

gravity_vector = matched[
    [
        "ned_gravity_n",
        "ned_gravity_e",
        "ned_gravity_d"
    ]
].to_numpy(
    dtype=float
)


# ============================================================
# NORMALIZE
# ============================================================

body_norm = np.linalg.norm(
    body_vector,
    axis=1
)

gravity_norm = np.linalg.norm(
    gravity_vector,
    axis=1
)

body_unit = (
    body_vector /
    (body_norm[:, None] + 1e-12)
)

gravity_unit = (
    gravity_vector /
    (gravity_norm[:, None] + 1e-12)
)


# ============================================================
# BASIC CHECK
# ============================================================

print("\n" + "=" * 80)
print("BASIC DATA CHECK")
print("=" * 80)

print(
    f"\nBody acceleration magnitude:"
    f"\n  Mean   : {body_norm.mean():.6f}"
    f"\n  Median : {np.median(body_norm):.6f}"
    f"\n  Std    : {body_norm.std():.6f}"
    f"\n  Min    : {body_norm.min():.6f}"
    f"\n  Max    : {body_norm.max():.6f}"
)

print(
    f"\nOrientation gravity magnitude:"
    f"\n  Mean   : {gravity_norm.mean():.6f}"
    f"\n  Median : {np.median(gravity_norm):.6f}"
    f"\n  Std    : {gravity_norm.std():.6f}"
)


# ============================================================
# TEST 48 CONVENTIONS
# ============================================================

results = []

axes = [0, 1, 2]

axis_names = [
    "X",
    "Y",
    "Z"
]

for permutation in itertools.permutations(axes):

    for signs in itertools.product(
        [-1, 1],
        repeat=3
    ):

        transformed = (
            body_unit[:, permutation]
            * np.array(signs)
        )

        dot = np.sum(
            transformed *
            gravity_unit,
            axis=1
        )

        dot = np.clip(
            dot,
            -1.0,
            1.0
        )

        angle = np.degrees(
            np.arccos(dot)
        )

        results.append({

            "axis_order":
                "".join(
                    axis_names[i]
                    for i in permutation
                ),

            "signs":
                "".join(
                    "+" if s == 1 else "-"
                    for s in signs
                ),

            "mean_angle_deg":
                angle.mean(),

            "median_angle_deg":
                np.median(angle),

            "std_angle_deg":
                angle.std(),

            "max_angle_deg":
                angle.max(),

            "within_5deg_percent":
                100.0 *
                np.mean(
                    angle <= 5.0
                ),

            "within_10deg_percent":
                100.0 *
                np.mean(
                    angle <= 10.0
                ),

            "within_20deg_percent":
                100.0 *
                np.mean(
                    angle <= 20.0
                )
        })


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    [
        "mean_angle_deg",
        "median_angle_deg"
    ]
).reset_index(drop=True)


# ============================================================
# TOP RESULTS
# ============================================================

print("\n" + "=" * 80)
print("BEST AXIS/SIGN CONVENTIONS")
print("=" * 80)

print(
    results_df.head(15).to_string(
        index=False
    )
)


# ============================================================
# BEST
# ============================================================

best = results_df.iloc[0]

print("\n" + "=" * 80)
print("BEST CONVENTION")
print("=" * 80)

print(
    f"\nAxis order: "
    f"{best['axis_order']}"
)

print(
    f"Signs: "
    f"{best['signs']}"
)

print(
    f"Mean angle: "
    f"{best['mean_angle_deg']:.4f}°"
)

print(
    f"Median angle: "
    f"{best['median_angle_deg']:.4f}°"
)

print(
    f"Within 5°: "
    f"{best['within_5deg_percent']:.2f}%"
)

print(
    f"Within 10°: "
    f"{best['within_10deg_percent']:.2f}%"
)

print(
    f"Within 20°: "
    f"{best['within_20deg_percent']:.2f}%"
)


# ============================================================
# HUMAN READABLE
# ============================================================

axis_order = best["axis_order"]
sign_string = best["signs"]

print("\nHuman-readable mapping:")

print(
    f"N = {sign_string[0]} * Body {axis_order[0]}"
)

print(
    f"E = {sign_string[1]} * Body {axis_order[1]}"
)

print(
    f"D = {sign_string[2]} * Body {axis_order[2]}"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved:"
    f"\n{OUTPUT_FILE}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 6.7 COMPLETE")
print("=" * 80)

print(
    "\nAll 48 signed axis permutations were tested."
)

print(
    "\nDo NOT modify the production NED transformation yet."
)

print(
    "\nWe will validate the best convention before "
    "changing anything in the navigation pipeline."
)

print("=" * 80)