import os
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

NED_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ned_acceleration.csv"
)

STATIONARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "strict_stationary_samples.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("STEP 6.8 - NED TRANSFORMATION VALIDATION")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

print("\nLoading original dataset...")

original = pd.read_csv(
    ORIGINAL_FILE,
    encoding="cp1252"
)

print(
    f"Original rows: {len(original)}"
)

print("\nLoading existing NED acceleration...")

ned = pd.read_csv(
    NED_FILE
)

print(
    f"NED rows: {len(ned)}"
)

print("\nLoading stationary detection...")

stationary = pd.read_csv(
    STATIONARY_FILE
)


# ============================================================
# FIND ACCELEROMETER COLUMNS
# ============================================================

accel_columns = []

for column in original.columns:

    if "ACCELEROMETER" in str(column).upper():

        accel_columns.append(column)


if len(accel_columns) < 3:

    raise ValueError(
        "Could not find three accelerometer columns."
    )


print("\nAccelerometer columns:")

for column in accel_columns:

    print(
        f"  {column}"
    )


# ============================================================
# SELECT X/Y/Z
# ============================================================

accel_x = None
accel_y = None
accel_z = None

for column in accel_columns:

    name = str(column).upper()

    if " X " in name or name.endswith(" X"):

        accel_x = column

    elif " Y " in name or name.endswith(" Y"):

        accel_y = column

    elif " Z " in name or name.endswith(" Z"):

        accel_z = column


if None in [
    accel_x,
    accel_y,
    accel_z
]:

    accel_x = accel_columns[0]
    accel_y = accel_columns[1]
    accel_z = accel_columns[2]


print("\nSelected:")

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
# DIRECT AXIS/SIGN TRANSFORMATION
# ============================================================

body_x = pd.to_numeric(
    original[accel_x],
    errors="coerce"
).to_numpy()

body_y = pd.to_numeric(
    original[accel_y],
    errors="coerce"
).to_numpy()

body_z = pd.to_numeric(
    original[accel_z],
    errors="coerce"
).to_numpy()


direct_n = -body_x
direct_e = -body_y
direct_d = body_z


# ============================================================
# FIND NED COLUMNS
# ============================================================

print("\nNED columns:")

print(
    ned.columns.tolist()
)


# ============================================================
# AUTO-DETECT NED COLUMNS
# ============================================================

def find_column(df, keywords):

    for column in df.columns:

        name = str(column).lower()

        if all(
            key.lower() in name
            for key in keywords
        ):

            return column

    return None


ned_n_col = find_column(
    ned,
    ["north"]
)

ned_e_col = find_column(
    ned,
    ["east"]
)

ned_d_col = find_column(
    ned,
    ["down"]
)


if None in [
    ned_n_col,
    ned_e_col,
    ned_d_col
]:

    raise ValueError(
        "Could not identify N/E/D columns."
    )


print(
    f"\nDetected NED:"
    f"\nNorth = {ned_n_col}"
    f"\nEast  = {ned_e_col}"
    f"\nDown  = {ned_d_col}"
)


# ============================================================
# ALIGN LENGTH
# ============================================================

n = min(
    len(original),
    len(ned)
)

direct_n = direct_n[:n]
direct_e = direct_e[:n]
direct_d = direct_d[:n]

existing_n = pd.to_numeric(
    ned[ned_n_col],
    errors="coerce"
).to_numpy()[:n]

existing_e = pd.to_numeric(
    ned[ned_e_col],
    errors="coerce"
).to_numpy()[:n]

existing_d = pd.to_numeric(
    ned[ned_d_col],
    errors="coerce"
).to_numpy()[:n]


# ============================================================
# DIFFERENCE
# ============================================================

difference_n = (
    existing_n -
    direct_n
)

difference_e = (
    existing_e -
    direct_e
)

difference_d = (
    existing_d -
    direct_d
)


# ============================================================
# STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("DIRECT AXIS/SIGN MAPPING vs EXISTING NED")
print("=" * 80)

print(
    "\nNorth difference:"
    f"\n  Mean : {np.nanmean(difference_n):.6f}"
    f"\n  MAE  : {np.nanmean(np.abs(difference_n)):.6f}"
    f"\n  RMSE : {np.sqrt(np.nanmean(difference_n**2)):.6f}"
)

print(
    "\nEast difference:"
    f"\n  Mean : {np.nanmean(difference_e):.6f}"
    f"\n  MAE  : {np.nanmean(np.abs(difference_e)):.6f}"
    f"\n  RMSE : {np.sqrt(np.nanmean(difference_e**2)):.6f}"
)

print(
    "\nDown difference:"
    f"\n  Mean : {np.nanmean(difference_d):.6f}"
    f"\n  MAE  : {np.nanmean(np.abs(difference_d)):.6f}"
    f"\n  RMSE : {np.sqrt(np.nanmean(difference_d**2)):.6f}"
)


# ============================================================
# CORRELATION
# ============================================================

valid_n = np.isfinite(
    direct_n
) & np.isfinite(
    existing_n
)

valid_e = np.isfinite(
    direct_e
) & np.isfinite(
    existing_e
)

valid_d = np.isfinite(
    direct_d
) & np.isfinite(
    existing_d
)

corr_n = np.corrcoef(
    direct_n[valid_n],
    existing_n[valid_n]
)[0, 1]

corr_e = np.corrcoef(
    direct_e[valid_e],
    existing_e[valid_e]
)[0, 1]

corr_d = np.corrcoef(
    direct_d[valid_d],
    existing_d[valid_d]
)[0, 1]


print(
    "\nCorrelation:"
    f"\n  North: {corr_n:.6f}"
    f"\n  East : {corr_e:.6f}"
    f"\n  Down : {corr_d:.6f}"
)


# ============================================================
# STATIONARY ANALYSIS
# ============================================================

strict_indices = np.flatnonzero(
    stationary[
        "strict_stationary"
    ].astype(bool)
    .to_numpy()
)

strict_indices = strict_indices[
    strict_indices < n
]

print(
    f"\nStrict stationary samples used:"
    f" {len(strict_indices)}"
)


print("\nStationary direct mapping:")

print(
    f"North mean: "
    f"{np.mean(direct_n[strict_indices]):.6f}"
)

print(
    f"East mean: "
    f"{np.mean(direct_e[strict_indices]):.6f}"
)

print(
    f"Down mean: "
    f"{np.mean(direct_d[strict_indices]):.6f}"
)

print("\nStationary existing NED:")

print(
    f"North mean: "
    f"{np.mean(existing_n[strict_indices]):.6f}"
)

print(
    f"East mean: "
    f"{np.mean(existing_e[strict_indices]):.6f}"
)

print(
    f"Down mean: "
    f"{np.mean(existing_d[strict_indices]):.6f}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 6.8 COMPLETE")
print("=" * 80)

print(
    "\nThis tells us whether the current NED acceleration "
    "pipeline is actually applying the discovered "
    "Body X/Y/Z → N/E/D convention."
)

print(
    "\nDo NOT modify ned_acceleration.py yet."
)

print("=" * 80)