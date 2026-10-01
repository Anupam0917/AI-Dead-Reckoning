import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATIONARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "strict_stationary_samples.csv"
)

ORIENTATION_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "dynamic_complementary_quaternion.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "quaternion_gravity_residual.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

G = 9.80665


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("STEP 6.6 - QUATERNION-BASED GRAVITY RESIDUAL ANALYSIS")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading stationary samples...")

stationary = pd.read_csv(STATIONARY_FILE)

print("Loading quaternion orientation...")

orientation = pd.read_csv(ORIENTATION_FILE)

print(f"Stationary rows:  {len(stationary)}")
print(f"Orientation rows: {len(orientation)}")


# ============================================================
# CHECK COLUMNS
# ============================================================

stationary_required = [
    "time_seconds",
    "accel_north",
    "accel_east",
    "accel_down",
    "strict_stationary"
]

orientation_required = [
    "time_seconds",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w"
]

missing_stationary = [
    c for c in stationary_required
    if c not in stationary.columns
]

missing_orientation = [
    c for c in orientation_required
    if c not in orientation.columns
]

if missing_stationary:
    raise ValueError(
        f"Missing stationary columns: {missing_stationary}"
    )

if missing_orientation:
    raise ValueError(
        f"Missing orientation columns: {missing_orientation}"
    )


# ============================================================
# SELECT STRICT STATIONARY SAMPLES
# ============================================================

strict = stationary[
    stationary["strict_stationary"].astype(bool)
].copy()

print(
    f"\nStrict stationary samples: {len(strict)}"
)


# ============================================================
# MATCH ORIENTATION
# ============================================================

print("\nMatching quaternion orientation...")

strict = strict.sort_values("time_seconds")

orientation = orientation.sort_values("time_seconds")

merged = pd.merge_asof(
    strict,
    orientation,
    on="time_seconds",
    direction="nearest",
    tolerance=0.051
)

merged = merged.dropna(
    subset=[
        "quat_x",
        "quat_y",
        "quat_z",
        "quat_w"
    ]
).copy()

print(
    f"Matched samples: {len(merged)}"
)


# ============================================================
# NORMALIZE QUATERNIONS
# ============================================================

qx = merged["quat_x"].to_numpy()
qy = merged["quat_y"].to_numpy()
qz = merged["quat_z"].to_numpy()
qw = merged["quat_w"].to_numpy()

q_norm = np.sqrt(
    qx**2 +
    qy**2 +
    qz**2 +
    qw**2
)

print(
    f"\nQuaternion norm:"
    f"\n  Mean : {q_norm.mean():.8f}"
    f"\n  Std  : {q_norm.std():.8f}"
    f"\n  Min  : {q_norm.min():.8f}"
    f"\n  Max  : {q_norm.max():.8f}"
)

# Normalize to protect against numerical drift.

qx = qx / q_norm
qy = qy / q_norm
qz = qz / q_norm
qw = qw / q_norm


# ============================================================
# QUATERNION -> NED GRAVITY VECTOR
# ============================================================
#
# The quaternion describes the estimated phone/body orientation.
#
# We use the corresponding rotation to determine where the
# gravity vector appears in NED coordinates.
#
# Body-frame gravity is approximately:
#
#       [0, 0, G]
#
# depending on the coordinate convention used by the existing
# orientation pipeline.
#
# We therefore calculate the rotated gravity vector explicitly.
# ============================================================

gravity_n = 2.0 * (
    qx * qz + qw * qy
) * G

gravity_e = 2.0 * (
    qy * qz - qw * qx
) * G

gravity_d = (
    qw**2
    - qx**2
    - qy**2
    + qz**2
) * G


# ============================================================
# ACTUAL NED ACCELERATION
# ============================================================

accel_n = merged["accel_north"].to_numpy()
accel_e = merged["accel_east"].to_numpy()
accel_d = merged["accel_down"].to_numpy()


# ============================================================
# STORE QUATERNION GRAVITY
# ============================================================

merged["quat_gravity_n"] = gravity_n
merged["quat_gravity_e"] = gravity_e
merged["quat_gravity_d"] = gravity_d


# ============================================================
# GRAVITY MAGNITUDE
# ============================================================

gravity_magnitude = np.sqrt(
    gravity_n**2 +
    gravity_e**2 +
    gravity_d**2
)

merged["quat_gravity_magnitude"] = gravity_magnitude


# ============================================================
# HORIZONTAL GRAVITY COMPONENT
# ============================================================

gravity_horizontal = np.sqrt(
    gravity_n**2 +
    gravity_e**2
)

merged["quat_gravity_horizontal"] = gravity_horizontal


# ============================================================
# BEFORE-CORRECTION RESIDUAL
# ============================================================

horizontal_before = np.sqrt(
    accel_n**2 +
    accel_e**2
)

merged["horizontal_residual_before"] = horizontal_before


# ============================================================
# ORIENTATION-BASED GRAVITY RESIDUAL
# ============================================================

residual_n = accel_n - gravity_n
residual_e = accel_e - gravity_e
residual_d = accel_d - gravity_d

merged["gravity_residual_n"] = residual_n
merged["gravity_residual_e"] = residual_e
merged["gravity_residual_d"] = residual_d

residual_horizontal = np.sqrt(
    residual_n**2 +
    residual_e**2
)

residual_total = np.sqrt(
    residual_n**2 +
    residual_e**2 +
    residual_d**2
)

merged["horizontal_residual_after"] = residual_horizontal
merged["total_gravity_residual"] = residual_total


# ============================================================
# RESIDUAL REDUCTION
# ============================================================

merged["horizontal_residual_reduction"] = (
    merged["horizontal_residual_before"]
    - merged["horizontal_residual_after"]
)

merged["horizontal_reduction_percent"] = (
    100.0
    * merged["horizontal_residual_reduction"]
    / (merged["horizontal_residual_before"] + 1e-9)
)


# ============================================================
# GRAVITY DIRECTION ERROR
# ============================================================

# Measured stationary acceleration should approximately point
# along the gravity direction in this convention.
#
# Calculate angle between measured acceleration and quaternion
# gravity vector.

actual_norm = np.sqrt(
    accel_n**2 +
    accel_e**2 +
    accel_d**2
)

dot_product = (
    accel_n * gravity_n
    + accel_e * gravity_e
    + accel_d * gravity_d
)

denominator = (
    actual_norm *
    gravity_magnitude
)

cos_angle = dot_product / (
    denominator + 1e-12
)

cos_angle = np.clip(
    cos_angle,
    -1.0,
    1.0
)

gravity_direction_error_deg = np.degrees(
    np.arccos(cos_angle)
)

merged["gravity_direction_error_deg"] = (
    gravity_direction_error_deg
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("GLOBAL RESULTS")
print("=" * 75)

print(
    "\nQuaternion gravity magnitude:"
    f"\n  Mean   : {gravity_magnitude.mean():.6f} m/s^2"
    f"\n  Median : {np.median(gravity_magnitude):.6f} m/s^2"
    f"\n  Std    : {gravity_magnitude.std():.6f} m/s^2"
)

print(
    "\nHorizontal residual BEFORE quaternion correction:"
    f"\n  Mean   : {horizontal_before.mean():.6f} m/s^2"
    f"\n  Median : {np.median(horizontal_before):.6f} m/s^2"
    f"\n  Std    : {horizontal_before.std():.6f} m/s^2"
    f"\n  Max    : {horizontal_before.max():.6f} m/s^2"
)

print(
    "\nHorizontal residual AFTER quaternion gravity subtraction:"
    f"\n  Mean   : {residual_horizontal.mean():.6f} m/s^2"
    f"\n  Median : {np.median(residual_horizontal):.6f} m/s^2"
    f"\n  Std    : {residual_horizontal.std():.6f} m/s^2"
    f"\n  Max    : {residual_horizontal.max():.6f} m/s^2"
)

print(
    "\nGravity direction error:"
    f"\n  Mean   : {gravity_direction_error_deg.mean():.6f} deg"
    f"\n  Median : {np.median(gravity_direction_error_deg):.6f} deg"
    f"\n  Std    : {gravity_direction_error_deg.std():.6f} deg"
    f"\n  Max    : {gravity_direction_error_deg.max():.6f} deg"
)


# ============================================================
# REDUCTION STATISTICS
# ============================================================

mean_before = horizontal_before.mean()
mean_after = residual_horizontal.mean()

if mean_before > 0:
    reduction = (
        100.0 *
        (mean_before - mean_after)
        / mean_before
    )
else:
    reduction = 0.0

print(
    "\nMean horizontal residual reduction:"
    f"\n  {reduction:.2f}%"
)


# ============================================================
# CORRELATION
# ============================================================

corr_horizontal = np.corrcoef(
    merged["quat_gravity_horizontal"],
    merged["horizontal_residual_before"]
)[0, 1]

corr_direction = np.corrcoef(
    merged["gravity_direction_error_deg"],
    merged["horizontal_residual_before"]
)[0, 1]

print(
    "\nCorrelations:"
    f"\n  Quaternion horizontal gravity vs "
    f"measured residual : {corr_horizontal:.4f}"
    f"\n  Gravity direction error vs "
    f"measured residual : {corr_direction:.4f}"
)


# ============================================================
# WINDOW ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("WINDOW ANALYSIS")
print("=" * 75)

time_diff = merged["time_seconds"].diff()

window_id = (
    (time_diff > 0.2)
    .fillna(True)
    .cumsum()
)

merged["window_id"] = window_id

window_results = []

for wid, group in merged.groupby("window_id"):

    if len(group) < 5:
        continue

    before = group[
        "horizontal_residual_before"
    ].mean()

    after = group[
        "horizontal_residual_after"
    ].mean()

    reduction_percent = (
        100.0 * (before - after) / (before + 1e-9)
    )

    window_results.append({
        "window_id": int(wid),
        "start_time_s": group["time_seconds"].min(),
        "end_time_s": group["time_seconds"].max(),
        "samples": len(group),
        "gravity_horizontal_mean": group[
            "quat_gravity_horizontal"
        ].mean(),
        "residual_before_mean": before,
        "residual_after_mean": after,
        "reduction_percent": reduction_percent,
        "gravity_direction_error_mean_deg": group[
            "gravity_direction_error_deg"
        ].mean()
    })


window_df = pd.DataFrame(window_results)

print(
    f"\nWindows analysed: {len(window_df)}"
)

print(
    window_df.to_string(index=False)
)


# ============================================================
# SAVE DATA
# ============================================================

merged.to_csv(
    OUTPUT_CSV,
    index=False
)

WINDOW_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "quaternion_gravity_residual_windows.csv"
)

window_df.to_csv(
    WINDOW_OUTPUT,
    index=False
)

print(
    f"\nSaved:"
    f"\n{OUTPUT_CSV}"
    f"\n{WINDOW_OUTPUT}"
)


# ============================================================
# PLOT 1
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    horizontal_before,
    bins=40,
    alpha=0.6,
    label="Before"
)

plt.hist(
    residual_horizontal,
    bins=40,
    alpha=0.6,
    label="After"
)

plt.xlabel("Horizontal Residual (m/s²)")
plt.ylabel("Number of Samples")

plt.title(
    "Horizontal Residual Before vs After Quaternion Gravity Compensation"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plot1 = os.path.join(
    OUTPUT_DIR,
    "horizontal_residual_before_after.png"
)

plt.savefig(plot1, dpi=200)

plt.close()


# ============================================================
# PLOT 2
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    gravity_horizontal,
    horizontal_before,
    s=8,
    alpha=0.35
)

plt.xlabel(
    "Quaternion Estimated Horizontal Gravity (m/s²)"
)

plt.ylabel(
    "Measured Horizontal Residual (m/s²)"
)

plt.title(
    "Quaternion Gravity Component vs Measured Residual"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plot2 = os.path.join(
    OUTPUT_DIR,
    "quaternion_gravity_residual.png"
)

plt.savefig(plot2, dpi=200)

plt.close()


# ============================================================
# PLOT 3
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    merged["time_seconds"],
    merged["gravity_direction_error_deg"],
    linewidth=1
)

plt.xlabel("Time (s)")
plt.ylabel("Gravity Direction Error (degrees)")

plt.title(
    "Quaternion Gravity Direction Error During Stationary Periods"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plot3 = os.path.join(
    OUTPUT_DIR,
    "gravity_vector_comparison.png"
)

plt.savefig(plot3, dpi=200)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 6.6 COMPLETE")
print("=" * 75)

print("\nGenerated:")
print("1.", OUTPUT_CSV)
print("2.", WINDOW_OUTPUT)
print("3.", plot1)
print("4.", plot2)
print("5.", plot3)

print(
    "\nNext decision will depend on whether quaternion-based "
    "gravity compensation actually reduces the stationary "
    "horizontal residual."
)

print("=" * 75)