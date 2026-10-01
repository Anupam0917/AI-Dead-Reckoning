import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "direct_orientation_gravity_validation.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("STEP 6.6 CORRECTED - DIRECT ORIENTATION GRAVITY VALIDATION")
print("=" * 75)

stationary = pd.read_csv(
    STATIONARY_FILE
)

orientation = pd.read_csv(
    ORIENTATION_FILE
)

print(
    f"\nStationary rows: {len(stationary)}"
)

print(
    f"Orientation rows: {len(orientation)}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_stationary = [
    "time_seconds",
    "accel_north",
    "accel_east",
    "accel_down",
    "strict_stationary"
]

required_orientation = [
    "time_seconds",
    "ned_gravity_n",
    "ned_gravity_e",
    "ned_gravity_d"
]

for column in required_stationary:

    if column not in stationary.columns:
        raise ValueError(
            f"Missing stationary column: {column}"
        )

for column in required_orientation:

    if column not in orientation.columns:
        raise ValueError(
            f"Missing orientation column: {column}"
        )


# ============================================================
# STRICT STATIONARY DATA
# ============================================================

strict = stationary[
    stationary["strict_stationary"].astype(bool)
].copy()

print(
    f"\nStrict stationary samples: {len(strict)}"
)


# ============================================================
# MATCH BY TIME
# ============================================================

strict = strict.sort_values(
    "time_seconds"
)

orientation = orientation.sort_values(
    "time_seconds"
)

merged = pd.merge_asof(
    strict,
    orientation,
    on="time_seconds",
    direction="nearest",
    tolerance=0.051
)

merged = merged.dropna(
    subset=[
        "ned_gravity_n",
        "ned_gravity_e",
        "ned_gravity_d"
    ]
).copy()

print(
    f"Matched samples: {len(merged)}"
)


# ============================================================
# MEASURED ACCELERATION
# ============================================================

a_n = merged[
    "accel_north"
].to_numpy()

a_e = merged[
    "accel_east"
].to_numpy()

a_d = merged[
    "accel_down"
].to_numpy()


# ============================================================
# ORIENTATION GRAVITY
# ============================================================

g_n = merged[
    "ned_gravity_n"
].to_numpy()

g_e = merged[
    "ned_gravity_e"
].to_numpy()

g_d = merged[
    "ned_gravity_d"
].to_numpy()


# ============================================================
# GRAVITY MAGNITUDE
# ============================================================

g_mag = np.sqrt(
    g_n**2 +
    g_e**2 +
    g_d**2
)

print(
    "\nOrientation gravity magnitude:"
    f"\n  Mean   : {g_mag.mean():.6f}"
    f"\n  Median : {np.median(g_mag):.6f}"
    f"\n  Std    : {g_mag.std():.6f}"
)


# ============================================================
# HORIZONTAL GRAVITY COMPONENT
# ============================================================

g_horizontal = np.sqrt(
    g_n**2 +
    g_e**2
)


# ============================================================
# MEASURED HORIZONTAL RESIDUAL
# ============================================================

horizontal_measured = np.sqrt(
    a_n**2 +
    a_e**2
)


# ============================================================
# GRAVITY-REMOVED ACCELERATION
# ============================================================

residual_n = a_n - g_n
residual_e = a_e - g_e
residual_d = a_d - g_d

horizontal_after = np.sqrt(
    residual_n**2 +
    residual_e**2
)

total_after = np.sqrt(
    residual_n**2 +
    residual_e**2 +
    residual_d**2
)


# ============================================================
# DIRECTION ERROR
# ============================================================

a_mag = np.sqrt(
    a_n**2 +
    a_e**2 +
    a_d**2
)

dot = (
    a_n * g_n +
    a_e * g_e +
    a_d * g_d
)

cos_angle = dot / (
    a_mag * g_mag + 1e-12
)

cos_angle = np.clip(
    cos_angle,
    -1.0,
    1.0
)

direction_error = np.degrees(
    np.arccos(cos_angle)
)


# ============================================================
# SAVE VALUES
# ============================================================

merged[
    "orientation_gravity_horizontal"
] = g_horizontal

merged[
    "measured_horizontal_residual"
] = horizontal_measured

merged[
    "gravity_removed_n"
] = residual_n

merged[
    "gravity_removed_e"
] = residual_e

merged[
    "gravity_removed_d"
] = residual_d

merged[
    "horizontal_after_gravity_removal"
] = horizontal_after

merged[
    "total_after_gravity_removal"
] = total_after

merged[
    "gravity_direction_error_deg"
] = direction_error


# ============================================================
# GLOBAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("GLOBAL RESULTS")
print("=" * 75)

print(
    "\nMeasured horizontal acceleration:"
    f"\n  Mean   : {horizontal_measured.mean():.6f}"
    f"\n  Median : {np.median(horizontal_measured):.6f}"
    f"\n  Std    : {horizontal_measured.std():.6f}"
    f"\n  Max    : {horizontal_measured.max():.6f}"
)

print(
    "\nOrientation horizontal gravity:"
    f"\n  Mean   : {g_horizontal.mean():.6f}"
    f"\n  Median : {np.median(g_horizontal):.6f}"
    f"\n  Max    : {g_horizontal.max():.6f}"
)

print(
    "\nHorizontal residual after gravity removal:"
    f"\n  Mean   : {horizontal_after.mean():.6f}"
    f"\n  Median : {np.median(horizontal_after):.6f}"
    f"\n  Std    : {horizontal_after.std():.6f}"
    f"\n  Max    : {horizontal_after.max():.6f}"
)

print(
    "\nGravity direction error:"
    f"\n  Mean   : {direction_error.mean():.6f} deg"
    f"\n  Median : {np.median(direction_error):.6f} deg"
    f"\n  Std    : {direction_error.std():.6f} deg"
    f"\n  Max    : {direction_error.max():.6f} deg"
)


# ============================================================
# RESIDUAL CHANGE
# ============================================================

before_mean = horizontal_measured.mean()
after_mean = horizontal_after.mean()

change_percent = (
    100.0 *
    (after_mean - before_mean)
    / before_mean
)

print(
    "\nChange after subtracting orientation gravity:"
    f"\n  {change_percent:+.2f}%"
)


# ============================================================
# CORRELATIONS
# ============================================================

corr = np.corrcoef(
    g_horizontal,
    horizontal_measured
)[0, 1]

corr_direction = np.corrcoef(
    direction_error,
    horizontal_measured
)[0, 1]

print(
    "\nCorrelations:"
    f"\n  Horizontal gravity vs measured residual: "
    f"{corr:.4f}"
    f"\n  Direction error vs measured residual: "
    f"{corr_direction:.4f}"
)


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved:"
    f"\n{OUTPUT_FILE}"
)


# ============================================================
# PLOT 1
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    horizontal_measured,
    bins=40,
    alpha=0.6,
    label="Measured"
)

plt.hist(
    horizontal_after,
    bins=40,
    alpha=0.6,
    label="After gravity removal"
)

plt.xlabel(
    "Horizontal acceleration (m/s²)"
)

plt.ylabel(
    "Samples"
)

plt.title(
    "Stationary Horizontal Acceleration Before and After Gravity Removal"
)

plt.legend()
plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "direct_gravity_removal_comparison.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    g_horizontal,
    horizontal_measured,
    s=8,
    alpha=0.35
)

plt.xlabel(
    "Orientation Estimated Horizontal Gravity (m/s²)"
)

plt.ylabel(
    "Measured Horizontal Acceleration (m/s²)"
)

plt.title(
    "Orientation Gravity vs Measured Stationary Acceleration"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "orientation_gravity_vs_measured.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PLOT 3
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    merged["time_seconds"],
    direction_error,
    linewidth=1
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Gravity Direction Error (degrees)"
)

plt.title(
    "Orientation Gravity Direction Error During Stationary Periods"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "direct_gravity_direction_error.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 6.6 CORRECTED COMPLETE")
print("=" * 75)

print(
    "\nThis experiment uses the NED gravity vector produced "
    "directly by the existing orientation pipeline."
)

print(
    "\nDo NOT use the previous quaternion gravity-removal "
    "result to make navigation decisions."
)

print("=" * 75)