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
    "orientation_leakage_comparison.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONSTANT
# ============================================================

G = 9.80665


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 6.5 - ORIENTATION vs GRAVITY LEAKAGE")
print("=" * 70)

print("\nLoading stationary samples...")
stationary = pd.read_csv(STATIONARY_FILE)

print("Loading orientation data...")
orientation = pd.read_csv(ORIENTATION_FILE)

print(f"Stationary rows:  {len(stationary)}")
print(f"Orientation rows: {len(orientation)}")


# ============================================================
# CHECK REQUIRED COLUMNS
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
    "roll_deg",
    "pitch_deg",
    "yaw_deg"
]

missing_stationary = [
    col for col in required_stationary
    if col not in stationary.columns
]

missing_orientation = [
    col for col in required_orientation
    if col not in orientation.columns
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

print(f"\nStrict stationary samples: {len(strict)}")


# ============================================================
# MERGE USING TIME
# ============================================================

print("\nMatching orientation with stationary samples...")

merged = pd.merge_asof(
    strict.sort_values("time_seconds"),
    orientation.sort_values("time_seconds"),
    on="time_seconds",
    direction="nearest",
    tolerance=0.051
)

# Remove samples for which orientation could not be matched
merged = merged.dropna(
    subset=["roll_deg", "pitch_deg"]
).copy()

print(f"Matched samples: {len(merged)}")


# ============================================================
# HORIZONTAL ACCELERATION RESIDUAL
# ============================================================

merged["horizontal_residual"] = np.sqrt(
    merged["accel_north"] ** 2 +
    merged["accel_east"] ** 2
)


# ============================================================
# TILT CALCULATION
# ============================================================

# Roll and pitch represent the estimated attitude tilt.
#
# We calculate a combined tilt magnitude rather than using
# roll or pitch independently.

roll_rad = np.deg2rad(merged["roll_deg"])
pitch_rad = np.deg2rad(merged["pitch_deg"])

merged["tilt_deg"] = np.sqrt(
    merged["roll_deg"] ** 2 +
    merged["pitch_deg"] ** 2
)


# ============================================================
# EXPECTED GRAVITY LEAKAGE
# ============================================================

tilt_rad = np.deg2rad(merged["tilt_deg"])

merged["expected_gravity_leakage"] = (
    G * np.sin(tilt_rad)
)


# ============================================================
# LEAKAGE RATIO
# ============================================================

merged["leakage_ratio"] = (
    merged["horizontal_residual"] /
    (merged["expected_gravity_leakage"] + 1e-6)
)


# ============================================================
# ERROR BETWEEN MEASURED AND EXPECTED
# ============================================================

merged["leakage_difference"] = (
    merged["horizontal_residual"]
    - merged["expected_gravity_leakage"]
)


# ============================================================
# CORRELATION
# ============================================================

correlation = merged[
    ["tilt_deg", "horizontal_residual"]
].corr().iloc[0, 1]

correlation_expected = merged[
    ["expected_gravity_leakage", "horizontal_residual"]
].corr().iloc[0, 1]


# ============================================================
# STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL RESULTS")
print("=" * 70)

print(
    f"\nTilt:"
    f"\n  Mean   : {merged['tilt_deg'].mean():.4f} deg"
    f"\n  Median : {merged['tilt_deg'].median():.4f} deg"
    f"\n  Std    : {merged['tilt_deg'].std():.4f} deg"
    f"\n  Max    : {merged['tilt_deg'].max():.4f} deg"
)

print(
    f"\nMeasured horizontal residual:"
    f"\n  Mean   : {merged['horizontal_residual'].mean():.6f} m/s^2"
    f"\n  Median : {merged['horizontal_residual'].median():.6f} m/s^2"
    f"\n  Std    : {merged['horizontal_residual'].std():.6f} m/s^2"
    f"\n  Max    : {merged['horizontal_residual'].max():.6f} m/s^2"
)

print(
    f"\nExpected gravity leakage:"
    f"\n  Mean   : {merged['expected_gravity_leakage'].mean():.6f} m/s^2"
    f"\n  Median : {merged['expected_gravity_leakage'].median():.6f} m/s^2"
    f"\n  Max    : {merged['expected_gravity_leakage'].max():.6f} m/s^2"
)

print(
    f"\nCorrelation:"
    f"\n  Tilt vs measured residual : {correlation:.4f}"
    f"\n  Expected leakage vs residual : {correlation_expected:.4f}"
)


# ============================================================
# WINDOW-LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("WINDOW-LEVEL ANALYSIS")
print("=" * 70)

# Identify stationary windows based on continuous gaps.
#
# A gap greater than 0.2 s means a new stationary segment.

time_diff = merged["time_seconds"].diff()

window_id = (
    (time_diff > 0.2)
    .fillna(True)
    .cumsum()
)

merged["window_id"] = window_id


window_results = []

for window_id_value, group in merged.groupby("window_id"):

    if len(group) < 5:
        continue

    tilt_mean = group["tilt_deg"].mean()

    residual_mean = group["horizontal_residual"].mean()

    expected_mean = group[
        "expected_gravity_leakage"
    ].mean()

    window_results.append({
        "window_id": int(window_id_value),
        "start_time_s": group["time_seconds"].min(),
        "end_time_s": group["time_seconds"].max(),
        "duration_s": (
            group["time_seconds"].max()
            - group["time_seconds"].min()
        ),
        "samples": len(group),
        "mean_roll_deg": group["roll_deg"].mean(),
        "mean_pitch_deg": group["pitch_deg"].mean(),
        "mean_tilt_deg": tilt_mean,
        "mean_horizontal_residual": residual_mean,
        "mean_expected_gravity_leakage": expected_mean,
        "mean_leakage_difference": (
            residual_mean - expected_mean
        )
    })


window_df = pd.DataFrame(window_results)

if len(window_df) > 1:

    window_corr = window_df[
        [
            "mean_tilt_deg",
            "mean_horizontal_residual"
        ]
    ].corr().iloc[0, 1]

else:

    window_corr = np.nan


print(f"\nStationary windows analysed: {len(window_df)}")

print(
    f"Window-level correlation: "
    f"{window_corr:.4f}"
)


# ============================================================
# PRINT WINDOW TABLE
# ============================================================

print("\nWindow results:")
print(
    window_df[
        [
            "window_id",
            "start_time_s",
            "end_time_s",
            "mean_roll_deg",
            "mean_pitch_deg",
            "mean_tilt_deg",
            "mean_horizontal_residual",
            "mean_expected_gravity_leakage"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE SAMPLE-LEVEL RESULTS
# ============================================================

merged.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"\nSaved sample-level results:"
    f"\n{OUTPUT_CSV}"
)


# ============================================================
# SAVE WINDOW RESULTS
# ============================================================

WINDOW_OUTPUT = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "orientation_leakage_windows.csv"
)

window_df.to_csv(
    WINDOW_OUTPUT,
    index=False
)

print(
    f"Saved window-level results:"
    f"\n{WINDOW_OUTPUT}"
)


# ============================================================
# PLOT 1
# TILT VS HORIZONTAL RESIDUAL
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    merged["tilt_deg"],
    merged["horizontal_residual"],
    s=8,
    alpha=0.35
)

plt.xlabel("Estimated Tilt (degrees)")
plt.ylabel("Horizontal Residual (m/s²)")
plt.title(
    "Estimated Tilt vs Stationary Horizontal Acceleration Residual"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plot1 = os.path.join(
    OUTPUT_DIR,
    "tilt_vs_horizontal_residual.png"
)

plt.savefig(plot1, dpi=200)

plt.close()


# ============================================================
# PLOT 2
# MEASURED VS EXPECTED GRAVITY LEAKAGE
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    merged["expected_gravity_leakage"],
    merged["horizontal_residual"],
    s=8,
    alpha=0.35
)

max_value = max(
    merged["expected_gravity_leakage"].max(),
    merged["horizontal_residual"].max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Expected Gravity Leakage (m/s²)")
plt.ylabel("Measured Horizontal Residual (m/s²)")
plt.title(
    "Measured vs Expected Gravity Leakage"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plot2 = os.path.join(
    OUTPUT_DIR,
    "orientation_vs_gravity_leakage.png"
)

plt.savefig(plot2, dpi=200)

plt.close()


# ============================================================
# PLOT 3
# ROLL / PITCH / TILT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    merged["time_seconds"],
    merged["roll_deg"],
    label="Roll"
)

plt.plot(
    merged["time_seconds"],
    merged["pitch_deg"],
    label="Pitch"
)

plt.plot(
    merged["time_seconds"],
    merged["tilt_deg"],
    label="Tilt Magnitude"
)

plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title(
    "Estimated Orientation During Strict Stationary Periods"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plot3 = os.path.join(
    OUTPUT_DIR,
    "orientation_leakage_windows.png"
)

plt.savefig(plot3, dpi=200)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 6.5 COMPLETE")
print("=" * 70)

print("\nGenerated:")
print("1.", OUTPUT_CSV)
print("2.", WINDOW_OUTPUT)
print("3.", plot1)
print("4.", plot2)
print("5.", plot3)

print("\nInterpretation:")
print(
    "A positive correlation between estimated tilt and "
    "horizontal residual would support the hypothesis that "
    "attitude-related gravity leakage contributes to the "
    "stationary acceleration residual."
)

print(
    "\nImportant: this analysis does NOT prove that orientation "
    "error is the only source of the residual."
)

print("=" * 70)