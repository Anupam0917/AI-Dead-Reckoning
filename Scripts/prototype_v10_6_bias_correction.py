import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PATHS
# =========================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")

TARGET_FILE = os.path.join(
    DATA,
    "prototype_v9_velocity_targets.csv"
)

V101_FILE = os.path.join(
    DATA,
    "prototype_v10_1_full_ai_velocity.csv"
)

OUTPUT_FILE = os.path.join(
    DATA,
    "prototype_v10_6_bias_correction_results.csv"
)

PLOT_FILE = os.path.join(
    DATA,
    "prototype_v10_6_bias_correction.png"
)


# =========================================================
# CONFIGURATION
# =========================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

# Use ONLY data before the outage to estimate bias.
CALIBRATION_START = 1000.0
CALIBRATION_END = 8400.0


print("=" * 75)
print("V10.6 BIAS-CORRECTED AI VELOCITY")
print("=" * 75)


# =========================================================
# 1. LOAD TARGET
# =========================================================

print("\n[1] Loading GNSS-derived velocity reference...")

target_full = pd.read_csv(TARGET_FILE)

target_full["timestamp"] = pd.to_datetime(
    target_full["timestamp"]
)

target_full = target_full.sort_values(
    "timestamp"
).reset_index(drop=True)

print(
    "Total target samples:",
    len(target_full)
)


# =========================================================
# 2. LOAD V10.1
# =========================================================

print("\n[2] Loading V10.1 AI velocity...")

v101 = pd.read_csv(V101_FILE)

v101["timestamp"] = pd.to_datetime(
    v101["timestamp"]
)

v101 = v101.sort_values(
    "timestamp"
).reset_index(drop=True)

print(
    "Total V10.1 samples:",
    len(v101)
)


# =========================================================
# 3. MATCH AI WITH REFERENCE
# =========================================================

print("\n[3] Matching AI velocity with reference...")

merged = pd.merge_asof(
    target_full[
        [
            "timestamp",
            "time_s",
            "vn_mps",
            "ve_mps"
        ]
    ].sort_values("timestamp"),

    v101[
        [
            "timestamp",
            "vn_ai_mps",
            "ve_ai_mps"
        ]
    ].sort_values("timestamp"),

    on="timestamp",

    direction="nearest",

    tolerance=pd.Timedelta("50ms")
)

matched = merged["vn_ai_mps"].notna().sum()

print(
    "Matched samples:",
    matched,
    "/",
    len(merged)
)

if matched < len(merged):
    print(
        "WARNING: Some samples were not matched."
    )

    merged["vn_ai_mps"] = (
        merged["vn_ai_mps"]
        .interpolate()
        .bfill()
        .ffill()
    )

    merged["ve_ai_mps"] = (
        merged["ve_ai_mps"]
        .interpolate()
        .bfill()
        .ffill()
    )


# =========================================================
# 4. CALIBRATION PERIOD
# =========================================================

print("\n[4] Estimating bias from PRE-OUTAGE data only")

calibration = merged[
    (merged["time_s"] >= CALIBRATION_START) &
    (merged["time_s"] <= CALIBRATION_END)
].copy()

print(
    f"Calibration period: "
    f"{CALIBRATION_START}s -> {CALIBRATION_END}s"
)

print(
    "Calibration samples:",
    len(calibration)
)


# AI error = AI prediction - reference
calibration["north_error"] = (
    calibration["vn_ai_mps"] -
    calibration["vn_mps"]
)

calibration["east_error"] = (
    calibration["ve_ai_mps"] -
    calibration["ve_mps"]
)


north_bias = calibration["north_error"].mean()
east_bias = calibration["east_error"].mean()


print("\nEstimated calibration bias:")

print(
    f"North bias: {north_bias:.6f} m/s"
)

print(
    f"East bias : {east_bias:.6f} m/s"
)


# =========================================================
# 5. OUTAGE DATA
# =========================================================

print("\n[5] Selecting GNSS outage")

outage = merged[
    (merged["time_s"] >= OUTAGE_START) &
    (merged["time_s"] <= OUTAGE_END)
].copy()

outage = outage.reset_index(drop=True)

print(
    "Outage samples:",
    len(outage)
)

if len(outage) < 2:
    raise RuntimeError(
        "Not enough outage samples."
    )


# =========================================================
# 6. COMMON REFERENCE TRAJECTORY
# =========================================================

time = outage["time_s"].to_numpy()

vn_true = outage["vn_mps"].to_numpy()
ve_true = outage["ve_mps"].to_numpy()


ref_n = np.zeros(len(outage))
ref_e = np.zeros(len(outage))


for i in range(1, len(outage)):

    dt = time[i] - time[i - 1]

    ref_n[i] = (
        ref_n[i - 1]
        + 0.5 *
        (
            vn_true[i - 1]
            + vn_true[i]
        ) *
        dt
    )

    ref_e[i] = (
        ref_e[i - 1]
        + 0.5 *
        (
            ve_true[i - 1]
            + ve_true[i]
        ) *
        dt
    )


reference_distance = np.sum(
    np.sqrt(
        np.diff(ref_n) ** 2 +
        np.diff(ref_e) ** 2
    )
)

print(
    f"Reference distance: "
    f"{reference_distance:.3f} m"
)


# =========================================================
# 7. ORIGINAL AI VELOCITY
# =========================================================

vn_original = outage[
    "vn_ai_mps"
].to_numpy()

ve_original = outage[
    "ve_ai_mps"
].to_numpy()


# =========================================================
# 8. EAST-ONLY CORRECTION
# =========================================================

vn_east_corrected = vn_original.copy()

ve_east_corrected = (
    ve_original -
    east_bias
)


# =========================================================
# 9. NORTH + EAST CORRECTION
# =========================================================

vn_full_corrected = (
    vn_original -
    north_bias
)

ve_full_corrected = (
    ve_original -
    east_bias
)


# =========================================================
# 10. TRAJECTORY FUNCTION
# =========================================================

def integrate_velocity(
    vn,
    ve
):

    nav_n = np.zeros(len(time))
    nav_e = np.zeros(len(time))

    for i in range(1, len(time)):

        dt = time[i] - time[i - 1]

        nav_n[i] = (
            nav_n[i - 1]
            + 0.5 *
            (
                vn[i - 1]
                + vn[i]
            ) *
            dt
        )

        nav_e[i] = (
            nav_e[i - 1]
            + 0.5 *
            (
                ve[i - 1]
                + ve[i]
            ) *
            dt
        )

    return nav_n, nav_e


# =========================================================
# 11. METRIC FUNCTION
# =========================================================

def calculate_metrics(
    name,
    vn,
    ve
):

    nav_n, nav_e = integrate_velocity(
        vn,
        ve
    )

    error = np.sqrt(
        (nav_n - ref_n) ** 2 +
        (nav_e - ref_e) ** 2
    )

    final_error = error[-1]
    mean_error = np.mean(error)
    max_error = np.max(error)

    model_distance = np.sum(
        np.sqrt(
            np.diff(nav_n) ** 2 +
            np.diff(nav_e) ** 2
        )
    )

    drift = (
        final_error /
        reference_distance
    ) * 100

    print("\n" + "-" * 65)
    print(name)
    print("-" * 65)

    print(
        f"Final error   : {final_error:.3f} m"
    )

    print(
        f"Mean error    : {mean_error:.3f} m"
    )

    print(
        f"Maximum error : {max_error:.3f} m"
    )

    print(
        f"Model path    : {model_distance:.3f} m"
    )

    print(
        f"Drift         : {drift:.3f} %"
    )

    return {
        "model": name,
        "final_error_m": final_error,
        "mean_error_m": mean_error,
        "max_error_m": max_error,
        "model_distance_m": model_distance,
        "reference_distance_m": reference_distance,
        "drift_percent": drift
    }, nav_n, nav_e


# =========================================================
# 12. RUN THREE VARIANTS
# =========================================================

print("\n[6] Evaluating models...")


result_original, n_original, e_original = (
    calculate_metrics(
        "V10.1 Original",
        vn_original,
        ve_original
    )
)


result_east, n_east, e_east = (
    calculate_metrics(
        "V10.1 East-Bias Corrected",
        vn_east_corrected,
        ve_east_corrected
    )
)


result_full, n_full, e_full = (
    calculate_metrics(
        "V10.1 North+East Bias Corrected",
        vn_full_corrected,
        ve_full_corrected
    )
)


# =========================================================
# 13. COMPARISON
# =========================================================

results = pd.DataFrame([
    result_original,
    result_east,
    result_full
])


print("\n")
print("=" * 75)
print("V10.6 COMPARISON")
print("=" * 75)

print(
    results.to_string(
        index=False
    )
)


# =========================================================
# 14. SAVE RESULTS
# =========================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)


# =========================================================
# 15. SAVE BIAS INFORMATION
# =========================================================

bias_file = os.path.join(
    DATA,
    "prototype_v10_6_calibration_bias.csv"
)

bias_df = pd.DataFrame({

    "sensor": [
        "north_velocity",
        "east_velocity"
    ],

    "bias_mps": [
        north_bias,
        east_bias
    ],

    "calibration_start_s": [
        CALIBRATION_START,
        CALIBRATION_START
    ],

    "calibration_end_s": [
        CALIBRATION_END,
        CALIBRATION_END
    ]

})

bias_df.to_csv(
    bias_file,
    index=False
)

print("Saved:")
print(bias_file)


# =========================================================
# 16. SAVE TRAJECTORIES
# =========================================================

trajectory = pd.DataFrame({

    "time_s": time,

    "reference_north_m": ref_n,
    "reference_east_m": ref_e,

    "original_north_m": n_original,
    "original_east_m": e_original,

    "east_corrected_north_m": n_east,
    "east_corrected_east_m": e_east,

    "full_corrected_north_m": n_full,
    "full_corrected_east_m": e_full
})


trajectory_file = os.path.join(
    DATA,
    "prototype_v10_6_trajectories.csv"
)

trajectory.to_csv(
    trajectory_file,
    index=False
)

print("Saved:")
print(trajectory_file)


# =========================================================
# 17. PLOT TRAJECTORIES
# =========================================================

plt.figure(figsize=(10, 7))

plt.plot(
    ref_e,
    ref_n,
    label="GNSS-derived reference"
)

plt.plot(
    e_original,
    n_original,
    label="V10.1 Original"
)

plt.plot(
    e_east,
    n_east,
    label="East-bias corrected"
)

plt.plot(
    e_full,
    n_full,
    label="North+East corrected"
)

plt.xlabel(
    "East displacement (m)"
)

plt.ylabel(
    "North displacement (m)"
)

plt.title(
    "V10.6 Bias Correction - 60 s GNSS Outage"
)

plt.legend()
plt.grid(True)

plt.savefig(
    PLOT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(PLOT_FILE)


# =========================================================
# 18. FINAL
# =========================================================

print("\n" + "=" * 75)
print("V10.6 COMPLETE")
print("=" * 75)

print(
    "\nBias was estimated ONLY from "
    f"{CALIBRATION_START}s to {CALIBRATION_END}s."
)

print(
    "The 8500-8560s outage was NOT used "
    "to calculate the correction."
)

print(
    "\nNext decision will depend on whether "
    "East-bias correction reduces the "
    "43.27 m V10.1 error."
)