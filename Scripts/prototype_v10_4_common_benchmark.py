import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")

TARGET_FILE = os.path.join(
    DATA, "prototype_v9_velocity_targets.csv"
)

V101_FILE = os.path.join(
    DATA, "prototype_v10_1_full_ai_velocity.csv"
)

V93_FILE = os.path.join(
    DATA, "prototype_v9_ai_velocity_results.csv"
)

START = 8500.0
END = 8560.0


print("=" * 70)
print("V10.4 COMMON BENCHMARK")
print("=" * 70)


# =========================================================
# 1. LOAD TARGET
# =========================================================

print("\n[1] Loading common reference...")

target_full = pd.read_csv(TARGET_FILE)

target_full["timestamp"] = pd.to_datetime(
    target_full["timestamp"]
)

target = target_full[
    (target_full["time_s"] >= START) &
    (target_full["time_s"] <= END)
].copy()

target = target.reset_index(drop=True)

print("Reference samples:", len(target))


# =========================================================
# 2. BUILD REFERENCE TRAJECTORY
# =========================================================

time = target["time_s"].to_numpy()

vn_ref = target["vn_mps"].to_numpy()
ve_ref = target["ve_mps"].to_numpy()

ref_n = np.zeros(len(target))
ref_e = np.zeros(len(target))

for i in range(1, len(target)):

    dt = time[i] - time[i - 1]

    ref_n[i] = (
        ref_n[i - 1]
        + 0.5 * (vn_ref[i - 1] + vn_ref[i]) * dt
    )

    ref_e[i] = (
        ref_e[i - 1]
        + 0.5 * (ve_ref[i - 1] + ve_ref[i]) * dt
    )


reference_distance = np.sum(
    np.sqrt(
        np.diff(ref_n) ** 2 +
        np.diff(ref_e) ** 2
    )
)

print("\nReference trajectory:")
print(f"North displacement : {ref_n[-1]:.3f} m")
print(f"East displacement  : {ref_e[-1]:.3f} m")
print(f"Path distance      : {reference_distance:.3f} m")


# =========================================================
# 3. EVALUATION FUNCTION
# =========================================================

def evaluate(name, vn, ve):

    vn = np.asarray(vn, dtype=float)
    ve = np.asarray(ve, dtype=float)

    if len(vn) != len(time):
        raise ValueError(
            f"{name}: expected {len(time)} samples, "
            f"got {len(vn)}"
        )

    nav_n = np.zeros(len(time))
    nav_e = np.zeros(len(time))

    for i in range(1, len(time)):

        dt = time[i] - time[i - 1]

        nav_n[i] = (
            nav_n[i - 1]
            + 0.5 * (vn[i - 1] + vn[i]) * dt
        )

        nav_e[i] = (
            nav_e[i - 1]
            + 0.5 * (ve[i - 1] + ve[i]) * dt
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

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"Final error   : {final_error:.3f} m")
    print(f"Mean error    : {mean_error:.3f} m")
    print(f"Maximum error : {max_error:.3f} m")
    print(f"Model path    : {model_distance:.3f} m")
    print(f"Reference     : {reference_distance:.3f} m")
    print(f"Drift         : {drift:.3f} %")

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
# 4. V10.1
# =========================================================

print("\n[2] Loading V10.1...")

v101 = pd.read_csv(V101_FILE)

v101["timestamp"] = pd.to_datetime(
    v101["timestamp"]
)

print("V10.1 rows:", len(v101))


# ---------------------------------------------------------
# Match V10.1 using nearest timestamp
# ---------------------------------------------------------

v101 = v101.sort_values("timestamp")

target_for_merge = target[
    ["timestamp", "time_s"]
].copy()

v101_match = pd.merge_asof(
    target_for_merge.sort_values("timestamp"),
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

print(
    "V10.1 matched samples:",
    v101_match["vn_ai_mps"].notna().sum()
)

if v101_match["vn_ai_mps"].isna().any():

    print("WARNING: Some V10.1 samples were not matched.")

    v101_match["vn_ai_mps"] = (
        v101_match["vn_ai_mps"]
        .interpolate()
        .bfill()
        .ffill()
    )

    v101_match["ve_ai_mps"] = (
        v101_match["ve_ai_mps"]
        .interpolate()
        .bfill()
        .ffill()
    )


result101, n101, e101 = evaluate(
    "V10.1 Full AI Velocity",
    v101_match["vn_ai_mps"].to_numpy(),
    v101_match["ve_ai_mps"].to_numpy()
)


# =========================================================
# 5. V9.3
# =========================================================

print("\n[3] Loading V9.3...")

v93 = pd.read_csv(V93_FILE)

v93["timestamp"] = pd.to_datetime(
    v93["timestamp"]
)

print("V9.3 rows:", len(v93))

print("\nV9.3 velocity columns:")
print("vnorth_ai_mps")
print("veast_ai_mps")


# ---------------------------------------------------------
# Match V9.3 using nearest timestamp
# ---------------------------------------------------------

v93 = v93.sort_values("timestamp")

v93_match = pd.merge_asof(
    target_for_merge.sort_values("timestamp"),
    v93[
        [
            "timestamp",
            "vnorth_ai_mps",
            "veast_ai_mps"
        ]
    ].sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("50ms")
)

print(
    "V9.3 matched samples:",
    v93_match["vnorth_ai_mps"].notna().sum()
)

if v93_match["vnorth_ai_mps"].isna().any():

    print("WARNING: Some V9.3 samples were not matched.")

    v93_match["vnorth_ai_mps"] = (
        v93_match["vnorth_ai_mps"]
        .interpolate()
        .bfill()
        .ffill()
    )

    v93_match["veast_ai_mps"] = (
        v93_match["veast_ai_mps"]
        .interpolate()
        .bfill()
        .ffill()
    )


result93, n93, e93 = evaluate(
    "V9.3 AI Velocity",
    v93_match["vnorth_ai_mps"].to_numpy(),
    v93_match["veast_ai_mps"].to_numpy()
)


# =========================================================
# 6. FINAL COMPARISON
# =========================================================

results = pd.DataFrame([
    result93,
    result101
])

print("\n")
print("=" * 70)
print("FINAL COMMON BENCHMARK")
print("=" * 70)

print(
    results.to_string(index=False)
)


# =========================================================
# 7. SAVE RESULTS
# =========================================================

result_file = os.path.join(
    DATA,
    "prototype_v10_4_common_benchmark.csv"
)

results.to_csv(
    result_file,
    index=False
)

print("\nSaved:")
print(result_file)


# =========================================================
# 8. SAVE TRAJECTORIES
# =========================================================

trajectory = pd.DataFrame({

    "time_s": time,

    "reference_north_m": ref_n,
    "reference_east_m": ref_e,

    "v9_3_north_m": n93,
    "v9_3_east_m": e93,

    "v10_1_north_m": n101,
    "v10_1_east_m": e101
})

trajectory_file = os.path.join(
    DATA,
    "prototype_v10_4_trajectories.csv"
)

trajectory.to_csv(
    trajectory_file,
    index=False
)

print("Saved:")
print(trajectory_file)


# =========================================================
# 9. PLOT
# =========================================================

plt.figure(figsize=(10, 7))

plt.plot(
    ref_e,
    ref_n,
    label="GNSS-derived reference"
)

plt.plot(
    e93,
    n93,
    label="V9.3 AI"
)

plt.plot(
    e101,
    n101,
    label="V10.1 AI"
)

plt.xlabel("East displacement (m)")
plt.ylabel("North displacement (m)")

plt.title(
    "V10.4 Common Benchmark - 60 Second GNSS Outage"
)

plt.legend()
plt.grid(True)

plot_file = os.path.join(
    DATA,
    "prototype_v10_4_common_benchmark.png"
)

plt.savefig(
    plot_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(plot_file)


print("\n" + "=" * 70)
print("V10.4 COMPLETE")
print("=" * 70)