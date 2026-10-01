import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")

TARGET_FILE = os.path.join(
    DATA, "prototype_v9_velocity_targets.csv"
)

V93_FILE = os.path.join(
    DATA, "prototype_v9_ai_velocity_results.csv"
)

V101_FILE = os.path.join(
    DATA, "prototype_v10_1_full_ai_velocity.csv"
)

START = 8500.0
END = 8560.0


print("=" * 75)
print("V10.5 VELOCITY DIAGNOSTICS")
print("=" * 75)


# =========================================================
# 1. LOAD REFERENCE
# =========================================================

print("\n[1] Loading reference...")

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
# 2. COMMON TIMESTAMPS
# =========================================================

target_ts = target[
    ["timestamp", "time_s"]
].copy()


# =========================================================
# 3. LOAD V9.3
# =========================================================

print("\n[2] Loading V9.3...")

v93 = pd.read_csv(V93_FILE)

v93["timestamp"] = pd.to_datetime(
    v93["timestamp"]
)

v93 = v93.sort_values("timestamp")

v93_match = pd.merge_asof(
    target_ts.sort_values("timestamp"),
    v93[
        [
            "timestamp",
            "vnorth_ai_mps",
            "veast_ai_mps",
            "true_speed_mps",
            "ai_speed_mps"
        ]
    ].sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("50ms")
)

print(
    "V9.3 matched:",
    v93_match["vnorth_ai_mps"].notna().sum()
)


# =========================================================
# 4. LOAD V10.1
# =========================================================

print("\n[3] Loading V10.1...")

v101 = pd.read_csv(V101_FILE)

v101["timestamp"] = pd.to_datetime(
    v101["timestamp"]
)

v101 = v101.sort_values("timestamp")

v101_match = pd.merge_asof(
    target_ts.sort_values("timestamp"),
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
    "V10.1 matched:",
    v101_match["vn_ai_mps"].notna().sum()
)


# =========================================================
# 5. BUILD DIAGNOSTIC DATAFRAME
# =========================================================

diag = pd.DataFrame({
    "time_s": target["time_s"].to_numpy(),

    "vn_true": target["vn_mps"].to_numpy(),
    "ve_true": target["ve_mps"].to_numpy(),

    "v9_vn": v93_match["vnorth_ai_mps"].to_numpy(),
    "v9_ve": v93_match["veast_ai_mps"].to_numpy(),

    "v10_vn": v101_match["vn_ai_mps"].to_numpy(),
    "v10_ve": v101_match["ve_ai_mps"].to_numpy()
})


# =========================================================
# 6. VELOCITY ERRORS
# =========================================================

diag["v9_n_error"] = (
    diag["v9_vn"] -
    diag["vn_true"]
)

diag["v9_e_error"] = (
    diag["v9_ve"] -
    diag["ve_true"]
)

diag["v10_n_error"] = (
    diag["v10_vn"] -
    diag["vn_true"]
)

diag["v10_e_error"] = (
    diag["v10_ve"] -
    diag["ve_true"]
)


# =========================================================
# 7. SPEED
# =========================================================

diag["true_speed"] = np.sqrt(
    diag["vn_true"] ** 2 +
    diag["ve_true"] ** 2
)

diag["v9_speed"] = np.sqrt(
    diag["v9_vn"] ** 2 +
    diag["v9_ve"] ** 2
)

diag["v10_speed"] = np.sqrt(
    diag["v10_vn"] ** 2 +
    diag["v10_ve"] ** 2
)


diag["v9_speed_error"] = (
    diag["v9_speed"] -
    diag["true_speed"]
)

diag["v10_speed_error"] = (
    diag["v10_speed"] -
    diag["true_speed"]
)


# =========================================================
# 8. METRIC FUNCTION
# =========================================================

def mae(x):
    return np.mean(np.abs(x))


def rmse(x):
    return np.sqrt(np.mean(x ** 2))


def bias(x):
    return np.mean(x)


def correlation(a, b):

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


# =========================================================
# 9. PRINT METRICS
# =========================================================

print("\n" + "=" * 75)
print("VELOCITY METRICS")
print("=" * 75)


print("\nV9.3 NORTH")

print(
    f"MAE        : {mae(diag['v9_n_error']):.4f} m/s"
)

print(
    f"RMSE       : {rmse(diag['v9_n_error']):.4f} m/s"
)

print(
    f"Bias       : {bias(diag['v9_n_error']):.4f} m/s"
)

print(
    f"Correlation: "
    f"{correlation(diag['vn_true'], diag['v9_vn']):.4f}"
)


print("\nV9.3 EAST")

print(
    f"MAE        : {mae(diag['v9_e_error']):.4f} m/s"
)

print(
    f"RMSE       : {rmse(diag['v9_e_error']):.4f} m/s"
)

print(
    f"Bias       : {bias(diag['v9_e_error']):.4f} m/s"
)

print(
    f"Correlation: "
    f"{correlation(diag['ve_true'], diag['v9_ve']):.4f}"
)


print("\nV10.1 NORTH")

print(
    f"MAE        : {mae(diag['v10_n_error']):.4f} m/s"
)

print(
    f"RMSE       : {rmse(diag['v10_n_error']):.4f} m/s"
)

print(
    f"Bias       : {bias(diag['v10_n_error']):.4f} m/s"
)

print(
    f"Correlation: "
    f"{correlation(diag['vn_true'], diag['v10_vn']):.4f}"
)


print("\nV10.1 EAST")

print(
    f"MAE        : {mae(diag['v10_e_error']):.4f} m/s"
)

print(
    f"RMSE       : {rmse(diag['v10_e_error']):.4f} m/s"
)

print(
    f"Bias       : {bias(diag['v10_e_error']):.4f} m/s"
)

print(
    f"Correlation: "
    f"{correlation(diag['ve_true'], diag['v10_ve']):.4f}"
)


# =========================================================
# 10. SPEED METRICS
# =========================================================

print("\n" + "=" * 75)
print("SPEED METRICS")
print("=" * 75)


print("\nV9.3 SPEED")

print(
    f"MAE  : {mae(diag['v9_speed_error']):.4f} m/s"
)

print(
    f"RMSE : {rmse(diag['v9_speed_error']):.4f} m/s"
)


print("\nV10.1 SPEED")

print(
    f"MAE  : {mae(diag['v10_speed_error']):.4f} m/s"
)

print(
    f"RMSE : {rmse(diag['v10_speed_error']):.4f} m/s"
)


# =========================================================
# 11. CUMULATIVE VELOCITY ERROR
# =========================================================

print("\n" + "=" * 75)
print("CUMULATIVE VELOCITY ERROR")
print("=" * 75)


time = diag["time_s"].to_numpy()


v9_n_cumulative = np.zeros(len(diag))
v9_e_cumulative = np.zeros(len(diag))

v10_n_cumulative = np.zeros(len(diag))
v10_e_cumulative = np.zeros(len(diag))


for i in range(1, len(diag)):

    dt = time[i] - time[i - 1]

    v9_n_cumulative[i] = (
        v9_n_cumulative[i - 1]
        + 0.5 * (
            diag["v9_n_error"].iloc[i - 1]
            + diag["v9_n_error"].iloc[i]
        ) * dt
    )

    v9_e_cumulative[i] = (
        v9_e_cumulative[i - 1]
        + 0.5 * (
            diag["v9_e_error"].iloc[i - 1]
            + diag["v9_e_error"].iloc[i]
        ) * dt
    )

    v10_n_cumulative[i] = (
        v10_n_cumulative[i - 1]
        + 0.5 * (
            diag["v10_n_error"].iloc[i - 1]
            + diag["v10_n_error"].iloc[i]
        ) * dt
    )

    v10_e_cumulative[i] = (
        v10_e_cumulative[i - 1]
        + 0.5 * (
            diag["v10_e_error"].iloc[i - 1]
            + diag["v10_e_error"].iloc[i]
        ) * dt
    )


v9_final_cumulative = np.sqrt(
    v9_n_cumulative[-1] ** 2 +
    v9_e_cumulative[-1] ** 2
)

v10_final_cumulative = np.sqrt(
    v10_n_cumulative[-1] ** 2 +
    v10_e_cumulative[-1] ** 2
)


print(
    f"V9.3 cumulative velocity error: "
    f"{v9_final_cumulative:.3f} m"
)

print(
    f"V10.1 cumulative velocity error: "
    f"{v10_final_cumulative:.3f} m"
)


# =========================================================
# 12. SAVE DIAGNOSTIC DATA
# =========================================================

output_file = os.path.join(
    DATA,
    "prototype_v10_5_velocity_diagnostics.csv"
)

diag.to_csv(
    output_file,
    index=False
)

print("\nSaved diagnostic CSV:")
print(output_file)


# =========================================================
# 13. PLOT VELOCITY
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    diag["vn_true"],
    label="True North velocity"
)

plt.plot(
    time,
    diag["v9_vn"],
    label="V9.3 North"
)

plt.plot(
    time,
    diag["v10_vn"],
    label="V10.1 North"
)

plt.xlabel("Time (s)")
plt.ylabel("North velocity (m/s)")
plt.title("North Velocity Comparison")

plt.legend()
plt.grid(True)

north_plot = os.path.join(
    DATA,
    "prototype_v10_5_north_velocity.png"
)

plt.savefig(
    north_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 14. EAST VELOCITY PLOT
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    diag["ve_true"],
    label="True East velocity"
)

plt.plot(
    time,
    diag["v9_ve"],
    label="V9.3 East"
)

plt.plot(
    time,
    diag["v10_ve"],
    label="V10.1 East"
)

plt.xlabel("Time (s)")
plt.ylabel("East velocity (m/s)")
plt.title("East Velocity Comparison")

plt.legend()
plt.grid(True)

east_plot = os.path.join(
    DATA,
    "prototype_v10_5_east_velocity.png"
)

plt.savefig(
    east_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 15. ERROR PLOT
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    diag["v9_n_error"],
    label="V9.3 North error"
)

plt.plot(
    time,
    diag["v10_n_error"],
    label="V10.1 North error"
)

plt.plot(
    time,
    diag["v9_e_error"],
    label="V9.3 East error"
)

plt.plot(
    time,
    diag["v10_e_error"],
    label="V10.1 East error"
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity error (m/s)")
plt.title("Velocity Prediction Error")

plt.legend()
plt.grid(True)

error_plot = os.path.join(
    DATA,
    "prototype_v10_5_velocity_error.png"
)

plt.savefig(
    error_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("\nSaved plots:")
print(north_plot)
print(east_plot)
print(error_plot)


print("\n" + "=" * 75)
print("V10.5 COMPLETE")
print("=" * 75)