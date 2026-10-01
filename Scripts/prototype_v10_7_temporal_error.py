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
    "prototype_v10_7_temporal_error.csv"
)

EAST_PLOT = os.path.join(
    DATA,
    "prototype_v10_7_east_error.png"
)

WINDOW_PLOT = os.path.join(
    DATA,
    "prototype_v10_7_window_error.png"
)

# =========================================================
# OUTAGE
# =========================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0


print("=" * 75)
print("V10.7 TEMPORAL ERROR ANALYSIS")
print("=" * 75)


# =========================================================
# 1. LOAD REFERENCE
# =========================================================

print("\n[1] Loading reference...")

target = pd.read_csv(TARGET_FILE)

target["timestamp"] = pd.to_datetime(
    target["timestamp"]
)

target = target.sort_values(
    "timestamp"
).reset_index(drop=True)

print(
    "Total reference samples:",
    len(target)
)


# =========================================================
# 2. LOAD V10.1
# =========================================================

print("\n[2] Loading V10.1 AI velocity...")

ai = pd.read_csv(V101_FILE)

ai["timestamp"] = pd.to_datetime(
    ai["timestamp"]
)

ai = ai.sort_values(
    "timestamp"
).reset_index(drop=True)

print(
    "Total AI samples:",
    len(ai)
)


# =========================================================
# 3. MATCH REFERENCE + AI
# =========================================================

print("\n[3] Matching reference and AI...")

merged = pd.merge_asof(
    target[
        [
            "timestamp",
            "time_s",
            "vn_mps",
            "ve_mps"
        ]
    ].sort_values("timestamp"),

    ai[
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
        "WARNING: Missing AI samples found."
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
# 4. SELECT OUTAGE
# =========================================================

print("\n[4] Selecting outage...")

outage = merged[
    (merged["time_s"] >= OUTAGE_START) &
    (merged["time_s"] <= OUTAGE_END)
].copy()

outage = outage.reset_index(drop=True)

print(
    f"Outage period: "
    f"{OUTAGE_START}s -> {OUTAGE_END}s"
)

print(
    "Outage samples:",
    len(outage)
)


if len(outage) < 2:

    raise RuntimeError(
        "Not enough samples in outage."
    )


# =========================================================
# 5. VELOCITY ERRORS
# =========================================================

outage["north_error"] = (
    outage["vn_ai_mps"] -
    outage["vn_mps"]
)

outage["east_error"] = (
    outage["ve_ai_mps"] -
    outage["ve_mps"]
)


# =========================================================
# 6. SPEED
# =========================================================

outage["true_speed"] = np.sqrt(
    outage["vn_mps"] ** 2 +
    outage["ve_mps"] ** 2
)

outage["ai_speed"] = np.sqrt(
    outage["vn_ai_mps"] ** 2 +
    outage["ve_ai_mps"] ** 2
)

outage["speed_error"] = (
    outage["ai_speed"] -
    outage["true_speed"]
)


# =========================================================
# 7. MOTION CHARACTERISTICS
# =========================================================

time = outage["time_s"].to_numpy()

vn_true = outage["vn_mps"].to_numpy()
ve_true = outage["ve_mps"].to_numpy()

speed = outage["true_speed"].to_numpy()


# Change in velocity
outage["dv_n"] = outage["vn_mps"].diff()
outage["dv_e"] = outage["ve_mps"].diff()

outage["dv_magnitude"] = np.sqrt(
    outage["dv_n"] ** 2 +
    outage["dv_e"] ** 2
)


# Speed change
outage["speed_change"] = (
    outage["true_speed"].diff()
)


# Reference acceleration
outage["reference_acceleration"] = np.gradient(
    speed,
    time
)


# =========================================================
# 8. HEADING
# =========================================================

heading_rad = np.unwrap(
    np.arctan2(
        outage["ve_mps"].to_numpy(),
        outage["vn_mps"].to_numpy()
    )
)

outage["heading_deg"] = np.degrees(
    heading_rad
)


outage["heading_rate_deg_s"] = np.degrees(
    np.gradient(
        heading_rad,
        time
    )
)


# =========================================================
# 9. 10 SECOND WINDOWS
# =========================================================

print("\n[5] 10-second temporal analysis...")

WINDOW = 10.0

window_rows = []

window_start = OUTAGE_START


while window_start < OUTAGE_END:

    window_end = min(
        window_start + WINDOW,
        OUTAGE_END
    )

    w = outage[
        (outage["time_s"] >= window_start) &
        (outage["time_s"] < window_end)
    ].copy()

    if len(w) == 0:

        window_start += WINDOW
        continue


    row = {

        "window_start_s":
            window_start,

        "window_end_s":
            window_end,

        "samples":
            len(w),

        # ---------------------------------------------
        # NORTH
        # ---------------------------------------------

        "north_mae_mps":
            np.mean(
                np.abs(
                    w["north_error"]
                )
            ),

        "north_rmse_mps":
            np.sqrt(
                np.mean(
                    w["north_error"] ** 2
                )
            ),

        "north_bias_mps":
            np.mean(
                w["north_error"]
            ),

        # ---------------------------------------------
        # EAST
        # ---------------------------------------------

        "east_mae_mps":
            np.mean(
                np.abs(
                    w["east_error"]
                )
            ),

        "east_rmse_mps":
            np.sqrt(
                np.mean(
                    w["east_error"] ** 2
                )
            ),

        "east_bias_mps":
            np.mean(
                w["east_error"]
            ),

        # ---------------------------------------------
        # SPEED
        # ---------------------------------------------

        "speed_mae_mps":
            np.mean(
                np.abs(
                    w["speed_error"]
                )
            ),

        "speed_bias_mps":
            np.mean(
                w["speed_error"]
            ),

        # ---------------------------------------------
        # MOTION
        # ---------------------------------------------

        "mean_speed_mps":
            np.mean(
                w["true_speed"]
            ),

        "mean_acceleration_mps2":
            np.mean(
                np.abs(
                    w["reference_acceleration"]
                )
            ),

        "mean_heading_rate_deg_s":
            np.mean(
                np.abs(
                    w["heading_rate_deg_s"]
                )
            ),

        "mean_velocity_change_mps":
            np.mean(
                np.abs(
                    w["dv_magnitude"]
                )
            )
    }


    window_rows.append(row)

    window_start += WINDOW


windows = pd.DataFrame(
    window_rows
)


# =========================================================
# 10. PRINT WINDOWS
# =========================================================

print("\n" + "=" * 75)
print("10-SECOND WINDOW RESULTS")
print("=" * 75)

print(
    windows.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =========================================================
# 11. WORST WINDOWS
# =========================================================

print("\n" + "=" * 75)
print("KEY WINDOWS")
print("=" * 75)


worst_east = windows.loc[
    windows["east_mae_mps"].idxmax()
]

worst_north = windows.loc[
    windows["north_mae_mps"].idxmax()
]

worst_speed = windows.loc[
    windows["speed_mae_mps"].idxmax()
]

highest_turning = windows.loc[
    windows["mean_heading_rate_deg_s"].idxmax()
]


print("\nWorst East velocity error:")

print(
    f"Window: "
    f"{worst_east['window_start_s']:.0f}"
    f"-"
    f"{worst_east['window_end_s']:.0f}s"
)

print(
    f"East MAE: "
    f"{worst_east['east_mae_mps']:.4f} m/s"
)

print(
    f"East bias: "
    f"{worst_east['east_bias_mps']:.4f} m/s"
)


print("\nWorst North velocity error:")

print(
    f"Window: "
    f"{worst_north['window_start_s']:.0f}"
    f"-"
    f"{worst_north['window_end_s']:.0f}s"
)

print(
    f"North MAE: "
    f"{worst_north['north_mae_mps']:.4f} m/s"
)


print("\nWorst speed error:")

print(
    f"Window: "
    f"{worst_speed['window_start_s']:.0f}"
    f"-"
    f"{worst_speed['window_end_s']:.0f}s"
)

print(
    f"Speed MAE: "
    f"{worst_speed['speed_mae_mps']:.4f} m/s"
)


print("\nHighest heading-change window:")

print(
    f"Window: "
    f"{highest_turning['window_start_s']:.0f}"
    f"-"
    f"{highest_turning['window_end_s']:.0f}s"
)

print(
    f"Heading rate: "
    f"{highest_turning['mean_heading_rate_deg_s']:.4f} deg/s"
)


# =========================================================
# 12. CORRELATION FUNCTION
# =========================================================

def safe_corr(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    valid = (
        np.isfinite(a) &
        np.isfinite(b)
    )

    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        return np.nan

    if np.std(a) == 0:
        return np.nan

    if np.std(b) == 0:
        return np.nan

    return np.corrcoef(
        a,
        b
    )[0, 1]


# =========================================================
# 13. ERROR vs MOTION
# =========================================================

print("\n" + "=" * 75)
print("ERROR vs MOTION CORRELATION")
print("=" * 75)


east_abs = np.abs(
    outage["east_error"]
)

north_abs = np.abs(
    outage["north_error"]
)

speed_abs = np.abs(
    outage["speed_error"]
)

accel_abs = np.abs(
    outage["reference_acceleration"]
)

heading_rate_abs = np.abs(
    outage["heading_rate_deg_s"]
)

speed_values = outage[
    "true_speed"
].to_numpy()


print(
    "East error vs speed:",
    f"{safe_corr(east_abs, speed_values):.4f}"
)

print(
    "East error vs acceleration:",
    f"{safe_corr(east_abs, accel_abs):.4f}"
)

print(
    "East error vs heading rate:",
    f"{safe_corr(east_abs, heading_rate_abs):.4f}"
)

print(
    "North error vs speed:",
    f"{safe_corr(north_abs, speed_values):.4f}"
)

print(
    "North error vs acceleration:",
    f"{safe_corr(north_abs, accel_abs):.4f}"
)

print(
    "North error vs heading rate:",
    f"{safe_corr(north_abs, heading_rate_abs):.4f}"
)

print(
    "Speed error vs acceleration:",
    f"{safe_corr(speed_abs, accel_abs):.4f}"
)


# =========================================================
# 14. SAVE WINDOW RESULTS
# =========================================================

windows.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)


# =========================================================
# 15. EAST ERROR PLOT
# =========================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    outage["time_s"],
    outage["east_error"],
    label="East velocity error"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "East velocity error (m/s)"
)

plt.title(
    "V10.7 East Velocity Error During GNSS Outage"
)

plt.legend()
plt.grid(True)

plt.savefig(
    EAST_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 16. WINDOW ERROR PLOT
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    windows["window_start_s"],
    windows["east_mae_mps"],
    marker="o",
    label="East MAE"
)

plt.plot(
    windows["window_start_s"],
    windows["north_mae_mps"],
    marker="o",
    label="North MAE"
)

plt.xlabel(
    "Window start time (s)"
)

plt.ylabel(
    "Velocity MAE (m/s)"
)

plt.title(
    "V10.7 Temporal Velocity Error"
)

plt.legend()
plt.grid(True)

plt.savefig(
    WINDOW_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("\nSaved plots:")
print(EAST_PLOT)
print(WINDOW_PLOT)


# =========================================================
# 17. COMPLETE
# =========================================================

print("\n" + "=" * 75)
print("V10.7 COMPLETE")
print("=" * 75)