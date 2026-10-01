"""
V9.8-C
Uncertainty-Aware AI Dead Reckoning

Uses:
    V9.8-B AI velocity
    V9.8-B ensemble uncertainty
    Canonical GNSS reference

During GNSS outage:
    AI velocity is weighted according to its uncertainty.

Outside outage:
    GNSS position is used to keep the navigation state
    synchronized.

Evaluation:
    8500 -> 8560 seconds
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

AI_FILE = (
    ROOT
    / "data"
    / "processed"
    / "prototype_v9_8_uncertainty_ai_results.csv"
)

REFERENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "canonical_gnss_reference.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)

PLOT_DIR = (
    ROOT
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

EARTH_RADIUS = 6371000.0

# Reference velocity blending.
#
# Lower value:
#   smoother but slower correction
#
# Higher value:
#   faster correction but more sensitive to noise
GNSS_POSITION_GAIN = 0.80

GNSS_VELOCITY_GAIN = 0.25

# AI uncertainty scale.
#
# Around 0.7-1.0 m/s is typical in the current model.
UNCERTAINTY_SCALE = 1.0

# Limits for velocity.
MAX_SPEED_MPS = 20.0


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("V9.8-C UNCERTAINTY-AWARE DEAD RECKONING")
print("=" * 80)

print("\nLoading AI predictions...")

ai = pd.read_csv(
    AI_FILE
)

ai["timestamp"] = pd.to_datetime(
    ai["timestamp"],
    errors="coerce",
)

print(
    "AI rows:",
    len(ai)
)


print("\nLoading canonical GNSS reference...")

ref = pd.read_csv(
    REFERENCE_FILE
)

ref["timestamp"] = pd.to_datetime(
    ref["timestamp"],
    errors="coerce",
)

print(
    "Reference rows:",
    len(ref)
)


# ============================================================
# MERGE
# ============================================================

df = pd.merge_asof(
    ref.sort_values("timestamp"),
    ai.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(
        milliseconds=60
    ),
)

print(
    "\nMerged rows:",
    len(df)
)


# ============================================================
# TIME
# ============================================================

df["time_s"] = (
    df["timestamp"]
    -
    df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# AI VELOCITY
# ============================================================

ai_vn = pd.to_numeric(
    df["vn_ai_mps"],
    errors="coerce",
).to_numpy()

ai_ve = pd.to_numeric(
    df["ve_ai_mps"],
    errors="coerce",
).to_numpy()

vn_std = pd.to_numeric(
    df["vn_std_mps"],
    errors="coerce",
).to_numpy()

ve_std = pd.to_numeric(
    df["ve_std_mps"],
    errors="coerce",
).to_numpy()


# ============================================================
# REFERENCE POSITION
# ============================================================

ref_n = pd.to_numeric(
    df["north_m"],
    errors="coerce",
).to_numpy()

ref_e = pd.to_numeric(
    df["east_m"],
    errors="coerce",
).to_numpy()


# ============================================================
# INITIALIZE NAVIGATION
# ============================================================

nav_n = np.zeros(
    len(df)
)

nav_e = np.zeros(
    len(df)
)

fused_vn = np.zeros(
    len(df)
)

fused_ve = np.zeros(
    len(df)
)

confidence = np.zeros(
    len(df)
)


# Start at true GNSS position.

nav_n[0] = ref_n[0]
nav_e[0] = ref_e[0]


# ============================================================
# HELPER: AI WEIGHT
# ============================================================

def calculate_weight(
    north_std,
    east_std,
):
    """
    Convert ensemble uncertainty into a smooth
    AI velocity weight.

    Low uncertainty -> weight near 1
    High uncertainty -> lower weight
    """

    if (
        not np.isfinite(north_std)
        or
        not np.isfinite(east_std)
    ):
        return 0.0

    uncertainty = np.sqrt(
        north_std ** 2
        +
        east_std ** 2
    )

    weight = np.exp(
        -uncertainty
        /
        UNCERTAINTY_SCALE
    )

    return float(
        np.clip(
            weight,
            0.10,
            1.00,
        )
    )


# ============================================================
# NAVIGATION LOOP
# ============================================================

print("\nRunning navigation...")

for i in range(1, len(df)):

    t = df["time_s"].iloc[i]

    previous_vn = fused_vn[i - 1]
    previous_ve = fused_ve[i - 1]

    # --------------------------------------------------------
    # Time step
    # --------------------------------------------------------

    dt = (
        df["time_s"].iloc[i]
        -
        df["time_s"].iloc[i - 1]
    )

    if (
        not np.isfinite(dt)
        or dt <= 0
        or dt > 1.0
    ):
        dt = 0.1

    # --------------------------------------------------------
    # AI velocity
    # --------------------------------------------------------

    if (
        np.isfinite(ai_vn[i])
        and
        np.isfinite(ai_ve[i])
    ):

        weight = calculate_weight(
            vn_std[i],
            ve_std[i],
        )

        # ----------------------------------------------------
        # Adaptive AI fusion
        # ----------------------------------------------------

        current_vn = (
            weight * ai_vn[i]
            +
            (1.0 - weight)
            *
            previous_vn
        )

        current_ve = (
            weight * ai_ve[i]
            +
            (1.0 - weight)
            *
            previous_ve
        )

    else:

        weight = 0.0

        current_vn = previous_vn
        current_ve = previous_ve

    # --------------------------------------------------------
    # Speed constraint
    # --------------------------------------------------------

    speed = np.sqrt(
        current_vn ** 2
        +
        current_ve ** 2
    )

    if speed > MAX_SPEED_MPS:

        scale = (
            MAX_SPEED_MPS
            /
            speed
        )

        current_vn *= scale
        current_ve *= scale

    fused_vn[i] = current_vn
    fused_ve[i] = current_ve

    confidence[i] = weight

    # --------------------------------------------------------
    # GNSS availability
    # --------------------------------------------------------

    gnss_available = not (
        OUTAGE_START
        <= t
        <= OUTAGE_END
    )

    if gnss_available:

        # ----------------------------------------------------
        # GNSS position correction
        # ----------------------------------------------------

        nav_n[i] = (
            ref_n[i]
            +
            (
                nav_n[i - 1]
                -
                ref_n[i]
            )
            *
            (
                1.0
                -
                GNSS_POSITION_GAIN
            )
        )

        nav_e[i] = (
            ref_e[i]
            +
            (
                nav_e[i - 1]
                -
                ref_e[i]
            )
            *
            (
                1.0
                -
                GNSS_POSITION_GAIN
            )
        )

        # ----------------------------------------------------
        # GNSS velocity correction
        # ----------------------------------------------------

        ref_vn = (
            ref_n[i]
            -
            ref_n[i - 1]
        ) / dt

        ref_ve = (
            ref_e[i]
            -
            ref_e[i - 1]
        ) / dt

        # Avoid unrealistic GPS coordinate jumps
        # entering the velocity state.

        ref_speed = np.sqrt(
            ref_vn ** 2
            +
            ref_ve ** 2
        )

        if ref_speed < 20.0:

            fused_vn[i] = (
                (
                    1
                    -
                    GNSS_VELOCITY_GAIN
                )
                *
                fused_vn[i]
                +
                GNSS_VELOCITY_GAIN
                *
                ref_vn
            )

            fused_ve[i] = (
                (
                    1
                    -
                    GNSS_VELOCITY_GAIN
                )
                *
                fused_ve[i]
                +
                GNSS_VELOCITY_GAIN
                *
                ref_ve
            )

    else:

        # ----------------------------------------------------
        # GNSS DENIED
        # ----------------------------------------------------

        nav_n[i] = (
            nav_n[i - 1]
            +
            fused_vn[i] * dt
        )

        nav_e[i] = (
            nav_e[i - 1]
            +
            fused_ve[i] * dt
        )


# ============================================================
# ERROR
# ============================================================

position_error = np.sqrt(
    (
        nav_n
        -
        ref_n
    ) ** 2
    +
    (
        nav_e
        -
        ref_e
    ) ** 2
)


# ============================================================
# OUTAGE RESULTS
# ============================================================

outage_mask = (
    (df["time_s"] >= OUTAGE_START)
    &
    (df["time_s"] <= OUTAGE_END)
)

outage_error = position_error[
    outage_mask
]

outage_confidence = confidence[
    outage_mask
]


reference_n_outage = ref_n[
    outage_mask
]

reference_e_outage = ref_e[
    outage_mask
]


# Reference path distance.

dn = np.diff(
    reference_n_outage
)

de = np.diff(
    reference_e_outage
)

reference_distance = np.sum(
    np.sqrt(
        dn ** 2
        +
        de ** 2
    )
)


# ============================================================
# RESULTS
# ============================================================

mean_error = np.mean(
    outage_error
)

median_error = np.median(
    outage_error
)

final_error = outage_error[-1]

max_error = np.max(
    outage_error
)

drift = (
    final_error
    /
    reference_distance
    *
    100
)


print("\n")
print("=" * 80)
print("V9.8-C RESULTS")
print("=" * 80)

print(
    f"\nGNSS outage:"
    f" {OUTAGE_START:.0f}s -> "
    f"{OUTAGE_END:.0f}s"
)

print(
    "\nSamples:",
    len(outage_error)
)

print(
    f"\nReference distance:"
    f" {reference_distance:.3f} m"
)

print(
    f"Mean position error:"
    f" {mean_error:.3f} m"
)

print(
    f"Median position error:"
    f" {median_error:.3f} m"
)

print(
    f"Final position error:"
    f" {final_error:.3f} m"
)

print(
    f"Maximum position error:"
    f" {max_error:.3f} m"
)

print(
    f"Relative final drift:"
    f" {drift:.3f}%"
)

print(
    f"\nMean AI confidence:"
    f" {np.mean(outage_confidence):.4f}"
)

print(
    f"Minimum AI confidence:"
    f" {np.min(outage_confidence):.4f}"
)

print(
    f"Maximum AI confidence:"
    f" {np.max(outage_confidence):.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

result = pd.DataFrame(
    {
        "timestamp":
            df["timestamp"],

        "time_s":
            df["time_s"],

        "gnss_available":
            ~(
                outage_mask
            ),

        "reference_n":
            ref_n,

        "reference_e":
            ref_e,

        "nav_n":
            nav_n,

        "nav_e":
            nav_e,

        "ai_vn":
            ai_vn,

        "ai_ve":
            ai_ve,

        "vn_std":
            vn_std,

        "ve_std":
            ve_std,

        "ai_confidence":
            confidence,

        "fused_vn":
            fused_vn,

        "fused_ve":
            fused_ve,

        "position_error_m":
            position_error,
    }
)


result_file = (
    OUTPUT_DIR
    /
    "prototype_v9_8_c_uncertainty_dr_results.csv"
)

result.to_csv(
    result_file,
    index=False,
)


summary = pd.DataFrame(
    [
        {
            "outage_start_s":
                OUTAGE_START,

            "outage_end_s":
                OUTAGE_END,

            "duration_s":
                OUTAGE_END
                -
                OUTAGE_START,

            "samples":
                len(outage_error),

            "mean_error_m":
                mean_error,

            "median_error_m":
                median_error,

            "final_error_m":
                final_error,

            "max_error_m":
                max_error,

            "reference_distance_m":
                reference_distance,

            "drift_percent":
                drift,

            "mean_confidence":
                np.mean(
                    outage_confidence
                ),
        }
    ]
)


summary_file = (
    OUTPUT_DIR
    /
    "prototype_v9_8_c_uncertainty_dr_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# TRAJECTORY PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.plot(
    ref_e[outage_mask],
    ref_n[outage_mask],
    label="GNSS reference",
)

plt.plot(
    nav_e[outage_mask],
    nav_n[outage_mask],
    label="V9.8-C navigation",
)

plt.xlabel(
    "East (m)"
)

plt.ylabel(
    "North (m)"
)

plt.title(
    "V9.8-C GNSS-Denied Trajectory"
)

plt.axis("equal")

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

trajectory_plot = (
    PLOT_DIR
    /
    "prototype_v9_8_c_trajectory.png"
)

plt.savefig(
    trajectory_plot,
    dpi=200,
)

plt.close()


# ============================================================
# ERROR PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

time_outage = df.loc[
    outage_mask,
    "time_s"
]

plt.plot(
    time_outage,
    outage_error,
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Position error (m)"
)

plt.title(
    "V9.8-C Position Error During GNSS Outage"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

error_plot = (
    PLOT_DIR
    /
    "prototype_v9_8_c_error.png"
)

plt.savefig(
    error_plot,
    dpi=200,
)

plt.close()


# ============================================================
# CONFIDENCE PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    time_outage,
    outage_confidence,
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "AI weight"
)

plt.title(
    "V9.8-C Adaptive AI Weight During GNSS Outage"
)

plt.ylim(
    0,
    1.05,
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

confidence_plot = (
    PLOT_DIR
    /
    "prototype_v9_8_c_confidence.png"
)

plt.savefig(
    confidence_plot,
    dpi=200,
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("V9.8-C COMPLETE")
print("=" * 80)

print("\nResults:")
print(result_file)

print("\nSummary:")
print(summary_file)

print("\nTrajectory:")
print(trajectory_plot)

print("\nError:")
print(error_plot)

print("\nConfidence:")
print(confidence_plot)