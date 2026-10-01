import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# V9 STEP 3
# AI VELOCITY DEAD RECKONING
# ============================================================

print("=" * 70)
print("V9 AI VELOCITY DEAD RECKONING")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

AI_PATH = (
    "data/processed/"
    "prototype_v9_ai_velocity_results.csv"
)

TARGET_PATH = (
    "data/processed/"
    "prototype_v9_velocity_targets.csv"
)

OUTPUT_DIR = "outputs"
PROCESSED_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD AI RESULTS
# ------------------------------------------------------------

print("\nLoading AI velocity predictions...")

df = pd.read_csv(AI_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print(
    "AI test rows:",
    len(df)
)


# ------------------------------------------------------------
# LOAD V9 TARGET TIMELINE
# ------------------------------------------------------------

print("\nLoading original V9 timeline...")

timeline = pd.read_csv(
    TARGET_PATH,
    usecols=[
        "timestamp",
        "time_s"
    ]
)

timeline["timestamp"] = pd.to_datetime(
    timeline["timestamp"]
)

print(
    "Full timeline rows:",
    len(timeline)
)


# ------------------------------------------------------------
# MERGE GLOBAL TIME
# ------------------------------------------------------------

df = pd.merge_asof(
    df.sort_values("timestamp"),
    timeline.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("150ms")
)

df = df.dropna(
    subset=["time_s"]
).reset_index(drop=True)


print(
    "Rows with global time:",
    len(df)
)

print(
    "Test time range:",
    f"{df['time_s'].iloc[0]:.2f}s"
    " -> "
    f"{df['time_s'].iloc[-1]:.2f}s"
)


# ------------------------------------------------------------
# TIME DIFFERENCE
# ------------------------------------------------------------

df["dt"] = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
)

df["dt"] = df["dt"].clip(
    lower=0.01,
    upper=0.5
)

df["dt"] = df["dt"].fillna(
    0.1
)


# ------------------------------------------------------------
# AI VELOCITY
# ------------------------------------------------------------

vn = df[
    "vnorth_ai_mps"
].values

ve = df[
    "veast_ai_mps"
].values


# ------------------------------------------------------------
# OUTAGE BENCHMARK
# ------------------------------------------------------------

# These are GLOBAL times from the original dataset.
# All are inside the V9 held-out test region.

outage_starts = [
    7500,
    8500,
    9500,
    10000
]

durations = [
    30,
    60,
    120
]


benchmark = []


print("\nRunning outage benchmark...")


for start in outage_starts:

    for duration in durations:

        end = start + duration


        mask = (
            (df["time_s"] >= start)
            &
            (df["time_s"] <= end)
        )


        indices = np.where(
            mask
        )[0]


        if len(indices) < 20:

            print(
                f"Skipping {start}s / "
                f"{duration}s: "
                f"only {len(indices)} samples"
            )

            continue


        # ----------------------------------------------------
        # LOCAL AI VELOCITY
        # ----------------------------------------------------

        local_vn = vn[
            indices
        ]

        local_ve = ve[
            indices
        ]

        local_dt = df[
            "dt"
        ].iloc[
            indices
        ].values


        # ----------------------------------------------------
        # AI DEAD RECKONING
        # ----------------------------------------------------

        dr_n = np.zeros(
            len(indices)
        )

        dr_e = np.zeros(
            len(indices)
        )


        for j in range(1, len(indices)):

            dr_n[j] = (
                dr_n[j - 1]
                + local_vn[j] *
                local_dt[j]
            )

            dr_e[j] = (
                dr_e[j - 1]
                + local_ve[j] *
                local_dt[j]
            )


        # ----------------------------------------------------
        # GNSS-DERIVED REFERENCE VELOCITY
        # ----------------------------------------------------

        true_vn = df[
            "vnorth_true_mps"
        ].iloc[
            indices
        ].values

        true_ve = df[
            "veast_true_mps"
        ].iloc[
            indices
        ].values


        true_n = np.zeros(
            len(indices)
        )

        true_e = np.zeros(
            len(indices)
        )


        for j in range(1, len(indices)):

            true_n[j] = (
                true_n[j - 1]
                + true_vn[j] *
                local_dt[j]
            )

            true_e[j] = (
                true_e[j - 1]
                + true_ve[j] *
                local_dt[j]
            )


        # ----------------------------------------------------
        # POSITION ERROR
        # ----------------------------------------------------

        error = np.sqrt(
            (dr_n - true_n) ** 2
            +
            (dr_e - true_e) ** 2
        )


        final_error = error[-1]

        mean_error = np.mean(
            error
        )

        max_error = np.max(
            error
        )


        # ----------------------------------------------------
        # TRUE TRAVEL DISTANCE
        # ----------------------------------------------------

        true_distance = np.sum(
            np.sqrt(
                np.diff(true_n) ** 2
                +
                np.diff(true_e) ** 2
            )
        )


        if true_distance > 1:

            drift_percent = (
                final_error /
                true_distance
            ) * 100

        else:

            drift_percent = np.nan


        benchmark.append({

            "start_s": start,

            "duration_s": duration,

            "samples": len(indices),

            "final_error_m":
                final_error,

            "mean_error_m":
                mean_error,

            "max_error_m":
                max_error,

            "true_distance_m":
                true_distance,

            "drift_percent":
                drift_percent
        })


        print(
            f"{start:5d}s | "
            f"{duration:3d}s | "
            f"samples={len(indices):4d} | "
            f"final={final_error:8.2f}m | "
            f"mean={mean_error:8.2f}m | "
            f"drift={drift_percent:7.2f}%"
        )


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

results = pd.DataFrame(
    benchmark
)


print("\n" + "=" * 70)
print("V9 DEAD RECKONING RESULTS")
print("=" * 70)


if len(results) > 0:

    print(
        "\nValid outage windows:",
        len(results)
    )

    print(
        "\nMean final error:",
        f"{results['final_error_m'].mean():.2f} m"
    )

    print(
        "Median final error:",
        f"{results['final_error_m'].median():.2f} m"
    )

    print(
        "Maximum final error:",
        f"{results['final_error_m'].max():.2f} m"
    )

    print(
        "\nMean position error:",
        f"{results['mean_error_m'].mean():.2f} m"
    )

    print(
        "Mean drift:",
        f"{results['drift_percent'].mean():.2f}%"
    )

else:

    print(
        "\nERROR: No valid outage windows found."
    )


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

output_csv = (
    "data/processed/"
    "prototype_v9_dead_reckoning_results.csv"
)

results.to_csv(
    output_csv,
    index=False
)


# ------------------------------------------------------------
# 60 SECOND TRAJECTORY
# ------------------------------------------------------------

if len(results) > 0:

    example_start = 8500
    example_duration = 60

    example_end = (
        example_start +
        example_duration
    )


    mask = (
        (df["time_s"] >= example_start)
        &
        (df["time_s"] <= example_end)
    )


    example = df.loc[
        mask
    ].copy()


    if len(example) > 1:

        evn = example[
            "vnorth_ai_mps"
        ].values

        eve = example[
            "veast_ai_mps"
        ].values


        evn_true = example[
            "vnorth_true_mps"
        ].values

        eve_true = example[
            "veast_true_mps"
        ].values


        dte = example[
            "dt"
        ].values


        dr_n = np.zeros(
            len(example)
        )

        dr_e = np.zeros(
            len(example)
        )

        true_n = np.zeros(
            len(example)
        )

        true_e = np.zeros(
            len(example)
        )


        for i in range(1, len(example)):

            dr_n[i] = (
                dr_n[i - 1]
                + evn[i] *
                dte[i]
            )

            dr_e[i] = (
                dr_e[i - 1]
                + eve[i] *
                dte[i]
            )


            true_n[i] = (
                true_n[i - 1]
                + evn_true[i] *
                dte[i]
            )

            true_e[i] = (
                true_e[i - 1]
                + eve_true[i] *
                dte[i]
            )


        plt.figure(
            figsize=(10, 8)
        )


        plt.plot(
            true_e,
            true_n,
            label="GNSS Reference"
        )


        plt.plot(
            dr_e,
            dr_n,
            label="AI Dead Reckoning"
        )


        plt.xlabel(
            "East (m)"
        )

        plt.ylabel(
            "North (m)"
        )

        plt.title(
            "V9 AI Dead Reckoning - "
            "60 s GNSS Outage"
        )

        plt.legend()

        plt.axis("equal")

        plt.grid(True)

        plt.tight_layout()


        trajectory_path = (
            "outputs/"
            "prototype_v9_dead_reckoning.png"
        )


        plt.savefig(
            trajectory_path,
            dpi=150
        )

        plt.close()


# ------------------------------------------------------------
# ERROR PLOT
# ------------------------------------------------------------

if len(results) > 0:

    plt.figure(
        figsize=(10, 7)
    )


    for duration in durations:

        subset = results[
            results["duration_s"] == duration
        ]


        if len(subset) == 0:
            continue


        plt.plot(
            subset["start_s"],
            subset["final_error_m"],
            marker="o",
            label=f"{duration}s"
        )


    plt.xlabel(
        "Outage Start Time (s)"
    )

    plt.ylabel(
        "Final Position Error (m)"
    )

    plt.title(
        "V9 AI Dead Reckoning Error"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


    error_plot = (
        "outputs/"
        "prototype_v9_dead_reckoning_error.png"
    )


    plt.savefig(
        error_plot,
        dpi=150
    )

    plt.close()


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V9 STEP 3 COMPLETE")
print("=" * 70)

print("\nFiles created:")

print(
    "1.",
    output_csv
)

print(
    "2.",
    "outputs/prototype_v9_dead_reckoning.png"
)

print(
    "3.",
    "outputs/prototype_v9_dead_reckoning_error.png"
)

print("\nPipeline:")

print(
    "IMU + Magnetometer"
    " -> AI Vnorth/Veast"
    " -> Dead Reckoning"
)

print("=" * 70)