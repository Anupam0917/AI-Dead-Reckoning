"""
V9.7.2 - Corrected Standardized Benchmark

Important:
Only evaluate the ACTUAL GNSS outage contained in each
time-series result file.

Current common outage:
    8500 s -> 8560 s

V9.3 is a special summary file and already contains
30s / 60s / 120s simulated outage results.

For the other versions, only the actual 60s outage
is evaluated.

This prevents post-GNSS-recovery samples from being
mistaken for dead-reckoning samples.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# COMMON ACTUAL OUTAGE
# ============================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0


# ============================================================
# FILES
# ============================================================

FILES = {
    "V9.4 Bias EKF":
        PROCESSED / "prototype_v9_bias_ekf_results.csv",

    "V9.5 IMU EKF":
        PROCESSED / "prototype_v9_5_imu_ekf_results.csv",

    "V9.5.2 IMU+AI EKF":
        PROCESSED / "prototype_v9_5_2_imu_ai_ekf_results.csv",

    "V9.6 Confidence DR":
        PROCESSED / "prototype_v9_6_confidence_dr_results.csv",
}


# ============================================================
# V9.3 SPECIAL RESULTS
# ============================================================

V93_FILE = (
    PROCESSED /
    "prototype_v9_dead_reckoning_results.csv"
)


# ============================================================
# HELPERS
# ============================================================

def calculate_reference_distance(
    df,
    north_col,
    east_col,
):
    """
    Calculate total reference trajectory distance
    during the actual GNSS outage.
    """

    north = pd.to_numeric(
        df[north_col],
        errors="coerce",
    ).to_numpy()

    east = pd.to_numeric(
        df[east_col],
        errors="coerce",
    ).to_numpy()

    valid = (
        np.isfinite(north)
        &
        np.isfinite(east)
    )

    north = north[valid]
    east = east[valid]

    if len(north) < 2:
        return np.nan

    dn = np.diff(north)
    de = np.diff(east)

    distance = np.sqrt(
        dn ** 2 +
        de ** 2
    )

    return float(
        np.sum(distance)
    )


def calculate_error_from_positions(
    df,
    est_north_col,
    est_east_col,
    ref_north_col,
    ref_east_col,
):
    """
    Calculate position error directly from
    estimated and reference N/E coordinates.
    """

    est_n = pd.to_numeric(
        df[est_north_col],
        errors="coerce",
    ).to_numpy()

    est_e = pd.to_numeric(
        df[est_east_col],
        errors="coerce",
    ).to_numpy()

    ref_n = pd.to_numeric(
        df[ref_north_col],
        errors="coerce",
    ).to_numpy()

    ref_e = pd.to_numeric(
        df[ref_east_col],
        errors="coerce",
    ).to_numpy()

    error = np.sqrt(
        (est_n - ref_n) ** 2
        +
        (est_e - ref_e) ** 2
    )

    return error


# ============================================================
# V9.4
# ============================================================

def evaluate_v94():

    path = FILES["V9.4 Bias EKF"]

    df = pd.read_csv(path)

    outage = df[
        (df["time_s"] >= OUTAGE_START)
        &
        (df["time_s"] <= OUTAGE_END)
        &
        (df["gnss_available"] == False)
    ].copy()

    print("\nV9.4")
    print("-" * 60)
    print("Outage rows:", len(outage))

    error = calculate_error_from_positions(
        outage,
        "ekf_north_m",
        "ekf_east_m",
        "reference_north_m",
        "reference_east_m",
    )

    distance = calculate_reference_distance(
        outage,
        "reference_north_m",
        "reference_east_m",
    )

    return {
        "model": "V9.4 Bias EKF",
        "duration_s": 60,
        "samples": len(outage),
        "mean_error_m": np.mean(error),
        "median_error_m": np.median(error),
        "final_error_m": error[-1],
        "max_error_m": np.max(error),
        "reference_distance_m": distance,
        "drift_percent": (
            error[-1] / distance * 100
            if distance > 0
            else np.nan
        ),
    }


# ============================================================
# V9.5
# ============================================================

def evaluate_v95():

    path = FILES["V9.5 IMU EKF"]

    df = pd.read_csv(path)

    outage = df[
        (df["time_s"] >= OUTAGE_START)
        &
        (df["time_s"] <= OUTAGE_END)
        &
        (df["gnss_available"] == False)
    ].copy()

    print("\nV9.5")
    print("-" * 60)
    print("Outage rows:", len(outage))

    error = calculate_error_from_positions(
        outage,
        "north",
        "east",
        "gps_north",
        "gps_east",
    )

    distance = calculate_reference_distance(
        outage,
        "gps_north",
        "gps_east",
    )

    return {
        "model": "V9.5 IMU EKF",
        "duration_s": 60,
        "samples": len(outage),
        "mean_error_m": np.mean(error),
        "median_error_m": np.median(error),
        "final_error_m": error[-1],
        "max_error_m": np.max(error),
        "reference_distance_m": distance,
        "drift_percent": (
            error[-1] / distance * 100
            if distance > 0
            else np.nan
        ),
    }


# ============================================================
# V9.5.2
# ============================================================

def evaluate_v952():

    path = FILES["V9.5.2 IMU+AI EKF"]

    df = pd.read_csv(path)

    outage = df[
        (df["time_s"] >= OUTAGE_START)
        &
        (df["time_s"] <= OUTAGE_END)
    ].copy()

    # V9.5.2 does not contain a GNSS availability
    # column, so the known simulated outage is used.

    outage = outage[
        (outage["time_s"] >= OUTAGE_START)
        &
        (outage["time_s"] <= OUTAGE_END)
    ]

    print("\nV9.5.2")
    print("-" * 60)
    print("Outage rows:", len(outage))

    error = calculate_error_from_positions(
        outage,
        "estimated_n",
        "estimated_e",
        "gnss_n",
        "gnss_e",
    )

    distance = calculate_reference_distance(
        outage,
        "gnss_n",
        "gnss_e",
    )

    return {
        "model": "V9.5.2 IMU+AI EKF",
        "duration_s": 60,
        "samples": len(outage),
        "mean_error_m": np.mean(error),
        "median_error_m": np.median(error),
        "final_error_m": error[-1],
        "max_error_m": np.max(error),
        "reference_distance_m": distance,
        "drift_percent": (
            error[-1] / distance * 100
            if distance > 0
            else np.nan
        ),
    }


# ============================================================
# V9.6
# ============================================================

def evaluate_v96():

    path = FILES["V9.6 Confidence DR"]

    df = pd.read_csv(path)

    outage = df[
        (df["time_s"] >= OUTAGE_START)
        &
        (df["time_s"] <= OUTAGE_END)
        &
        (df["gnss_available"] == False)
    ].copy()

    print("\nV9.6")
    print("-" * 60)
    print("Outage rows:", len(outage))

    error = calculate_error_from_positions(
        outage,
        "nav_n",
        "nav_e",
        "gnss_n",
        "gnss_e",
    )

    distance = calculate_reference_distance(
        outage,
        "gnss_n",
        "gnss_e",
    )

    return {
        "model": "V9.6 Confidence DR",
        "duration_s": 60,
        "samples": len(outage),
        "mean_error_m": np.mean(error),
        "median_error_m": np.median(error),
        "final_error_m": error[-1],
        "max_error_m": np.max(error),
        "reference_distance_m": distance,
        "drift_percent": (
            error[-1] / distance * 100
            if distance > 0
            else np.nan
        ),
    }


# ============================================================
# V9.3
# ============================================================

def evaluate_v93():

    if not V93_FILE.exists():

        print(
            "\nV9.3 file not found."
        )

        return []

    df = pd.read_csv(V93_FILE)

    # Only use 60-second window so that comparison
    # is exactly aligned with the other implementations.

    row = df[
        (df["start_s"] == 8500)
        &
        (df["duration_s"] == 60)
    ]

    if len(row) == 0:
        return []

    row = row.iloc[0]

    print("\nV9.3")
    print("-" * 60)
    print(
        "Using existing V9.3 "
        "8500 -> 8560 result."
    )

    return [{
        "model": "V9.3 AI DR",
        "duration_s": 60,
        "samples": int(row["samples"]),
        "mean_error_m": float(row["mean_error_m"]),
        "median_error_m": np.nan,
        "final_error_m": float(row["final_error_m"]),
        "max_error_m": float(row["max_error_m"]),
        "reference_distance_m": float(
            row["true_distance_m"]
        ),
        "drift_percent": float(
            row["drift_percent"]
        ),
    }]


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("V9.7.2 CORRECTED BENCHMARK")
    print("=" * 80)

    print(
        f"\nCommon actual outage:"
        f" {OUTAGE_START:.0f}s -> "
        f"{OUTAGE_END:.0f}s"
    )

    results = []

    # --------------------------------------------------------
    # V9.3
    # --------------------------------------------------------

    results.extend(
        evaluate_v93()
    )

    # --------------------------------------------------------
    # V9.4
    # --------------------------------------------------------

    results.append(
        evaluate_v94()
    )

    # --------------------------------------------------------
    # V9.5
    # --------------------------------------------------------

    results.append(
        evaluate_v95()
    )

    # --------------------------------------------------------
    # V9.5.2
    # --------------------------------------------------------

    results.append(
        evaluate_v952()
    )

    # --------------------------------------------------------
    # V9.6
    # --------------------------------------------------------

    results.append(
        evaluate_v96()
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_csv = (
        PROCESSED
        /
        "prototype_v9_7_2_corrected_benchmark.csv"
    )

    result_df.to_csv(
        output_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CORRECTED RESULTS")
    print("=" * 80)

    print(
        result_df[
            [
                "model",
                "samples",
                "mean_error_m",
                "median_error_m",
                "final_error_m",
                "max_error_m",
                "reference_distance_m",
                "drift_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # ========================================================
    # BAR CHART
    # ========================================================

    plt.figure(
        figsize=(11, 6)
    )

    x = np.arange(
        len(result_df)
    )

    plt.bar(
        x,
        result_df["final_error_m"],
    )

    plt.xticks(
        x,
        result_df["model"],
        rotation=20,
        ha="right",
    )

    plt.ylabel(
        "Final position error (m)"
    )

    plt.title(
        "Final Position Error During Same 60s GNSS Outage"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plot_path = (
        OUTPUTS
        /
        "prototype_v9_7_2_final_error.png"
    )

    plt.savefig(
        plot_path,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # DRIFT PLOT
    # ========================================================

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        x,
        result_df["drift_percent"],
    )

    plt.xticks(
        x,
        result_df["model"],
        rotation=20,
        ha="right",
    )

    plt.ylabel(
        "Relative final drift (%)"
    )

    plt.title(
        "Relative Drift During Same 60s GNSS Outage"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    drift_path = (
        OUTPUTS
        /
        "prototype_v9_7_2_drift.png"
    )

    plt.savefig(
        drift_path,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 80)
    print("V9.7.2 COMPLETE")
    print("=" * 80)

    print(
        "\nCSV:"
    )
    print(output_csv)

    print(
        "\nFinal-error plot:"
    )
    print(plot_path)

    print(
        "\nDrift plot:"
    )
    print(drift_path)

    print(
        "\nThis benchmark evaluates ONLY the actual "
        "GNSS-denied interval."
    )


if __name__ == "__main__":
    main()