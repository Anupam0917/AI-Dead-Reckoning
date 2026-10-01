"""
V9.7 - Standardized Benchmark for Dead-Reckoning Prototypes

Purpose:
    Compare V9.3, V9.4, V9.5, V9.5.2 and V9.6 using the SAME:
        - outage windows
        - reference trajectory
        - position coordinate system
        - error metrics

This script does NOT change any navigation algorithm.
It only standardizes evaluation.

Author: AI Dead Reckoning Project
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(parents=True, exist_ok=True)

# Common evaluation windows.
# These are deliberately inside the held-out test period.
OUTAGES = [
    (7500, 30),
    (7500, 60),
    (7500, 120),

    (8500, 30),
    (8500, 60),
    (8500, 120),

    (9500, 30),
    (9500, 60),
    (9500, 120),

    (10000, 30),
    (10000, 60),
    (10000, 120),
]

EARTH_RADIUS_M = 6371000.0


# ============================================================
# FILES
# ============================================================

FILES = {
    "V9.3 AI DR":
        PROCESSED / "prototype_v9_dead_reckoning_results.csv",

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
# HELPERS
# ============================================================

def find_column(df, candidates):
    """
    Find a column using case-insensitive matching.
    """
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    # Partial matching
    for candidate in candidates:
        key = candidate.strip().lower()

        for normalized_name, original_name in normalized.items():
            if key in normalized_name:
                return original_name

    return None


def load_result_file(path):
    """
    Load a prototype result CSV and identify important columns.
    """

    if not path.exists():
        print(f"[WARNING] File not found: {path}")
        return None

    df = pd.read_csv(path)

    print("\n" + "=" * 70)
    print(f"FILE: {path.name}")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print("Columns:")
    print(list(df.columns))

    return df


def detect_time_column(df):
    return find_column(
        df,
        [
            "time_s",
            "time",
            "timestamp_s",
            "elapsed_time",
            "seconds",
        ],
    )


def detect_error_column(df):
    return find_column(
        df,
        [
            "position_error_m",
            "error_m",
            "position_error",
            "distance_error_m",
            "error",
        ],
    )


def detect_outage_column(df):
    return find_column(
        df,
        [
            "gnss_available",
            "gps_available",
            "gnss",
        ],
    )


def detect_lat_lon(df):
    lat = find_column(
        df,
        [
            "latitude",
            "lat",
            "gps_latitude",
        ],
    )

    lon = find_column(
        df,
        [
            "longitude",
            "lon",
            "gps_longitude",
        ],
    )

    return lat, lon


def detect_reference_position(df):
    """
    Try to find reference GNSS position columns.
    """

    ref_lat = find_column(
        df,
        [
            "reference_latitude",
            "true_latitude",
            "gps_latitude",
            "latitude_true",
        ],
    )

    ref_lon = find_column(
        df,
        [
            "reference_longitude",
            "true_longitude",
            "gps_longitude",
            "longitude_true",
        ],
    )

    return ref_lat, ref_lon


# ============================================================
# POSITION CONVERSION
# ============================================================

def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Convert latitude/longitude to local East/North coordinates.

    Returns:
        north_m
        east_m
    """

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    lat0_rad = np.radians(lat0)

    north = np.radians(lat - lat0) * EARTH_RADIUS_M

    east = (
        np.radians(lon - lon0)
        * EARTH_RADIUS_M
        * np.cos(lat0_rad)
    )

    return north, east


# ============================================================
# GENERIC ERROR CALCULATION
# ============================================================

def calculate_position_error(df):
    """
    Calculate position error when latitude/longitude pairs
    for estimate and reference are available.

    Returns:
        numpy array of errors in meters
    """

    # --------------------------------------------------------
    # Estimated position
    # --------------------------------------------------------

    est_lat = find_column(
        df,
        [
            "estimated_latitude",
            "est_latitude",
            "predicted_latitude",
            "dr_latitude",
            "navigation_latitude",
            "latitude_est",
        ],
    )

    est_lon = find_column(
        df,
        [
            "estimated_longitude",
            "est_longitude",
            "predicted_longitude",
            "dr_longitude",
            "navigation_longitude",
            "longitude_est",
        ],
    )

    # --------------------------------------------------------
    # Reference position
    # --------------------------------------------------------

    ref_lat = find_column(
        df,
        [
            "reference_latitude",
            "true_latitude",
            "gps_latitude",
            "latitude_true",
            "latitude_ref",
        ],
    )

    ref_lon = find_column(
        df,
        [
            "reference_longitude",
            "true_longitude",
            "gps_longitude",
            "longitude_true",
            "longitude_ref",
        ],
    )

    if (
        est_lat is not None
        and est_lon is not None
        and ref_lat is not None
        and ref_lon is not None
    ):

        valid = (
            df[est_lat].notna()
            & df[est_lon].notna()
            & df[ref_lat].notna()
            & df[ref_lon].notna()
        )

        if valid.sum() > 0:

            lat0 = df.loc[valid, ref_lat].iloc[0]
            lon0 = df.loc[valid, ref_lon].iloc[0]

            est_n, est_e = latlon_to_local_xy(
                df.loc[valid, est_lat],
                df.loc[valid, est_lon],
                lat0,
                lon0,
            )

            ref_n, ref_e = latlon_to_local_xy(
                df.loc[valid, ref_lat],
                df.loc[valid, ref_lon],
                lat0,
                lon0,
            )

            errors = np.sqrt(
                (est_n - ref_n) ** 2
                +
                (est_e - ref_e) ** 2
            )

            return errors.to_numpy() if hasattr(errors, "to_numpy") else errors

    return None


# ============================================================
# EXTRACT EXISTING ERROR
# ============================================================

def get_existing_error(df):
    """
    Use an already calculated position-error column if available.
    """

    column = detect_error_column(df)

    if column is None:
        return None

    error = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return error.to_numpy()


# ============================================================
# TIME EXTRACTION
# ============================================================

def get_time(df):
    """
    Extract time in seconds.
    """

    column = detect_time_column(df)

    if column is not None:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        return values.to_numpy()

    # Try timestamp
    timestamp_col = find_column(
        df,
        [
            "timestamp",
            "date",
            "datetime",
        ],
    )

    if timestamp_col is not None:

        timestamps = pd.to_datetime(
            df[timestamp_col],
            errors="coerce"
        )

        seconds = (
            timestamps
            - timestamps.iloc[0]
        ).dt.total_seconds()

        return seconds.to_numpy()

    return None


# ============================================================
# OUTAGE EXTRACTION
# ============================================================

def extract_window(
    df,
    time,
    start,
    duration,
):
    """
    Extract a common outage window.
    """

    if time is None:
        return None

    end = start + duration

    mask = (
        np.isfinite(time)
        &
        (time >= start)
        &
        (time <= end)
    )

    if mask.sum() < 5:
        return None

    return df.loc[mask].copy(), time[mask]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    df_window,
    time_window,
    existing_error=None,
):
    """
    Calculate:

        mean error
        median error
        final error
        max error
        relative drift

    Relative drift:

        final_error / reference_distance * 100
    """

    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    error = None

    if existing_error is not None:

        # Need same window size
        if len(existing_error) == len(df_window):
            error = existing_error

    if error is None:
        error = calculate_position_error(df_window)

    if error is None:
        return None

    error = np.asarray(error, dtype=float)

    valid = np.isfinite(error)

    if valid.sum() < 2:
        return None

    error = error[valid]

    # --------------------------------------------------------
    # Reference distance
    # --------------------------------------------------------

    reference_distance = np.nan

    # Try direct reference distance column
    distance_col = find_column(
        df_window,
        [
            "gnss_distance_m",
            "reference_distance_m",
            "gps_distance_m",
            "distance_m",
        ],
    )

    if distance_col is not None:

        values = pd.to_numeric(
            df_window[distance_col],
            errors="coerce"
        )

        if values.notna().any():
            reference_distance = float(
                values.dropna().iloc[-1]
            )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    mean_error = float(np.mean(error))
    median_error = float(np.median(error))
    final_error = float(error[-1])
    max_error = float(np.max(error))

    if (
        np.isfinite(reference_distance)
        and reference_distance > 0
    ):
        drift = (
            final_error
            /
            reference_distance
            *
            100.0
        )
    else:
        drift = np.nan

    return {
        "mean_error_m": mean_error,
        "median_error_m": median_error,
        "final_error_m": final_error,
        "max_error_m": max_error,
        "reference_distance_m": reference_distance,
        "relative_drift_percent": drift,
        "samples": len(error),
    }


# ============================================================
# SPECIAL HANDLING FOR SAVED RESULTS
# ============================================================

def evaluate_file(name, path):
    """
    Evaluate one saved prototype result file.
    """

    df = load_result_file(path)

    if df is None:
        return []

    time = get_time(df)

    if time is None:
        print("[WARNING] Could not detect time column.")
        return []

    existing_error = get_existing_error(df)

    results = []

    for start, duration in OUTAGES:

        window_result = extract_window(
            df,
            time,
            start,
            duration,
        )

        if window_result is None:
            continue

        df_window, time_window = window_result

        # Find rows from the original error vector
        indices = np.where(
            (
                np.isfinite(time)
                &
                (time >= start)
                &
                (time <= start + duration)
            )
        )[0]

        error_window = None

        if existing_error is not None:
            error_window = existing_error[indices]

        metrics = calculate_metrics(
            df_window,
            time_window,
            error_window,
        )

        if metrics is None:

            print(
                f"[WARNING] {name}: "
                f"unable to calculate "
                f"{start}s/{duration}s"
            )

            continue

        row = {
            "model": name,
            "outage_start_s": start,
            "outage_duration_s": duration,
            **metrics,
        }

        results.append(row)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("V9.7 STANDARDIZED DEAD-RECKONING BENCHMARK")
    print("=" * 80)

    print("\nProject root:")
    print(ROOT)

    print("\nEvaluation windows:")
    for start, duration in OUTAGES:
        print(
            f"  Start = {start:5.0f}s"
            f" | Duration = {duration:3d}s"
        )

    all_results = []

    # --------------------------------------------------------
    # Evaluate every model
    # --------------------------------------------------------

    for name, path in FILES.items():

        print("\n")
        print("#" * 80)
        print(f"EVALUATING: {name}")
        print("#" * 80)

        results = evaluate_file(
            name,
            path,
        )

        all_results.extend(results)

        print(
            f"Valid windows: {len(results)}"
        )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not all_results:

        print("\nERROR:")
        print(
            "No benchmark results were generated."
        )

        print(
            "\nThis usually means the existing CSV "
            "files use different column names."
        )

        print(
            "\nThe column names printed above will "
            "tell us what needs to be mapped."
        )

        return

    results_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    detailed_path = (
        PROCESSED
        /
        "prototype_v9_7_benchmark.csv"
    )

    results_df.to_csv(
        detailed_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        results_df
        .groupby(
            ["model", "outage_duration_s"],
            as_index=False,
        )
        .agg(
            windows=(
                "final_error_m",
                "count",
            ),

            mean_final_error_m=(
                "final_error_m",
                "mean",
            ),

            median_final_error_m=(
                "final_error_m",
                "median",
            ),

            max_final_error_m=(
                "final_error_m",
                "max",
            ),

            mean_position_error_m=(
                "mean_error_m",
                "mean",
            ),

            mean_drift_percent=(
                "relative_drift_percent",
                "mean",
            ),
        )
    )

    summary_path = (
        PROCESSED
        /
        "prototype_v9_7_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("V9.7 BENCHMARK SUMMARY")
    print("=" * 80)

    for _, row in summary.iterrows():

        print(
            f"\n{row['model']}"
            f" | {int(row['outage_duration_s'])}s"
        )

        print(
            f"  Windows       : "
            f"{int(row['windows'])}"
        )

        print(
            f"  Mean final    : "
            f"{row['mean_final_error_m']:.2f} m"
        )

        print(
            f"  Median final  : "
            f"{row['median_final_error_m']:.2f} m"
        )

        print(
            f"  Max final     : "
            f"{row['max_final_error_m']:.2f} m"
        )

        print(
            f"  Mean position : "
            f"{row['mean_position_error_m']:.2f} m"
        )

        if np.isfinite(
            row["mean_drift_percent"]
        ):

            print(
                f"  Mean drift    : "
                f"{row['mean_drift_percent']:.2f}%"
            )

        else:

            print(
                "  Mean drift    : N/A"
            )

    # ========================================================
    # PLOT 1
    # ========================================================

    plt.figure(
        figsize=(11, 6)
    )

    for model in results_df["model"].unique():

        subset = (
            results_df[
                results_df["model"] == model
            ]
            .groupby(
                "outage_duration_s"
            )["final_error_m"]
            .mean()
        )

        plt.plot(
            subset.index,
            subset.values,
            marker="o",
            label=model,
        )

    plt.xlabel(
        "GNSS outage duration (seconds)"
    )

    plt.ylabel(
        "Mean final position error (m)"
    )

    plt.title(
        "V9.7 Standardized Dead-Reckoning Comparison"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plot1 = (
        OUTPUTS
        /
        "prototype_v9_7_comparison.png"
    )

    plt.savefig(
        plot1,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # PLOT 2: FINAL ERROR
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    duration_values = sorted(
        results_df[
            "outage_duration_s"
        ].unique()
    )

    model_values = (
        results_df["model"]
        .unique()
    )

    x = np.arange(
        len(duration_values)
    )

    width = 0.8 / max(
        len(model_values),
        1,
    )

    for i, model in enumerate(
        model_values
    ):

        values = []

        for duration in duration_values:

            subset = results_df[
                (
                    results_df["model"]
                    == model
                )
                &
                (
                    results_df[
                        "outage_duration_s"
                    ]
                    == duration
                )
            ]

            if len(subset) > 0:

                values.append(
                    subset[
                        "final_error_m"
                    ].mean()
                )

            else:

                values.append(
                    np.nan
                )

        plt.bar(
            x
            +
            (
                i
                -
                len(model_values) / 2
                +
                0.5
            )
            * width,
            values,
            width=width,
            label=model,
        )

    plt.xticks(
        x,
        [
            f"{int(v)}s"
            for v in duration_values
        ],
    )

    plt.xlabel(
        "GNSS outage duration"
    )

    plt.ylabel(
        "Mean final error (m)"
    )

    plt.title(
        "Final Position Error by Outage Duration"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plot2 = (
        OUTPUTS
        /
        "prototype_v9_7_final_error.png"
    )

    plt.savefig(
        plot2,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # PLOT 3: DRIFT
    # ========================================================

    plt.figure(
        figsize=(11, 6)
    )

    for model in results_df["model"].unique():

        subset = (
            results_df[
                results_df["model"] == model
            ]
            .groupby(
                "outage_duration_s"
            )[
                "relative_drift_percent"
            ]
            .mean()
        )

        plt.plot(
            subset.index,
            subset.values,
            marker="o",
            label=model,
        )

    plt.xlabel(
        "GNSS outage duration (seconds)"
    )

    plt.ylabel(
        "Mean relative drift (%)"
    )

    plt.title(
        "Dead-Reckoning Drift vs GNSS Outage Duration"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plot3 = (
        OUTPUTS
        /
        "prototype_v9_7_drift.png"
    )

    plt.savefig(
        plot3,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 80)
    print("V9.7 COMPLETE")
    print("=" * 80)

    print(
        f"\nDetailed results:"
        f"\n{detailed_path}"
    )

    print(
        f"\nSummary:"
        f"\n{summary_path}"
    )

    print(
        f"\nPlots:"
        f"\n{plot1}"
        f"\n{plot2}"
        f"\n{plot3}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "V9.7 only standardizes the benchmark."
    )

    print(
        "It does NOT claim that any model is "
        "universally better."
    )


if __name__ == "__main__":
    main()