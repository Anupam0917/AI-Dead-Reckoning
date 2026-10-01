from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"


FILES = [
    "prototype_v9_bias_ekf_results.csv",
    "prototype_v9_5_imu_ekf_results.csv",
    "prototype_v9_5_2_imu_ai_ekf_results.csv",
    "prototype_v9_6_confidence_dr_results.csv",
]


def inspect_file(filename):

    path = PROCESSED / filename

    print("\n")
    print("=" * 80)
    print(filename)
    print("=" * 80)

    if not path.exists():
        print("FILE NOT FOUND")
        return

    df = pd.read_csv(path)

    print("Rows:", len(df))

    print("\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"{i:2d}: {col}")

    # --------------------------------------------------
    # Time
    # --------------------------------------------------

    if "time_s" in df.columns:

        time = pd.to_numeric(
            df["time_s"],
            errors="coerce"
        )

        print("\nTIME")
        print("min:", time.min())
        print("max:", time.max())
        print("NaN:", time.isna().sum())

    # --------------------------------------------------
    # GNSS availability
    # --------------------------------------------------

    if "gnss_available" in df.columns:

        print("\nGNSS AVAILABLE")

        print(
            df["gnss_available"]
            .value_counts(dropna=False)
        )

    # --------------------------------------------------
    # Position error
    # --------------------------------------------------

    error_columns = [
        "position_error_m",
        "position_error",
    ]

    for col in error_columns:

        if col in df.columns:

            error = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            print(f"\n{col}")

            print("min :", error.min())
            print("mean:", error.mean())
            print("median:", error.median())
            print("max :", error.max())

            for start, duration in [
                (7500, 30),
                (7500, 60),
                (7500, 120),
                (8500, 30),
                (8500, 60),
                (8500, 120),
            ]:

                if "time_s" not in df.columns:
                    continue

                mask = (
                    (df["time_s"] >= start)
                    &
                    (df["time_s"] <= start + duration)
                )

                values = error[mask].dropna()

                if len(values) == 0:
                    continue

                print(
                    f"  {start}s -> {start + duration}s:"
                    f" samples={len(values)}"
                    f" final={values.iloc[-1]:.3f}m"
                    f" mean={values.mean():.3f}m"
                    f" max={values.max():.3f}m"
                )

    # --------------------------------------------------
    # Position columns
    # --------------------------------------------------

    position_candidates = [
        "reference_north_m",
        "reference_east_m",
        "gnss_n",
        "gnss_e",
        "gps_north",
        "gps_e",
        "estimated_n",
        "estimated_e",
        "ekf_north_m",
        "ekf_east_m",
        "nav_n",
        "nav_e",
        "north",
        "east",
    ]

    print("\nPOSITION COLUMNS")

    for col in position_candidates:

        if col in df.columns:

            series = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            print(
                f"{col:25s}"
                f" min={series.min():10.3f}"
                f" max={series.max():10.3f}"
                f" NaN={series.isna().sum()}"
            )


# ============================================================
# MAIN
# ============================================================

print("\n")
print("=" * 80)
print("V9.7.1 BENCHMARK AUDIT")
print("=" * 80)

for filename in FILES:
    inspect_file(filename)


# ============================================================
# V9.3 SPECIAL FILE
# ============================================================

v93_path = (
    PROCESSED
    /
    "prototype_v9_dead_reckoning_results.csv"
)

print("\n")
print("=" * 80)
print("V9.3 SPECIAL WINDOW-SUMMARY FILE")
print("=" * 80)

if v93_path.exists():

    df = pd.read_csv(v93_path)

    print("\nRows:")
    print(len(df))

    print("\nColumns:")
    print(list(df.columns))

    print("\nFull V9.3 results:")
    print(df.to_string(index=False))

else:

    print("V9.3 file not found.")


print("\n")
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)