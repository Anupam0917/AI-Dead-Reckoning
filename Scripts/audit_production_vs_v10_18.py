from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V10_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prototype_v10_18_temporal_gru.csv"
)

PRODUCTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "production_real_sensor_predictions.csv"
)

TARGET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prototype_v9_velocity_targets.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "production"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "production_vs_v10_18_audit.csv"


# ============================================================
# SETTINGS
# ============================================================

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

TIME_TOLERANCE = 0.02


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """
    Return the first matching column.
    """

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        "\nCould not find any of these columns:\n"
        f"{candidates}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


def print_columns(name, df):
    print()
    print("=" * 70)
    print(f"{name} COLUMNS")
    print("=" * 70)

    for i, col in enumerate(df.columns):
        print(f"{i:02d}: {col}")


def calculate_position_error(
    ai_north,
    ai_east,
    ref_north,
    ref_east
):
    """
    Euclidean horizontal position error.
    """

    return np.sqrt(
        (ai_north - ref_north) ** 2
        + (ai_east - ref_east) ** 2
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PRODUCTION VS V10.18 AUDIT")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading files...")

v10 = pd.read_csv(V10_FILE)
production = pd.read_csv(PRODUCTION_FILE)
targets = pd.read_csv(TARGET_FILE)

print(f"V10.18 rows       : {len(v10):,}")
print(f"Production rows   : {len(production):,}")
print(f"Target rows       : {len(targets):,}")


# ============================================================
# PRINT COLUMNS
# ============================================================

print_columns("V10.18", v10)
print_columns("PRODUCTION", production)
print_columns("TARGET", targets)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

# V10.18
v10_time_col = find_column(
    v10,
    ["time_s"]
)

v10_vn_col = find_column(
    v10,
    [
        "vn_ai_gru",
        "vn_ai_mps",
        "vn_pred_mps"
    ]
)

v10_ve_col = find_column(
    v10,
    [
        "ve_ai_gru",
        "ve_ai_mps",
        "ve_pred_mps"
    ]
)


# Production
prod_time_col = find_column(
    production,
    ["time_s"]
)

prod_vn_col = find_column(
    production,
    ["vn_ai_mps"]
)

prod_ve_col = find_column(
    production,
    ["ve_ai_mps"]
)


# Target
target_time_col = find_column(
    targets,
    ["time_s"]
)

target_vn_col = find_column(
    targets,
    ["vn_mps"]
)

target_ve_col = find_column(
    targets,
    ["ve_mps"]
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print("\nConverting time and velocity columns...")

for df, columns in [
    (
        v10,
        [v10_time_col, v10_vn_col, v10_ve_col]
    ),
    (
        production,
        [prod_time_col, prod_vn_col, prod_ve_col]
    ),
    (
        targets,
        [target_time_col, target_vn_col, target_ve_col]
    ),
]:
    for col in columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

v10 = v10.dropna(
    subset=[
        v10_time_col,
        v10_vn_col,
        v10_ve_col
    ]
).copy()

production = production.dropna(
    subset=[
        prod_time_col,
        prod_vn_col,
        prod_ve_col
    ]
).copy()

targets = targets.dropna(
    subset=[
        target_time_col,
        target_vn_col,
        target_ve_col
    ]
).copy()


# ============================================================
# BASIC TIME INFORMATION
# ============================================================

print()
print("=" * 70)
print("TIME COVERAGE")
print("=" * 70)

print(
    f"V10.18       : "
    f"{v10[v10_time_col].min():.3f}s "
    f"to "
    f"{v10[v10_time_col].max():.3f}s"
)

print(
    f"Production   : "
    f"{production[prod_time_col].min():.3f}s "
    f"to "
    f"{production[prod_time_col].max():.3f}s"
)

print(
    f"Targets      : "
    f"{targets[target_time_col].min():.3f}s "
    f"to "
    f"{targets[target_time_col].max():.3f}s"
)


# ============================================================
# OUTAGE FILTER
# ============================================================

v10_outage = v10[
    (v10[v10_time_col] >= OUTAGE_START)
    & (v10[v10_time_col] < OUTAGE_END)
].copy()

prod_outage = production[
    (production[prod_time_col] >= OUTAGE_START)
    & (production[prod_time_col] < OUTAGE_END)
].copy()

target_outage = targets[
    (targets[target_time_col] >= OUTAGE_START)
    & (targets[target_time_col] < OUTAGE_END)
].copy()


print()
print("=" * 70)
print("OUTAGE WINDOW")
print("=" * 70)

print(f"Start       : {OUTAGE_START:.1f}s")
print(f"End         : {OUTAGE_END:.1f}s")
print(
    f"Duration    : "
    f"{OUTAGE_END - OUTAGE_START:.1f}s"
)

print(f"V10.18 rows : {len(v10_outage)}")
print(f"Production  : {len(prod_outage)}")
print(f"Targets     : {len(target_outage)}")


# ============================================================
# V10.18 VS PRODUCTION
# ============================================================

print()
print("=" * 70)
print("V10.18 VS PRODUCTION PREDICTIONS")
print("=" * 70)


v10_compare = v10_outage[
    [
        v10_time_col,
        v10_vn_col,
        v10_ve_col
    ]
].copy()

v10_compare = v10_compare.rename(
    columns={
        v10_time_col: "time_s",
        v10_vn_col: "vn_v10",
        v10_ve_col: "ve_v10"
    }
)


prod_compare = prod_outage[
    [
        prod_time_col,
        prod_vn_col,
        prod_ve_col
    ]
].copy()

prod_compare = prod_compare.rename(
    columns={
        prod_time_col: "time_s",
        prod_vn_col: "vn_production",
        prod_ve_col: "ve_production"
    }
)


# ============================================================
# MATCH PREDICTIONS BY TIME
# ============================================================

print("\nMatching V10.18 and production timestamps...")

v10_compare = v10_compare.sort_values("time_s")
prod_compare = prod_compare.sort_values("time_s")

comparison = pd.merge_asof(
    v10_compare,
    prod_compare,
    on="time_s",
    direction="nearest",
    tolerance=TIME_TOLERANCE
)

comparison = comparison.dropna(
    subset=[
        "vn_production",
        "ve_production"
    ]
).copy()


# ============================================================
# PREDICTION DIFFERENCE
# ============================================================

comparison["vn_difference"] = (
    comparison["vn_production"]
    - comparison["vn_v10"]
)

comparison["ve_difference"] = (
    comparison["ve_production"]
    - comparison["ve_v10"]
)

comparison["vn_abs_difference"] = (
    comparison["vn_difference"].abs()
)

comparison["ve_abs_difference"] = (
    comparison["ve_difference"].abs()
)


print(
    f"\nMatched prediction rows: "
    f"{len(comparison)}"
)


if len(comparison) > 0:

    print()
    print("NORTH VELOCITY")
    print(
        f"MAE difference : "
        f"{comparison['vn_abs_difference'].mean():.8f} m/s"
    )

    print(
        f"Max difference : "
        f"{comparison['vn_abs_difference'].max():.8f} m/s"
    )

    print()
    print("EAST VELOCITY")
    print(
        f"MAE difference : "
        f"{comparison['ve_abs_difference'].mean():.8f} m/s"
    )

    print(
        f"Max difference : "
        f"{comparison['ve_abs_difference'].max():.8f} m/s"
    )

    print()
    print("FIRST 5 COMPARISONS")

    print(
        comparison[
            [
                "time_s",
                "vn_v10",
                "vn_production",
                "vn_difference",
                "ve_v10",
                "ve_production",
                "ve_difference"
            ]
        ].head(5).to_string(index=False)
    )


# ============================================================
# REFERENCE 1: RAW GNSS
# ============================================================

print()
print("=" * 70)
print("REFERENCE 1: RAW GNSS POSITION")
print("=" * 70)


raw_north = None
raw_east = None
raw_reference_final = np.nan


if (
    "latitude" in prod_outage.columns
    and "longitude" in prod_outage.columns
):

    prod_outage = prod_outage.sort_values(
        prod_time_col
    ).reset_index(drop=True)

    lat0 = float(
        prod_outage["latitude"].iloc[0]
    )

    lon0 = float(
        prod_outage["longitude"].iloc[0]
    )

    earth_radius = 6371000.0

    latitude_radians = np.radians(lat0)

    raw_north = (
        np.radians(
            prod_outage["latitude"].to_numpy()
            - lat0
        )
        * earth_radius
    )

    raw_east = (
        np.radians(
            prod_outage["longitude"].to_numpy()
            - lon0
        )
        * earth_radius
        * np.cos(latitude_radians)
    )

    raw_reference_final = np.sqrt(
        raw_north[-1] ** 2
        + raw_east[-1] ** 2
    )

    print(
        f"Starting latitude  : {lat0:.8f}"
    )

    print(
        f"Starting longitude : {lon0:.8f}"
    )

    print(
        f"Raw GNSS final displacement : "
        f"{raw_reference_final:.3f} m"
    )

else:

    print(
        "Production file does not contain "
        "latitude/longitude."
    )


# ============================================================
# REFERENCE 2: TARGET-DERIVED VELOCITY
# ============================================================

print()
print("=" * 70)
print("REFERENCE 2: TARGET-DERIVED VELOCITY")
print("=" * 70)


target_north = None
target_east = None
target_reference_final = np.nan


if len(target_outage) > 1:

    target_outage = target_outage.sort_values(
        target_time_col
    ).reset_index(drop=True)

    target_time = (
        target_outage[target_time_col]
        .to_numpy()
    )

    target_vn = (
        target_outage[target_vn_col]
        .to_numpy()
    )

    target_ve = (
        target_outage[target_ve_col]
        .to_numpy()
    )

    target_dt = np.diff(
        target_time,
        prepend=target_time[0]
    )

    target_dt[0] = 0.0

    target_dt = np.where(
        (target_dt > 0.0)
        & (target_dt <= 1.0),
        target_dt,
        0.1
    )

    target_north = np.cumsum(
        target_vn * target_dt
    )

    target_east = np.cumsum(
        target_ve * target_dt
    )

    target_reference_final = np.sqrt(
        target_north[-1] ** 2
        + target_east[-1] ** 2
    )

    print(
        f"Target-derived final displacement : "
        f"{target_reference_final:.3f} m"
    )

else:

    print(
        "Not enough target data."
    )


# ============================================================
# PRODUCTION AI TRAJECTORY
# ============================================================

print()
print("=" * 70)
print("PRODUCTION AI TRAJECTORY")
print("=" * 70)


prod_outage = prod_outage.sort_values(
    prod_time_col
).reset_index(drop=True)


prod_time = (
    prod_outage[prod_time_col]
    .to_numpy()
)

prod_vn = (
    prod_outage[prod_vn_col]
    .to_numpy()
)

prod_ve = (
    prod_outage[prod_ve_col]
    .to_numpy()
)


prod_dt = np.diff(
    prod_time,
    prepend=prod_time[0]
)

prod_dt[0] = 0.0

prod_dt = np.where(
    (prod_dt > 0.0)
    & (prod_dt <= 1.0),
    prod_dt,
    0.1
)


ai_north = np.cumsum(
    prod_vn * prod_dt
)

ai_east = np.cumsum(
    prod_ve * prod_dt
)


ai_final_displacement = np.sqrt(
    ai_north[-1] ** 2
    + ai_east[-1] ** 2
)


print(
    f"AI final displacement : "
    f"{ai_final_displacement:.3f} m"
)


# ============================================================
# RAW GNSS ERROR
# ============================================================

print()
print("=" * 70)
print("AI VS RAW GNSS")
print("=" * 70)


raw_error = None


if raw_north is not None:

    n = min(
        len(ai_north),
        len(raw_north)
    )

    raw_error = calculate_position_error(
        ai_north[:n],
        ai_east[:n],
        raw_north[:n],
        raw_east[:n]
    )

    print(
        f"Mean position error : "
        f"{raw_error.mean():.3f} m"
    )

    print(
        f"Final position error: "
        f"{raw_error[-1]:.3f} m"
    )

    print(
        f"Maximum error       : "
        f"{raw_error.max():.3f} m"
    )

    if raw_reference_final > 0:

        raw_drift = (
            raw_error[-1]
            / raw_reference_final
            * 100.0
        )

        print(
            f"Final drift         : "
            f"{raw_drift:.3f}%"
        )


# ============================================================
# TARGET-DERIVED ERROR
# ============================================================

print()
print("=" * 70)
print("AI VS TARGET-DERIVED REFERENCE")
print("=" * 70)


target_error = None


if target_north is not None:

    # Match target and production by time.
    target_reference = pd.DataFrame(
        {
            "time_s": target_time,
            "target_north": target_north,
            "target_east": target_east
        }
    )

    ai_reference = pd.DataFrame(
        {
            "time_s": prod_time,
            "ai_north": ai_north,
            "ai_east": ai_east
        }
    )

    trajectory_compare = pd.merge_asof(
        ai_reference.sort_values("time_s"),
        target_reference.sort_values("time_s"),
        on="time_s",
        direction="nearest",
        tolerance=TIME_TOLERANCE
    )

    trajectory_compare = trajectory_compare.dropna(
        subset=[
            "target_north",
            "target_east"
        ]
    ).copy()

    target_error = calculate_position_error(
        trajectory_compare["ai_north"].to_numpy(),
        trajectory_compare["ai_east"].to_numpy(),
        trajectory_compare["target_north"].to_numpy(),
        trajectory_compare["target_east"].to_numpy()
    )

    print(
        f"Matched trajectory rows : "
        f"{len(trajectory_compare)}"
    )

    print(
        f"Mean position error      : "
        f"{target_error.mean():.3f} m"
    )

    print(
        f"Final position error     : "
        f"{target_error[-1]:.3f} m"
    )

    print(
        f"Maximum error            : "
        f"{target_error.max():.3f} m"
    )

    if target_reference_final > 0:

        target_drift = (
            target_error[-1]
            / target_reference_final
            * 100.0
        )

        print(
            f"Final drift              : "
            f"{target_drift:.3f}%"
        )


# ============================================================
# DIAGNOSTIC CONCLUSION
# ============================================================

print()
print("=" * 70)
print("DIAGNOSTIC INTERPRETATION")
print("=" * 70)


if len(comparison) == 0:

    print(
        "Could not compare V10.18 and production predictions."
    )

else:

    vn_mae = (
        comparison["vn_abs_difference"].mean()
    )

    ve_mae = (
        comparison["ve_abs_difference"].mean()
    )

    max_difference = max(
        comparison["vn_abs_difference"].max(),
        comparison["ve_abs_difference"].max()
    )

    print(
        f"Prediction North MAE difference : "
        f"{vn_mae:.8f} m/s"
    )

    print(
        f"Prediction East MAE difference  : "
        f"{ve_mae:.8f} m/s"
    )

    print(
        f"Maximum prediction difference   : "
        f"{max_difference:.8f} m/s"
    )

    if max_difference < 0.001:

        print()
        print(
            "RESULT: Production inference closely "
            "matches V10.18."
        )

        print(
            "The production model implementation "
            "is consistent with the trained V10.18 model."
        )

    elif max_difference < 0.01:

        print()
        print(
            "RESULT: Production inference is very "
            "close to V10.18."
        )

        print(
            "Small numerical/preprocessing differences "
            "may exist."
        )

    else:

        print()
        print(
            "RESULT: Production predictions differ "
            "noticeably from V10.18."
        )

        print(
            "We need to investigate preprocessing, "
            "timestamps, or sequence construction."
        )


# ============================================================
# SAVE AUDIT DATA
# ============================================================

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print()
print("Saved prediction comparison:")
print(OUTPUT_FILE)