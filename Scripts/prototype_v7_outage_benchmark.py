import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SIH 2026
# PROTOTYPE V7
#
# MULTI-OUTAGE GNSS-DENIED NAVIGATION BENCHMARK
#
# Inputs:
#   AI speed        -> Prototype V1
#   IMU heading     -> Prototype V5
#
# For multiple simulated GNSS outages:
#   AI speed + heading
#            ↓
#       Dead Reckoning
#            ↓
#       Position error
#
# Durations:
#   30 s
#   60 s
#   120 s
#
# The benchmark uses several different locations in the
# dataset rather than relying on one outage.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RAW_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Synchronised V abd S datasets",
    "Categorised IOVNB Dataset",
    "M (Driver B)",
    "S-M.csv"
)

AI_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "prototype_ai_speed_results.csv"
)

HEADING_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "prototype_v5_imu_heading.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

EARTH_RADIUS = 6371000.0

# Different outage durations
OUTAGE_DURATIONS = [
    30.0,
    60.0,
    120.0
]

# Candidate outage starting points.
#
# These are spread throughout the journey rather than using
# only the V2/V6 test location.
CANDIDATE_STARTS = [
    1500.0,
    2500.0,
    3500.0,
    4500.0,
    5500.0,
    6500.0,
    7500.0,
    8500.0,
    9500.0,
    10000.0
]

# Minimum speed during an outage.
#
# We prefer moving sections because a stationary outage does
# not meaningfully test dead reckoning.
MIN_MOVING_SPEED = 2.0


# ============================================================
# LOAD RAW DATA
# ============================================================

print("\n" + "=" * 75)
print("SIH PROTOTYPE V7 - MULTI-OUTAGE BENCHMARK")
print("=" * 75)

print("\nLoading raw dataset...")

df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

print(
    f"Rows: {len(df)}"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(
        " ",
        "_"
    )
    .str.replace(
        "(",
        "",
        regex=False
    )
    .str.replace(
        ")",
        "",
        regex=False
    )
    .str.replace(
        "/",
        "_",
        regex=False
    )
)


def find_column(keyword):

    matches = [
        c
        for c in df.columns
        if keyword.lower() in c.lower()
    ]

    if not matches:

        raise ValueError(
            f"Column not found: {keyword}"
        )

    return matches[0]


TIME_COL = find_column(
    "DATE"
)

LAT_COL = find_column(
    "GPS_LATITUDE"
)

LON_COL = find_column(
    "GPS_LONGITUDE"
)

GPS_SPEED_COL = find_column(
    "GPS_SPEED"
)


print("\nDetected columns:")

print(
    "Timestamp:",
    TIME_COL
)

print(
    "Latitude:",
    LAT_COL
)

print(
    "Longitude:",
    LON_COL
)

print(
    "GPS speed:",
    GPS_SPEED_COL
)


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df[TIME_COL],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)

df = df.dropna(
    subset=[
        "timestamp"
    ]
).copy()

df["time_seconds"] = (
    df["timestamp"]
    -
    df["timestamp"].iloc[0]
).dt.total_seconds()


time = df[
    "time_seconds"
].values


# ============================================================
# GPS DATA
# ============================================================

lat = pd.to_numeric(
    df[LAT_COL],
    errors="coerce"
).values

lon = pd.to_numeric(
    df[LON_COL],
    errors="coerce"
).values

gps_speed = pd.to_numeric(
    df[GPS_SPEED_COL],
    errors="coerce"
).values


gps_speed = (
    pd.Series(
        gps_speed
    )
    .interpolate(
        limit_direction="both"
    )
    .values
)


# ============================================================
# GPS → LOCAL NORTH/EAST
# ============================================================

lat0 = lat[0]

lon0 = lon[0]

lat0_rad = np.radians(
    lat0
)


gps_north = (
    np.radians(
        lat - lat0
    )
    * EARTH_RADIUS
)

gps_east = (
    np.radians(
        lon - lon0
    )
    * EARTH_RADIUS
    * np.cos(
        lat0_rad
    )
)


# ============================================================
# LOAD AI SPEED
# ============================================================

print(
    "\nLoading AI speed..."
)

ai_df = pd.read_csv(
    AI_FILE
)

ai_time = pd.to_numeric(
    ai_df["time_s"],
    errors="coerce"
).values

ai_speed = pd.to_numeric(
    ai_df["ai_speed_kmh"],
    errors="coerce"
).values


valid_ai = (
    np.isfinite(
        ai_time
    )
    &
    np.isfinite(
        ai_speed
    )
)


ai_time = ai_time[
    valid_ai
]

ai_speed = ai_speed[
    valid_ai
]


df["ai_speed_kmh"] = np.interp(
    time,
    ai_time,
    ai_speed
)


# ============================================================
# LOAD V5 HEADING
# ============================================================

print(
    "\nLoading V5 heading..."
)

heading_df = pd.read_csv(
    HEADING_FILE
)

heading_time = pd.to_numeric(
    heading_df[
        "time_seconds"
    ],
    errors="coerce"
).values

heading_values = pd.to_numeric(
    heading_df[
        "estimated_heading_deg"
    ],
    errors="coerce"
).values


valid_heading = (
    np.isfinite(
        heading_time
    )
    &
    np.isfinite(
        heading_values
    )
)


heading_time = heading_time[
    valid_heading
]

heading_values = heading_values[
    valid_heading
]


# ============================================================
# CIRCULAR INTERPOLATION
# ============================================================

heading_rad = np.radians(
    heading_values
)

heading_sin = np.sin(
    heading_rad
)

heading_cos = np.cos(
    heading_rad
)


interp_sin = np.interp(
    time,
    heading_time,
    heading_sin
)

interp_cos = np.interp(
    time,
    heading_time,
    heading_cos
)


heading = (
    np.degrees(
        np.arctan2(
            interp_sin,
            interp_cos
        )
    )
    + 360.0
) % 360.0


# ============================================================
# HELPER: DEAD RECKONING
# ============================================================

def run_dead_reckoning(
    start_time,
    duration
):

    end_time = (
        start_time
        + duration
    )


    start_idx = np.searchsorted(
        time,
        start_time
    )

    end_idx = np.searchsorted(
        time,
        end_time
    )


    if start_idx >= len(time):

        return None


    if end_idx >= len(time):

        return None


    # --------------------------------------------------------
    # Ensure the selected section is actually moving
    # --------------------------------------------------------

    section_speed = gps_speed[
        start_idx:end_idx + 1
    ]

    moving_fraction = np.mean(
        section_speed
        > MIN_MOVING_SPEED
    )


    if moving_fraction < 0.60:

        return None


    # --------------------------------------------------------
    # Initialize DR at true GNSS position
    # --------------------------------------------------------

    dr_north = gps_north[
        start_idx
    ]

    dr_east = gps_east[
        start_idx
    ]


    errors = []

    trajectory_n = [
        dr_north
    ]

    trajectory_e = [
        dr_east
    ]


    # --------------------------------------------------------
    # Propagation
    # --------------------------------------------------------

    for i in range(
        start_idx + 1,
        end_idx + 1
    ):

        dt = (
            time[i]
            -
            time[i - 1]
        )


        if (
            dt <= 0
            or dt > 0.5
        ):

            dt = 0.1


        # AI speed
        speed_ms = (
            df[
                "ai_speed_kmh"
            ].iloc[i]
            / 3.6
        )


        # V5 heading
        heading_rad = np.radians(
            heading[i]
        )


        velocity_north = (
            speed_ms
            * np.cos(
                heading_rad
            )
        )

        velocity_east = (
            speed_ms
            * np.sin(
                heading_rad
            )
        )


        dr_north += (
            velocity_north
            * dt
        )

        dr_east += (
            velocity_east
            * dt
        )


        # ----------------------------------------------------
        # Compare with GNSS reference
        # ----------------------------------------------------

        error = np.sqrt(
            (
                dr_north
                -
                gps_north[i]
            ) ** 2
            +
            (
                dr_east
                -
                gps_east[i]
            ) ** 2
        )


        errors.append(
            error
        )

        trajectory_n.append(
            dr_north
        )

        trajectory_e.append(
            dr_east
        )


    errors = np.asarray(
        errors
    )


    # --------------------------------------------------------
    # GNSS distance
    # --------------------------------------------------------

    gps_n = gps_north[
        start_idx:end_idx + 1
    ]

    gps_e = gps_east[
        start_idx:end_idx + 1
    ]


    gps_distance = np.sum(
        np.sqrt(
            np.diff(
                gps_n
            ) ** 2
            +
            np.diff(
                gps_e
            ) ** 2
        )
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    final_error = errors[-1]

    mean_error = np.mean(
        errors
    )

    median_error = np.median(
        errors
    )

    max_error = np.max(
        errors
    )


    if gps_distance > 0:

        drift_percent = (
            final_error
            /
            gps_distance
            *
            100.0
        )

    else:

        drift_percent = np.nan


    return {

        "start_time_s":
            start_time,

        "end_time_s":
            end_time,

        "duration_s":
            duration,

        "start_index":
            start_idx,

        "end_index":
            end_idx,

        "moving_fraction":
            moving_fraction,

        "gps_distance_m":
            gps_distance,

        "mean_error_m":
            mean_error,

        "median_error_m":
            median_error,

        "final_error_m":
            final_error,

        "max_error_m":
            max_error,

        "drift_percent":
            drift_percent,

        "errors":
            errors,

        "trajectory_n":
            np.asarray(
                trajectory_n
            ),

        "trajectory_e":
            np.asarray(
                trajectory_e
            ),

        "gps_n":
            gps_n,

        "gps_e":
            gps_e
    }


# ============================================================
# RUN BENCHMARK
# ============================================================

results = []

trajectory_examples = []


print(
    "\n" + "=" * 75
)

print(
    "RUNNING MULTI-OUTAGE BENCHMARK"
)

print(
    "=" * 75
)


for duration in OUTAGE_DURATIONS:

    print(
        f"\nTesting {duration:.0f}-second outages..."
    )


    for start_time in CANDIDATE_STARTS:

        result = run_dead_reckoning(
            start_time,
            duration
        )


        if result is None:

            print(
                f"  {start_time:7.1f}s "
                f"-> skipped"
            )

            continue


        results.append(
            result
        )


        print(
            f"  {start_time:7.1f}s "
            f"-> "
            f"final={result['final_error_m']:.2f} m "
            f"mean={result['mean_error_m']:.2f} m"
        )


# ============================================================
# CHECK RESULTS
# ============================================================

if len(results) == 0:

    raise RuntimeError(
        "No valid moving outage windows found."
    )


results_df = pd.DataFrame([
    {
        "start_time_s":
            r["start_time_s"],

        "end_time_s":
            r["end_time_s"],

        "duration_s":
            r["duration_s"],

        "moving_fraction":
            r["moving_fraction"],

        "gps_distance_m":
            r["gps_distance_m"],

        "mean_error_m":
            r["mean_error_m"],

        "median_error_m":
            r["median_error_m"],

        "final_error_m":
            r["final_error_m"],

        "max_error_m":
            r["max_error_m"],

        "drift_percent":
            r["drift_percent"]
    }
    for r in results
])


# ============================================================
# SUMMARY BY OUTAGE DURATION
# ============================================================

summary = (
    results_df
    .groupby(
        "duration_s"
    )
    .agg(
        windows=(
            "final_error_m",
            "count"
        ),

        mean_final_error_m=(
            "final_error_m",
            "mean"
        ),

        median_final_error_m=(
            "final_error_m",
            "median"
        ),

        max_final_error_m=(
            "final_error_m",
            "max"
        ),

        mean_position_error_m=(
            "mean_error_m",
            "mean"
        ),

        mean_drift_percent=(
            "drift_percent",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "BENCHMARK SUMMARY"
)

print(
    "=" * 75
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.2f}"
    )
)


# ============================================================
# OVERALL
# ============================================================

overall_mean_final = (
    results_df[
        "final_error_m"
    ].mean()
)

overall_median_final = (
    results_df[
        "final_error_m"
    ].median()
)

overall_max_final = (
    results_df[
        "final_error_m"
    ].max()
)

overall_mean_drift = (
    results_df[
        "drift_percent"
    ].mean()
)


print(
    "\nOVERALL BENCHMARK"
)

print(
    f"Total valid outages:"
    f" {len(results_df)}"
)

print(
    f"Mean final error:"
    f" {overall_mean_final:.2f} m"
)

print(
    f"Median final error:"
    f" {overall_median_final:.2f} m"
)

print(
    f"Maximum final error:"
    f" {overall_max_final:.2f} m"
)

print(
    f"Mean relative drift:"
    f" {overall_mean_drift:.2f}%"
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results_file = os.path.join(
    PROCESSED_DIR,
    "prototype_v7_outage_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    PROCESSED_DIR,
    "prototype_v7_outage_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PLOT 1
#
# Final error vs outage duration
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)


for duration in OUTAGE_DURATIONS:

    subset = results_df[
        results_df[
            "duration_s"
        ]
        == duration
    ]


    if len(subset) == 0:

        continue


    ax.scatter(
        np.full(
            len(subset),
            duration
        ),
        subset[
            "final_error_m"
        ],
        s=70,
        label=f"{duration:.0f} s"
    )


ax.set_title(
    "AI Dead-Reckoning Final Position Error"
)

ax.set_xlabel(
    "GNSS outage duration (seconds)"
)

ax.set_ylabel(
    "Final position error (m)"
)

ax.grid(True)

ax.legend()

plt.tight_layout()


error_plot = os.path.join(
    OUTPUT_DIR,
    "prototype_v7_error_vs_outage.png"
)

plt.savefig(
    error_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 2
#
# Drift percentage
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)


for duration in OUTAGE_DURATIONS:

    subset = results_df[
        results_df[
            "duration_s"
        ]
        == duration
    ]


    if len(subset) == 0:

        continue


    ax.scatter(
        np.full(
            len(subset),
            duration
        ),
        subset[
            "drift_percent"
        ],
        s=70,
        label=f"{duration:.0f} s"
    )


ax.set_title(
    "Relative Position Drift During GNSS Outages"
)

ax.set_xlabel(
    "GNSS outage duration (seconds)"
)

ax.set_ylabel(
    "Relative drift (%)"
)

ax.grid(True)

ax.legend()

plt.tight_layout()


drift_plot = os.path.join(
    OUTPUT_DIR,
    "prototype_v7_drift_vs_outage.png"
)

plt.savefig(
    drift_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 3
#
# Example trajectories
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 9)
)


# Choose one example from each duration
for duration in OUTAGE_DURATIONS:

    matching = [
        r
        for r in results
        if r["duration_s"]
        == duration
    ]


    if len(matching) == 0:

        continue


    # Pick the median-error case
    matching = sorted(
        matching,
        key=lambda r:
        r["final_error_m"]
    )

    selected = matching[
        len(matching) // 2
    ]


    # Relative to outage start
    gn = (
        selected["gps_n"]
        -
        selected["gps_n"][0]
    )

    ge = (
        selected["gps_e"]
        -
        selected["gps_e"][0]
    )

    drn = (
        selected["trajectory_n"]
        -
        selected["trajectory_n"][0]
    )

    dre = (
        selected["trajectory_e"]
        -
        selected["trajectory_e"][0]
    )


    ax.plot(
        ge,
        gn,
        linewidth=2,
        label=(
            f"GNSS reference "
            f"({duration:.0f}s)"
        )
    )

    ax.plot(
        dre,
        drn,
        linestyle="--",
        linewidth=2,
        label=(
            f"AI DR "
            f"({duration:.0f}s)"
        )
    )


ax.set_title(
    "Representative GNSS-Denied Trajectories"
)

ax.set_xlabel(
    "East displacement (m)"
)

ax.set_ylabel(
    "North displacement (m)"
)

ax.grid(True)

ax.legend()

plt.tight_layout()


trajectory_plot = os.path.join(
    OUTPUT_DIR,
    "prototype_v7_trajectories.png"
)

plt.savefig(
    trajectory_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FILES CREATED"
)

print(
    "=" * 75
)

print(
    results_file
)

print(
    summary_file
)

print(
    error_plot
)

print(
    drift_plot
)

print(
    trajectory_plot
)

print(
    "\nV7 multi-outage benchmark completed successfully."
)