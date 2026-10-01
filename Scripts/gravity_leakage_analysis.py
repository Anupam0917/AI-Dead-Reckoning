import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 6.4
# STATIONARY GRAVITY LEAKAGE ANALYSIS
#
# Goal:
# Determine whether horizontal stationary acceleration
# residuals could be explained by small orientation errors
# causing gravity to leak into the horizontal plane.
# ============================================================


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOW_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "strict_stationary_windows.csv"
)

SAMPLE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "strict_stationary_samples.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ------------------------------------------------------------
# CONSTANT
# ------------------------------------------------------------

G = 9.80665  # standard gravity, m/s^2


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("=" * 70)
print("STEP 6.4: STATIONARY GRAVITY LEAKAGE ANALYSIS")
print("=" * 70)

print("\nLoading stationary-window data...")

if not os.path.exists(WINDOW_FILE):
    raise FileNotFoundError(
        f"\nStationary window file not found:\n{WINDOW_FILE}\n"
        "\nRun strict_stationary_detection.py first."
    )

if not os.path.exists(SAMPLE_FILE):
    raise FileNotFoundError(
        f"\nStationary sample file not found:\n{SAMPLE_FILE}\n"
        "\nRun strict_stationary_detection.py first."
    )


windows = pd.read_csv(WINDOW_FILE)
samples = pd.read_csv(SAMPLE_FILE)

print(f"Windows loaded : {len(windows)}")
print(f"Samples loaded : {len(samples)}")


# ------------------------------------------------------------
# DISPLAY COLUMNS
# ------------------------------------------------------------

print("\nWindow columns:")
for col in windows.columns:
    print("  ", col)

print("\nSample columns:")
for col in samples.columns:
    print("  ", col)


# ------------------------------------------------------------
# FIND NED ACCELERATION COLUMNS
# ------------------------------------------------------------

def find_column(df, possible_names):
    """
    Find the first matching column from a list of possible names.
    """

    for name in possible_names:
        if name in df.columns:
            return name

    return None


north_col = find_column(
    samples,
    [
        "NED_ACCEL_NORTH_m_s2",
        "NORTH_ACCELERATION_m_s2",
        "NED_NORTH_m_s2",
        "north",
        "North"
    ]
)

east_col = find_column(
    samples,
    [
        "NED_ACCEL_EAST_m_s2",
        "EAST_ACCELERATION_m_s2",
        "NED_EAST_m_s2",
        "east",
        "East"
    ]
)

down_col = find_column(
    samples,
    [
        "NED_ACCEL_DOWN_m_s2",
        "DOWN_ACCELERATION_m_s2",
        "NED_DOWN_m_s2",
        "down",
        "Down"
    ]
)


# ------------------------------------------------------------
# FALLBACK: SEARCH USING COLUMN TEXT
# ------------------------------------------------------------

if north_col is None:
    for col in samples.columns:
        lower = col.lower()

        if "north" in lower and (
            "accel" in lower or "ned" in lower
        ):
            north_col = col
            break


if east_col is None:
    for col in samples.columns:
        lower = col.lower()

        if "east" in lower and (
            "accel" in lower or "ned" in lower
        ):
            east_col = col
            break


if down_col is None:
    for col in samples.columns:
        lower = col.lower()

        if "down" in lower and (
            "accel" in lower or "ned" in lower
        ):
            down_col = col
            break


print("\nDetected NED acceleration columns:")

print("North:", north_col)
print("East :", east_col)
print("Down :", down_col)


if north_col is None or east_col is None or down_col is None:

    raise ValueError(
        "\nCould not automatically identify NED acceleration columns.\n"
        "Look at the printed sample columns above and update the "
        "north_col/east_col/down_col variables."
    )


# ------------------------------------------------------------
# CLEAN DATA
# ------------------------------------------------------------

samples[north_col] = pd.to_numeric(
    samples[north_col],
    errors="coerce"
)

samples[east_col] = pd.to_numeric(
    samples[east_col],
    errors="coerce"
)

samples[down_col] = pd.to_numeric(
    samples[down_col],
    errors="coerce"
)


# ------------------------------------------------------------
# FIND WINDOW BOUNDARY COLUMNS
# ------------------------------------------------------------

start_col = find_column(
    windows,
    [
        "start_time_s",
        "START_TIME_s",
        "start",
        "START",
        "start_seconds"
    ]
)

end_col = find_column(
    windows,
    [
        "end_time_s",
        "END_TIME_s",
        "end",
        "END",
        "end_seconds"
    ]
)


# ------------------------------------------------------------
# FALLBACK SEARCH FOR TIME COLUMNS
# ------------------------------------------------------------

if start_col is None:

    for col in windows.columns:

        lower = col.lower()

        if "start" in lower and (
            "time" in lower or "second" in lower
        ):
            start_col = col
            break


if end_col is None:

    for col in windows.columns:

        lower = col.lower()

        if "end" in lower and (
            "time" in lower or "second" in lower
        ):
            end_col = col
            break


print("\nDetected window boundary columns:")

print("Start:", start_col)
print("End  :", end_col)


if start_col is None or end_col is None:

    raise ValueError(
        "\nCould not automatically identify window start/end columns.\n"
        "Check the printed window columns and update the script."
    )


windows[start_col] = pd.to_numeric(
    windows[start_col],
    errors="coerce"
)

windows[end_col] = pd.to_numeric(
    windows[end_col],
    errors="coerce"
)


# ------------------------------------------------------------
# FIND TIME COLUMN IN SAMPLE DATA
# ------------------------------------------------------------

time_col = find_column(
    samples,
    [
        "time_s",
        "TIME_s",
        "timestamp_s",
        "TIME_SINCE_START_s",
        "time"
    ]
)


if time_col is None:

    for col in samples.columns:

        lower = col.lower()

        if (
            "time" in lower
            and "start" in lower
            and (
                "sec" in lower
                or "_s" in lower
            )
        ):
            time_col = col
            break


print("\nDetected sample time column:", time_col)


# ------------------------------------------------------------
# IF TIME COLUMN EXISTS
# ------------------------------------------------------------

if time_col is not None:

    samples[time_col] = pd.to_numeric(
        samples[time_col],
        errors="coerce"
    )


# ------------------------------------------------------------
# ANALYZE EACH STATIONARY WINDOW
# ------------------------------------------------------------

results = []


print("\n")
print("=" * 70)
print("WINDOW-BY-WINDOW GRAVITY LEAKAGE")
print("=" * 70)


for index, window in windows.iterrows():

    start_time = window[start_col]
    end_time = window[end_col]

    # --------------------------------------------------------
    # SELECT SAMPLES
    # --------------------------------------------------------

    if time_col is not None:

        mask = (
            (samples[time_col] >= start_time)
            &
            (samples[time_col] <= end_time)
        )

        window_samples = samples.loc[mask]

    else:

        # Fallback:
        # If no time column exists, use the stored window
        # sample indices if available.

        start_index_col = find_column(
            windows,
            [
                "start_index",
                "START_INDEX",
                "start_row"
            ]
        )

        end_index_col = find_column(
            windows,
            [
                "end_index",
                "END_INDEX",
                "end_row"
            ]
        )

        if start_index_col is None or end_index_col is None:

            raise ValueError(
                "\nNo usable time or index information found "
                "for stationary windows."
            )

        start_index = int(window[start_index_col])
        end_index = int(window[end_index_col])

        window_samples = samples.iloc[
            start_index:end_index + 1
        ]


    # --------------------------------------------------------
    # REMOVE NaN VALUES
    # --------------------------------------------------------

    window_samples = window_samples[
        [
            north_col,
            east_col,
            down_col
        ]
    ].dropna()


    if len(window_samples) == 0:

        continue


    # --------------------------------------------------------
    # MEAN STATIONARY ACCELERATION
    # --------------------------------------------------------

    north = window_samples[north_col].to_numpy()
    east = window_samples[east_col].to_numpy()
    down = window_samples[down_col].to_numpy()


    north_mean = np.mean(north)
    east_mean = np.mean(east)
    down_mean = np.mean(down)


    # --------------------------------------------------------
    # STANDARD DEVIATION
    # --------------------------------------------------------

    north_std = np.std(north)
    east_std = np.std(east)
    down_std = np.std(down)


    # --------------------------------------------------------
    # HORIZONTAL RESIDUAL
    # --------------------------------------------------------

    horizontal_residual = np.sqrt(
        north_mean ** 2
        +
        east_mean ** 2
    )


    # --------------------------------------------------------
    # TOTAL RESIDUAL
    # --------------------------------------------------------

    total_residual = np.sqrt(
        north_mean ** 2
        +
        east_mean ** 2
        +
        down_mean ** 2
    )


    # --------------------------------------------------------
    # EQUIVALENT TILT ANGLE
    #
    # If gravity leaks into the horizontal plane:
    #
    # horizontal_error ≈ g * sin(theta)
    #
    # Therefore:
    #
    # theta = asin(horizontal_error / g)
    #
    # For small angles:
    #
    # theta ≈ horizontal_error / g
    # --------------------------------------------------------

    ratio = horizontal_residual / G

    ratio = np.clip(
        ratio,
        0.0,
        1.0
    )

    equivalent_tilt_rad = np.arcsin(ratio)

    equivalent_tilt_deg = np.degrees(
        equivalent_tilt_rad
    )


    # --------------------------------------------------------
    # NORTH/EAST CONTRIBUTION
    # --------------------------------------------------------

    north_tilt_deg = np.degrees(
        np.arctan2(
            abs(north_mean),
            G
        )
    )

    east_tilt_deg = np.degrees(
        np.arctan2(
            abs(east_mean),
            G
        )
    )


    # --------------------------------------------------------
    # HORIZONTAL ERROR DIRECTION
    # --------------------------------------------------------

    horizontal_direction_deg = np.degrees(
        np.arctan2(
            east_mean,
            north_mean
        )
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "window_number": index + 1,

            "start_time_s": start_time,
            "end_time_s": end_time,

            "duration_s": end_time - start_time,

            "samples": len(window_samples),

            "north_mean_m_s2": north_mean,
            "east_mean_m_s2": east_mean,
            "down_mean_m_s2": down_mean,

            "north_std_m_s2": north_std,
            "east_std_m_s2": east_std,
            "down_std_m_s2": down_std,

            "horizontal_residual_m_s2":
                horizontal_residual,

            "total_residual_m_s2":
                total_residual,

            "equivalent_tilt_deg":
                equivalent_tilt_deg,

            "north_equivalent_tilt_deg":
                north_tilt_deg,

            "east_equivalent_tilt_deg":
                east_tilt_deg,

            "horizontal_error_direction_deg":
                horizontal_direction_deg
        }
    )


# ------------------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------------------

result_df = pd.DataFrame(results)


if len(result_df) == 0:

    raise ValueError(
        "\nNo stationary-window results were generated."
    )


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print(
    "\n"
    f"{'Window':>7} "
    f"{'N':>10} "
    f"{'E':>10} "
    f"{'D':>10} "
    f"{'Horiz':>10} "
    f"{'Tilt(deg)':>12}"
)

print("-" * 70)


for _, row in result_df.iterrows():

    print(
        f"{int(row['window_number']):7d} "
        f"{row['north_mean_m_s2']:10.4f} "
        f"{row['east_mean_m_s2']:10.4f} "
        f"{row['down_mean_m_s2']:10.4f} "
        f"{row['horizontal_residual_m_s2']:10.4f} "
        f"{row['equivalent_tilt_deg']:12.4f}"
    )


# ------------------------------------------------------------
# GLOBAL STATISTICS
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("GLOBAL GRAVITY LEAKAGE STATISTICS")
print("=" * 70)


horizontal_values = result_df[
    "horizontal_residual_m_s2"
].to_numpy()

tilt_values = result_df[
    "equivalent_tilt_deg"
].to_numpy()


print(
    f"\nHorizontal residual:"
)

print(
    f"  Mean   : {np.mean(horizontal_values):.6f} m/s²"
)

print(
    f"  Median : {np.median(horizontal_values):.6f} m/s²"
)

print(
    f"  Std    : {np.std(horizontal_values):.6f} m/s²"
)

print(
    f"  Min    : {np.min(horizontal_values):.6f} m/s²"
)

print(
    f"  Max    : {np.max(horizontal_values):.6f} m/s²"
)


print(
    f"\nEquivalent tilt:"
)

print(
    f"  Mean   : {np.mean(tilt_values):.6f}°"
)

print(
    f"  Median : {np.median(tilt_values):.6f}°"
)

print(
    f"  Std    : {np.std(tilt_values):.6f}°"
)

print(
    f"  Min    : {np.min(tilt_values):.6f}°"
)

print(
    f"  Max    : {np.max(tilt_values):.6f}°"
)


# ------------------------------------------------------------
# OVERALL MEAN RESIDUAL
# ------------------------------------------------------------

global_north = result_df[
    "north_mean_m_s2"
].mean()

global_east = result_df[
    "east_mean_m_s2"
].mean()

global_down = result_df[
    "down_mean_m_s2"
].mean()


global_horizontal = np.sqrt(
    global_north ** 2
    +
    global_east ** 2
)


global_tilt = np.degrees(
    np.arcsin(
        np.clip(
            global_horizontal / G,
            0,
            1
        )
    )
)


print("\nGlobal mean residual:")

print(
    f"  North : {global_north:.6f} m/s²"
)

print(
    f"  East  : {global_east:.6f} m/s²"
)

print(
    f"  Down  : {global_down:.6f} m/s²"
)

print(
    f"  Horizontal : {global_horizontal:.6f} m/s²"
)

print(
    f"  Equivalent tilt : {global_tilt:.6f}°"
)


# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

output_csv = os.path.join(
    PROCESSED_DIR,
    "gravity_leakage_analysis.csv"
)

result_df.to_csv(
    output_csv,
    index=False
)

print(
    f"\nSaved analysis CSV:\n{output_csv}"
)


# ============================================================
# PLOTS
# ============================================================

print("\nGenerating plots...")


# ------------------------------------------------------------
# PLOT 1
# HORIZONTAL RESIDUAL PER WINDOW
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    result_df["window_number"],
    result_df["horizontal_residual_m_s2"],
    marker="o"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Stationary Window Number"
)

plt.ylabel(
    "Horizontal Residual Acceleration (m/s²)"
)

plt.title(
    "Stationary Horizontal Acceleration Residual"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plot1 = os.path.join(
    OUTPUT_DIR,
    "gravity_horizontal_residual.png"
)

plt.savefig(
    plot1,
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# PLOT 2
# EQUIVALENT TILT
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    result_df["window_number"],
    result_df["equivalent_tilt_deg"],
    marker="o"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Stationary Window Number"
)

plt.ylabel(
    "Equivalent Tilt Error (degrees)"
)

plt.title(
    "Equivalent Attitude Error from Gravity Leakage"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plot2 = os.path.join(
    OUTPUT_DIR,
    "gravity_equivalent_tilt.png"
)

plt.savefig(
    plot2,
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# PLOT 3
# N/E/D RESIDUALS
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    result_df["window_number"],
    result_df["north_mean_m_s2"],
    marker="o",
    label="North"
)

plt.plot(
    result_df["window_number"],
    result_df["east_mean_m_s2"],
    marker="s",
    label="East"
)

plt.plot(
    result_df["window_number"],
    result_df["down_mean_m_s2"],
    marker="^",
    label="Down"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Stationary Window Number"
)

plt.ylabel(
    "Mean Acceleration Residual (m/s²)"
)

plt.title(
    "Stationary NED Acceleration Residuals"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plot3 = os.path.join(
    OUTPUT_DIR,
    "gravity_ned_residuals.png"
)

plt.savefig(
    plot3,
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# PLOT 4
# RESIDUAL VS EQUIVALENT TILT
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    result_df["horizontal_residual_m_s2"],
    result_df["equivalent_tilt_deg"]
)

plt.xlabel(
    "Horizontal Residual Acceleration (m/s²)"
)

plt.ylabel(
    "Equivalent Tilt (degrees)"
)

plt.title(
    "Gravity Leakage Relationship"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plot4 = os.path.join(
    OUTPUT_DIR,
    "gravity_leakage_relationship.png"
)

plt.savefig(
    plot4,
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# FINAL MESSAGE
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("STEP 6.4 COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    f"1. {output_csv}"
)

print(
    f"2. {plot1}"
)

print(
    f"3. {plot2}"
)

print(
    f"4. {plot3}"
)

print(
    f"5. {plot4}"
)

print("\nNext step:")
print(
    "Inspect the equivalent tilt and N/E residual patterns "
    "before applying any acceleration-bias correction."
)

print("\nDone.")