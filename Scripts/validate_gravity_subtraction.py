import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

RAW_DATA_PATH = Path(
    "data/raw/Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

WINDOWS_PATH = Path(
    "data/processed/strict_stationary_windows.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

ACCEL_BIAS = np.array([
    -0.00645854,
     0.00997073,
     0.03871098
])


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data():

    print("=" * 80)
    print("STEP 6.9 - GRAVITY SUBTRACTION VALIDATION")
    print("=" * 80)

    print("\nLoading raw dataset...")

    df = pd.read_csv(
        RAW_DATA_PATH,
        encoding="cp1252"
    )

    print(f"Raw rows: {len(df)}")

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    date_column = "DATE_YYYY-MO-DD_HH-MI-SS_SSS"

    df["timestamp"] = pd.to_datetime(
        df[date_column],
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

    df["time_seconds"] = (
        df["timestamp"] - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    return df


# ============================================================
# LOAD STATIONARY WINDOWS
# ============================================================

def load_windows():

    print("\nLoading strict stationary windows...")

    windows = pd.read_csv(
        WINDOWS_PATH
    )

    print(
        f"Stationary windows found: "
        f"{len(windows)}"
    )

    print("\nWindows:")

    for _, row in windows.iterrows():

        print(
            f"  {row['start_time_s']:.2f} - "
            f"{row['end_time_s']:.2f} s"
        )

    return windows

# ============================================================
# BUILD STATIONARY MASK
# ============================================================

def build_stationary_mask(
    df,
    windows
):

    print("\nBuilding stationary mask...")

    time = df["time_seconds"].to_numpy()

    mask = np.zeros(
        len(df),
        dtype=bool
    )

    for _, row in windows.iterrows():

        start = row["start_time_s"]
        end = row["end_time_s"]

        mask |= (
            (time >= start)
            &
            (time <= end)
        )

    print(
        "\nStationary samples selected:"
    )

    print(
        f"{np.sum(mask)} / {len(df)}"
    )

    return mask

# ============================================================
# FIND SENSOR COLUMNS
# ============================================================

def find_columns(df):

    accel_columns = [
        "ACCELEROMETER_X_m_s²",
        "ACCELEROMETER_Y_m_s²",
        "ACCELEROMETER_Z_m_s²"
    ]

    gravity_columns = [
        "GRAVITY_X_m_s²",
        "GRAVITY_Y_m_s²",
        "GRAVITY_Z_m_s²"
    ]

    return accel_columns, gravity_columns


# ============================================================
# ANALYSIS
# ============================================================

def analyze(
    df,
    accel_columns,
    gravity_columns,
    stationary_mask
):

    accel = df[
        accel_columns
    ].to_numpy(dtype=float)

    gravity = df[
        gravity_columns
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Bias correction
    # --------------------------------------------------------

    corrected_accel = (
        accel - ACCEL_BIAS
    )

    # --------------------------------------------------------
    # Gravity subtraction
    # --------------------------------------------------------

    linear_accel = (
        corrected_accel - gravity
    )

    # --------------------------------------------------------
    # Stationary samples only
    # --------------------------------------------------------

    a = corrected_accel[
        stationary_mask
    ]

    g = gravity[
        stationary_mask
    ]

    linear = linear_accel[
        stationary_mask
    ]

    # ========================================================
    # MAGNITUDES
    # ========================================================

    accel_mag = np.linalg.norm(
        a,
        axis=1
    )

    gravity_mag = np.linalg.norm(
        g,
        axis=1
    )

    residual_mag = np.linalg.norm(
        linear,
        axis=1
    )

    print("\n" + "=" * 80)
    print("1. MAGNITUDE VALIDATION")
    print("=" * 80)

    def stats(name, values):

        print(f"\n{name}")

        print(
            f"  Mean   : {np.mean(values):.6f}"
        )

        print(
            f"  Median : {np.median(values):.6f}"
        )

        print(
            f"  Std    : {np.std(values):.6f}"
        )

        print(
            f"  Min    : {np.min(values):.6f}"
        )

        print(
            f"  Max    : {np.max(values):.6f}"
        )

    stats(
        "Corrected accelerometer magnitude (m/s²)",
        accel_mag
    )

    stats(
        "Gravity sensor magnitude (m/s²)",
        gravity_mag
    )

    stats(
        "Linear acceleration residual magnitude (m/s²)",
        residual_mag
    )

    # ========================================================
    # COMPONENTS
    # ========================================================

    print("\n" + "=" * 80)
    print("2. COMPONENT-WISE STATIONARY RESIDUAL")
    print("=" * 80)

    for i, axis in enumerate(["X", "Y", "Z"]):

        print(f"\n{axis} axis")

        print(
            f"  Corrected accel mean : "
            f"{np.mean(a[:, i]):.6f}"
        )

        print(
            f"  Gravity mean         : "
            f"{np.mean(g[:, i]):.6f}"
        )

        print(
            f"  Residual mean        : "
            f"{np.mean(linear[:, i]):.6f}"
        )

        print(
            f"  Residual std         : "
            f"{np.std(linear[:, i]):.6f}"
        )

    # ========================================================
    # CORRELATION
    # ========================================================

    print("\n" + "=" * 80)
    print("3. ACCELEROMETER vs GRAVITY CORRELATION")
    print("=" * 80)

    for i, axis in enumerate(["X", "Y", "Z"]):

        corr = np.corrcoef(
            a[:, i],
            g[:, i]
        )[0, 1]

        print(
            f"{axis}: {corr:.6f}"
        )

    # ========================================================
    # RESIDUAL THRESHOLDS
    # ========================================================

    print("\n" + "=" * 80)
    print("4. RESIDUAL QUALITY")
    print("=" * 80)

    for threshold in [
        0.05,
        0.10,
        0.20,
        0.50,
        1.00
    ]:

        count = np.sum(
            residual_mag < threshold
        )

        percentage = (
            count /
            len(residual_mag)
            * 100
        )

        print(
            f"< {threshold:.2f} m/s² : "
            f"{count}/{len(residual_mag)} "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # GRAVITY MAGNITUDE ERROR
    # ========================================================

    print("\n" + "=" * 80)
    print("5. GRAVITY MAGNITUDE ERROR")
    print("=" * 80)

    gravity_error = (
        accel_mag - gravity_mag
    )

    stats(
        "Accelerometer magnitude - gravity magnitude",
        gravity_error
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 6.9 COMPLETE")
    print("=" * 80)

    print(
        f"\nActual stationary samples analyzed: "
        f"{len(residual_mag)}"
    )

    print(
        f"Mean stationary residual: "
        f"{np.mean(residual_mag):.6f} m/s²"
    )

    print(
        f"Median stationary residual: "
        f"{np.median(residual_mag):.6f} m/s²"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_raw_data()

    windows = load_windows()

    mask = build_stationary_mask(
        df,
        windows
    )

    accel_columns, gravity_columns = (
        find_columns(df)
    )

    analyze(
        df,
        accel_columns,
        gravity_columns,
        mask
    )


if __name__ == "__main__":
    main()