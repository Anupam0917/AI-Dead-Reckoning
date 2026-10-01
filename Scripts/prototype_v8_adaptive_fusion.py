import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_absolute_error


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    "data/raw/"
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

AI_SPEED_PATH = "data/processed/prototype_ai_speed_results.csv"
V5_HEADING_PATH = "data/processed/prototype_v5_imu_heading.csv"
V7_RESULTS_PATH = "data/processed/prototype_v7_outage_results.csv"

OUTPUT_RESULTS = "data/processed/prototype_v8_adaptive_fusion_results.csv"
OUTPUT_SUMMARY = "data/processed/prototype_v8_adaptive_fusion_summary.csv"

OUTPUT_PLOT = "outputs/prototype_v8_adaptive_fusion.png"
OUTPUT_RECOVERY = "outputs/prototype_v8_recovery.png"


# ============================================================
# SETTINGS
# ============================================================

CALIBRATION_SECONDS = 60.0

# Kinematic limits
MAX_ACCEL_MPS2 = 4.0
MAX_DECEL_MPS2 = 5.0

# Heading rate limit
MAX_HEADING_RATE_RAD_S = np.deg2rad(120.0)

# EKF parameters
BASE_POSITION_NOISE = 2.0
BASE_VELOCITY_NOISE = 1.5

EARTH_RADIUS_M = 6371000.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def wrap_angle_deg(angle):
    """
    Wrap angle to [-180, 180].
    """
    return (angle + 180.0) % 360.0 - 180.0


def circular_mean_deg(values):
    """
    Circular mean in degrees.
    """
    values = np.asarray(values)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    radians = np.deg2rad(values)

    return np.rad2deg(
        np.arctan2(
            np.mean(np.sin(radians)),
            np.mean(np.cos(radians))
        )
    )


def latlon_to_local(lat, lon, lat0, lon0):
    """
    Convert latitude/longitude to local North/East coordinates.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    lat0_rad = np.deg2rad(lat0)

    north = np.deg2rad(lat - lat0) * EARTH_RADIUS_M

    east = (
        np.deg2rad(lon - lon0)
        * EARTH_RADIUS_M
        * np.cos(lat0_rad)
    )

    return north, east


def calculate_heading_error_deg(a, b):
    """
    Circular difference a-b.
    """
    return wrap_angle_deg(a - b)


# ============================================================
# EKF CLASS
# ============================================================

class AdaptiveVelocityEKF:

    def __init__(self, north, east, vn, ve):

        # State:
        # x = [North, East, Vnorth, Veast]

        self.x = np.array(
            [north, east, vn, ve],
            dtype=float
        )

        self.P = np.diag([
            4.0,
            4.0,
            4.0,
            4.0
        ])

    def predict(self, dt, process_noise):

        if dt <= 0:
            return

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        self.x = F @ self.x

        q_pos = process_noise * dt * dt
        q_vel = process_noise * dt

        Q = np.diag([
            q_pos,
            q_pos,
            q_vel,
            q_vel
        ])

        self.P = F @ self.P @ F.T + Q

    def update_velocity(
        self,
        vn_measurement,
        ve_measurement,
        measurement_noise
    ):

        z = np.array([
            vn_measurement,
            ve_measurement
        ])

        H = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        R = np.diag([
            measurement_noise,
            measurement_noise
        ])

        y = z - H @ self.x

        S = H @ self.P @ H.T + R

        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(4)

        self.P = (I - K @ H) @ self.P

    def update_position(
        self,
        north_measurement,
        east_measurement
    ):

        z = np.array([
            north_measurement,
            east_measurement
        ])

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        R = np.diag([
            BASE_POSITION_NOISE,
            BASE_POSITION_NOISE
        ])

        y = z - H @ self.x

        S = H @ self.P @ H.T + R

        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(4)

        self.P = (I - K @ H) @ self.P


# ============================================================
# LOAD RAW DATA
# ============================================================

print("=" * 70)
print("V8 ADAPTIVE GNSS-AIDED DEAD RECKONING")
print("=" * 70)

print("\nLoading raw dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252"
)

print(f"Rows loaded: {len(df)}")


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_", regex=False)
)

# Locate columns safely
def find_column(text):
    for col in df.columns:
        if text.lower() in col.lower():
            return col

    raise KeyError(f"Could not find column containing: {text}")


LAT_COL = find_column("GPS_LATITUDE")
LON_COL = find_column("GPS_LONGITUDE")
SPEED_COL = find_column("GPS_SPEED")
GPS_HEADING_COL = find_column("GPS_ORIENTATION")
TIME_COL = find_column("DATE_YYYY")

print("\nDetected columns:")
print("Latitude :", LAT_COL)
print("Longitude:", LON_COL)
print("GPS speed:", SPEED_COL)
print("GPS heading:", GPS_HEADING_COL)
print("Timestamp:", TIME_COL)


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df[TIME_COL],
    format="%Y-%m-%d_%H-%M-%S_%f",
    errors="coerce"
)

# If the first parser did not work, use original timestamp format.
if df["timestamp"].isna().mean() > 0.5:

    raw_time = df[TIME_COL].astype(str)

    df["timestamp"] = pd.to_datetime(
        raw_time,
        format="%Y-%m-%d_%H-%M-%S_%f",
        errors="coerce"
    )

# Fallback for original dataset representation
if df["timestamp"].isna().mean() > 0.5:

    raw_time = df[TIME_COL].astype(str)

    df["timestamp"] = pd.to_datetime(
        raw_time,
        format="%Y-%m-%d %H:%M:%S:%f",
        errors="coerce"
    )

if df["timestamp"].isna().all():
    raise RuntimeError("Timestamp parsing failed.")


df = df.sort_values("timestamp").reset_index(drop=True)

df["time_s"] = (
    df["timestamp"] - df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# NUMERIC DATA
# ============================================================

df["gps_speed_kmh"] = pd.to_numeric(
    df[SPEED_COL],
    errors="coerce"
)

df["gps_heading_deg"] = pd.to_numeric(
    df[GPS_HEADING_COL],
    errors="coerce"
)

df["latitude"] = pd.to_numeric(
    df[LAT_COL],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df[LON_COL],
    errors="coerce"
)


# ============================================================
# LOAD V1 AI SPEED
# ============================================================

print("\nLoading V1 AI speed...")

ai = pd.read_csv(AI_SPEED_PATH)

ai["timestamp"] = pd.to_datetime(
    ai["timestamp"],
    errors="coerce"
)

ai = ai.sort_values("timestamp")

ai = ai[
    [
        "timestamp",
        "ai_speed_kmh"
    ]
].copy()


# Merge AI speed
df = pd.merge_asof(
    df.sort_values("timestamp"),
    ai.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("200ms")
)

df["ai_speed_kmh"] = pd.to_numeric(
    df["ai_speed_kmh"],
    errors="coerce"
)

df["ai_speed_kmh"] = (
    df["ai_speed_kmh"]
    .interpolate()
    .ffill()
    .bfill()
)


# ============================================================
# LOAD V5 HEADING
# ============================================================

print("Loading V5 heading...")

heading = pd.read_csv(V5_HEADING_PATH)

heading["timestamp"] = pd.to_datetime(
    heading["timestamp"],
    errors="coerce"
)

# Find heading column
heading_col = None

for candidate in [
    "estimated_heading_deg",
    "fused_heading_deg",
    "heading_deg",
    "estimated_heading"
]:

    if candidate in heading.columns:
        heading_col = candidate
        break


if heading_col is None:

    possible = [
        c for c in heading.columns
        if "heading" in c.lower()
    ]

    if not possible:
        raise KeyError(
            "Could not find heading column in V5 output."
        )

    heading_col = possible[0]


print("Using V5 heading column:", heading_col)

heading = heading[
    [
        "timestamp",
        heading_col
    ]
].copy()

heading = heading.rename(
    columns={
        heading_col: "v5_heading_deg"
    }
)

heading["v5_heading_deg"] = pd.to_numeric(
    heading["v5_heading_deg"],
    errors="coerce"
)

heading = heading.sort_values("timestamp")

df = pd.merge_asof(
    df.sort_values("timestamp"),
    heading,
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("200ms")
)

df["v5_heading_deg"] = (
    df["v5_heading_deg"]
    .interpolate()
    .ffill()
    .bfill()
)


# ============================================================
# LOCAL N/E COORDINATES
# ============================================================

lat0 = df["latitude"].iloc[0]
lon0 = df["longitude"].iloc[0]

df["gps_north_m"], df["gps_east_m"] = latlon_to_local(
    df["latitude"].values,
    df["longitude"].values,
    lat0,
    lon0
)


# ============================================================
# GPS VELOCITY
# ============================================================

dt = df["time_s"].diff()

df["dt"] = dt

df["gps_vn"] = df["gps_north_m"].diff() / dt
df["gps_ve"] = df["gps_east_m"].diff() / dt

df["gps_vn"] = (
    df["gps_vn"]
    .replace([np.inf, -np.inf], np.nan)
    .rolling(5, center=True, min_periods=1)
    .median()
)

df["gps_ve"] = (
    df["gps_ve"]
    .replace([np.inf, -np.inf], np.nan)
    .rolling(5, center=True, min_periods=1)
    .median()
)


# ============================================================
# LOAD V7 OUTAGE WINDOWS
# ============================================================

print("\nLoading V7 benchmark windows...")

v7 = pd.read_csv(V7_RESULTS_PATH)

print(
    f"V7 contains {len(v7)} outage windows."
)


# ============================================================
# CALIBRATION FUNCTION
# ============================================================

def calibrate_window(data, start_time):

    calibration_start = max(
        0.0,
        start_time - CALIBRATION_SECONDS
    )

    mask = (
        (data["time_s"] >= calibration_start)
        &
        (data["time_s"] < start_time)
        &
        (data["gps_speed_kmh"] > 2.0)
        &
        (data["ai_speed_kmh"] > 1.0)
    )

    c = data.loc[mask].copy()

    if len(c) < 20:

        return {
            "speed_scale": 1.0,
            "speed_bias": 0.0,
            "heading_offset": 0.0,
            "speed_std": 3.0,
            "heading_std": 30.0,
            "samples": len(c)
        }

    # --------------------------------------------------------
    # SPEED CALIBRATION
    # --------------------------------------------------------

    x = c["ai_speed_kmh"].values
    y = c["gps_speed_kmh"].values

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
        &
        (x > 1)
    )

    x = x[valid]
    y = y[valid]

    if len(x) >= 20:

        A = np.vstack([
            x,
            np.ones(len(x))
        ]).T

        scale, bias = np.linalg.lstsq(
            A,
            y,
            rcond=None
        )[0]

        scale = float(
            np.clip(scale, 0.80, 1.20)
        )

        bias = float(
            np.clip(bias, -5.0, 5.0)
        )

    else:

        scale = 1.0
        bias = 0.0


    # --------------------------------------------------------
    # HEADING CALIBRATION
    # --------------------------------------------------------

    heading_difference = calculate_heading_error_deg(
        c["gps_heading_deg"].values,
        c["v5_heading_deg"].values
    )

    heading_difference = heading_difference[
        np.isfinite(heading_difference)
    ]

    if len(heading_difference) > 10:

        heading_offset = circular_mean_deg(
            heading_difference
        )

        heading_std = np.std(
            wrap_angle_deg(
                heading_difference - heading_offset
            )
        )

    else:

        heading_offset = 0.0
        heading_std = 30.0


    # --------------------------------------------------------
    # SPEED ERROR
    # --------------------------------------------------------

    corrected_speed = (
        scale * x + bias
    )

    speed_error = y - corrected_speed

    speed_std = np.std(speed_error)

    return {
        "speed_scale": scale,
        "speed_bias": bias,
        "heading_offset": heading_offset,
        "speed_std": float(
            np.clip(speed_std, 0.5, 10.0)
        ),
        "heading_std": float(
            np.clip(heading_std, 2.0, 60.0)
        ),
        "samples": len(c)
    }


# ============================================================
# KINEMATIC SPEED CONSTRAINT
# ============================================================

def apply_speed_constraint(
    current_speed,
    previous_speed,
    dt_value
):

    if not np.isfinite(previous_speed):
        return max(0.0, current_speed)

    if dt_value <= 0:
        return max(0.0, current_speed)

    delta = current_speed - previous_speed

    max_increase = (
        MAX_ACCEL_MPS2
        * dt_value
        * 3.6
    )

    max_decrease = (
        MAX_DECEL_MPS2
        * dt_value
        * 3.6
    )

    delta = np.clip(
        delta,
        -max_decrease,
        max_increase
    )

    result = previous_speed + delta

    return max(
        0.0,
        result
    )


# ============================================================
# PROCESS ONE OUTAGE
# ============================================================

def process_outage(data, start_time, end_time):

    outage_mask = (
        (data["time_s"] >= start_time)
        &
        (data["time_s"] <= end_time)
    )

    indices = np.where(
        outage_mask.values
    )[0]

    if len(indices) < 5:
        return None

    first_idx = indices[0]

    last_idx = indices[-1]

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    calibration = calibrate_window(
        data,
        start_time
    )

    speed_scale = calibration["speed_scale"]
    speed_bias = calibration["speed_bias"]

    heading_offset = calibration["heading_offset"]

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    north0 = data.loc[
        first_idx,
        "gps_north_m"
    ]

    east0 = data.loc[
        first_idx,
        "gps_east_m"
    ]

    previous_idx = max(
        0,
        first_idx - 1
    )

    initial_speed = (
        speed_scale
        * data.loc[
            previous_idx,
            "ai_speed_kmh"
        ]
        + speed_bias
    )

    initial_heading = wrap_angle_deg(
        data.loc[
            previous_idx,
            "v5_heading_deg"
        ]
        + heading_offset
    )

    initial_heading_rad = np.deg2rad(
        initial_heading
    )

    initial_vn = (
        initial_speed
        / 3.6
        * np.cos(initial_heading_rad)
    )

    initial_ve = (
        initial_speed
        / 3.6
        * np.sin(initial_heading_rad)
    )

    ekf = AdaptiveVelocityEKF(
        north0,
        east0,
        initial_vn,
        initial_ve
    )

    previous_speed = initial_speed

    previous_heading = initial_heading

    rows = []

    # --------------------------------------------------------
    # RUN NAVIGATION
    # --------------------------------------------------------

    for idx in range(
        first_idx,
        last_idx + 1
    ):

        row = data.loc[idx]

        if idx == first_idx:

            dt_value = 0.1

        else:

            dt_value = (
                row["time_s"]
                -
                data.loc[
                    idx - 1,
                    "time_s"
                ]
            )

            if (
                not np.isfinite(dt_value)
                or dt_value <= 0
                or dt_value > 1.0
            ):

                dt_value = 0.1

        # ----------------------------------------------------
        # AI SPEED CORRECTION
        # ----------------------------------------------------

        raw_ai_speed = row[
            "ai_speed_kmh"
        ]

        corrected_speed = (
            speed_scale
            * raw_ai_speed
            + speed_bias
        )

        corrected_speed = apply_speed_constraint(
            corrected_speed,
            previous_speed,
            dt_value
        )

        # ----------------------------------------------------
        # HEADING CORRECTION
        # ----------------------------------------------------

        raw_heading = row[
            "v5_heading_deg"
        ]

        corrected_heading = wrap_angle_deg(
            raw_heading
            + heading_offset
        )

        heading_delta = wrap_angle_deg(
            corrected_heading
            - previous_heading
        )

        max_heading_delta = (
            np.rad2deg(
                MAX_HEADING_RATE_RAD_S
                * dt_value
            )
        )

        heading_delta = np.clip(
            heading_delta,
            -max_heading_delta,
            max_heading_delta
        )

        corrected_heading = wrap_angle_deg(
            previous_heading
            + heading_delta
        )

        # ----------------------------------------------------
        # VELOCITY
        # ----------------------------------------------------

        heading_rad = np.deg2rad(
            corrected_heading
        )

        speed_mps = (
            corrected_speed / 3.6
        )

        vn = (
            speed_mps
            * np.cos(heading_rad)
        )

        ve = (
            speed_mps
            * np.sin(heading_rad)
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        speed_confidence = np.exp(
            -calibration["speed_std"]
            / 5.0
        )

        heading_confidence = np.exp(
            -calibration["heading_std"]
            / 30.0
        )

        elapsed = (
            row["time_s"]
            - start_time
        )

        time_confidence = np.exp(
            -max(0.0, elapsed)
            / 180.0
        )

        confidence = (
            speed_confidence
            * heading_confidence
            * time_confidence
        )

        confidence = float(
            np.clip(
                confidence,
                0.05,
                1.0
            )
        )

        # ----------------------------------------------------
        # ADAPTIVE PROCESS NOISE
        # ----------------------------------------------------

        process_noise = (
            0.5
            +
            (1.0 - confidence)
            * 8.0
        )

        # ----------------------------------------------------
        # EKF PREDICTION
        # ----------------------------------------------------

        ekf.predict(
            dt_value,
            process_noise
        )

        # ----------------------------------------------------
        # VELOCITY UPDATE
        # ----------------------------------------------------

        measurement_noise = (
            BASE_VELOCITY_NOISE
            /
            confidence
        )

        ekf.update_velocity(
            vn,
            ve,
            measurement_noise
        )

        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------

        nav_north = ekf.x[0]
        nav_east = ekf.x[1]

        gps_north = row[
            "gps_north_m"
        ]

        gps_east = row[
            "gps_east_m"
        ]

        error = np.sqrt(
            (
                nav_north
                - gps_north
            ) ** 2
            +
            (
                nav_east
                - gps_east
            ) ** 2
        )

        rows.append({

            "time_s": row["time_s"],

            "north_m": nav_north,
            "east_m": nav_east,

            "gps_north_m": gps_north,
            "gps_east_m": gps_east,

            "raw_ai_speed_kmh":
                raw_ai_speed,

            "corrected_speed_kmh":
                corrected_speed,

            "raw_heading_deg":
                raw_heading,

            "corrected_heading_deg":
                corrected_heading,

            "vn_mps": vn,
            "ve_mps": ve,

            "confidence":
                confidence,

            "speed_scale":
                speed_scale,

            "speed_bias":
                speed_bias,

            "heading_offset_deg":
                heading_offset,

            "position_error_m":
                error

        })

        previous_speed = corrected_speed

        previous_heading = corrected_heading

    result = pd.DataFrame(rows)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    final_error = result[
        "position_error_m"
    ].iloc[-1]

    mean_error = result[
        "position_error_m"
    ].mean()

    median_error = result[
        "position_error_m"
    ].median()

    max_error = result[
        "position_error_m"
    ].max()

    gps_distance = np.sqrt(
        (
            result["gps_north_m"].iloc[-1]
            -
            result["gps_north_m"].iloc[0]
        ) ** 2
        +
        (
            result["gps_east_m"].iloc[-1]
            -
            result["gps_east_m"].iloc[0]
        ) ** 2
    )

    if gps_distance > 1:

        drift_percent = (
            final_error
            / gps_distance
            * 100.0
        )

    else:

        drift_percent = np.nan

    return {
        "trajectory": result,

        "calibration": calibration,

        "final_error_m": final_error,

        "mean_error_m": mean_error,

        "median_error_m": median_error,

        "max_error_m": max_error,

        "gps_distance_m": gps_distance,

        "drift_percent": drift_percent
    }


# ============================================================
# RUN ALL V7 WINDOWS
# ============================================================

all_results = []

trajectories = {}

print("\n")
print("=" * 70)
print("RUNNING V8 ON V7 BENCHMARK WINDOWS")
print("=" * 70)

for window_number, row in v7.iterrows():

    start_time = float(
        row["start_time_s"]
        if "start_time_s" in v7.columns
        else row["start_s"]
    )

    end_time = float(
        row["end_time_s"]
        if "end_time_s" in v7.columns
        else row["end_s"]
    )

    duration = end_time - start_time

    print(
        f"\nWindow {window_number + 1}: "
        f"{start_time:.2f}s -> "
        f"{end_time:.2f}s "
        f"({duration:.0f}s)"
    )

    result = process_outage(
        df,
        start_time,
        end_time
    )

    if result is None:

        print("  Skipped")

        continue

    c = result["calibration"]

    print(
        f"  Speed scale    : "
        f"{c['speed_scale']:.4f}"
    )

    print(
        f"  Speed bias     : "
        f"{c['speed_bias']:.3f} km/h"
    )

    print(
        f"  Heading offset  : "
        f"{c['heading_offset']:.2f}°"
    )

    print(
        f"  Final error     : "
        f"{result['final_error_m']:.2f} m"
    )

    print(
        f"  Mean error      : "
        f"{result['mean_error_m']:.2f} m"
    )

    print(
        f"  Max error       : "
        f"{result['max_error_m']:.2f} m"
    )

    print(
        f"  Drift           : "
        f"{result['drift_percent']:.2f}%"
    )

    key = (
        f"{start_time:.2f}_{duration:.0f}"
    )

    trajectories[key] = result[
        "trajectory"
    ]

    all_results.append({

        "window":
            window_number + 1,

        "start_time_s":
            start_time,

        "end_time_s":
            end_time,

        "duration_s":
            duration,

        "final_error_m":
            result["final_error_m"],

        "mean_error_m":
            result["mean_error_m"],

        "median_error_m":
            result["median_error_m"],

        "max_error_m":
            result["max_error_m"],

        "gps_distance_m":
            result["gps_distance_m"],

        "drift_percent":
            result["drift_percent"],

        "speed_scale":
            c["speed_scale"],

        "speed_bias_kmh":
            c["speed_bias"],

        "heading_offset_deg":
            c["heading_offset"],

        "speed_std_kmh":
            c["speed_std"],

        "heading_std_deg":
            c["heading_std"],

        "calibration_samples":
            c["samples"]
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

os.makedirs(
    "data/processed",
    exist_ok=True
)

os.makedirs(
    "outputs",
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for duration, group in results_df.groupby(
    results_df["duration_s"].round()
):

    summary_rows.append({

        "duration_s":
            duration,

        "windows":
            len(group),

        "mean_final_error_m":
            group[
                "final_error_m"
            ].mean(),

        "median_final_error_m":
            group[
                "final_error_m"
            ].median(),

        "max_final_error_m":
            group[
                "final_error_m"
            ].max(),

        "mean_position_error_m":
            group[
                "mean_error_m"
            ].mean(),

        "mean_drift_percent":
            group[
                "drift_percent"
            ].mean(),

        "median_drift_percent":
            group[
                "drift_percent"
            ].median(),

        "mean_heading_offset_deg":
            group[
                "heading_offset_deg"
            ].mean(),

        "mean_speed_bias_kmh":
            group[
                "speed_bias_kmh"
            ].mean()
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("V8 OVERALL RESULTS")
print("=" * 70)

if len(results_df) > 0:

    print(
        f"\nValid windows: "
        f"{len(results_df)}"
    )

    print(
        f"Mean final error: "
        f"{results_df['final_error_m'].mean():.2f} m"
    )

    print(
        f"Median final error: "
        f"{results_df['final_error_m'].median():.2f} m"
    )

    print(
        f"Maximum final error: "
        f"{results_df['final_error_m'].max():.2f} m"
    )

    print(
        f"Mean position error: "
        f"{results_df['mean_error_m'].mean():.2f} m"
    )

    print(
        f"Mean drift: "
        f"{results_df['drift_percent'].mean():.2f}%"
    )


# ============================================================
# PLOT 1: ERROR VS OUTAGE DURATION
# ============================================================

plt.figure(figsize=(10, 6))

for duration, group in results_df.groupby(
    results_df["duration_s"].round()
):

    plt.scatter(
        np.full(
            len(group),
            duration
        ),
        group["final_error_m"],
        label=f"{int(duration)} s"
    )

plt.xlabel(
    "GNSS outage duration (seconds)"
)

plt.ylabel(
    "Final position error (m)"
)

plt.title(
    "V8 Adaptive Fusion: GNSS Outage Error"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2: REPRESENTATIVE TRAJECTORY
# ============================================================

if len(trajectories) > 0:

    first_key = list(
        trajectories.keys()
    )[0]

    traj = trajectories[
        first_key
    ]

    plt.figure(figsize=(10, 7))

    plt.plot(
        traj["gps_east_m"],
        traj["gps_north_m"],
        label="GNSS reference",
        linewidth=2
    )

    plt.plot(
        traj["east_m"],
        traj["north_m"],
        label="V8 navigation",
        linewidth=2
    )

    plt.scatter(
        traj["east_m"].iloc[0],
        traj["north_m"].iloc[0],
        s=80,
        label="Outage start"
    )

    plt.scatter(
        traj["east_m"].iloc[-1],
        traj["north_m"].iloc[-1],
        s=80,
        label="Outage end"
    )

    plt.xlabel(
        "East (m)"
    )

    plt.ylabel(
        "North (m)"
    )

    plt.title(
        "V8 GNSS-Denied Navigation"
    )

    plt.axis("equal")

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_RECOVERY,
        dpi=200
    )

    plt.close()


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 70)
print("V8 COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_RESULTS
)

print(
    OUTPUT_SUMMARY
)

print(
    OUTPUT_PLOT
)

print(
    OUTPUT_RECOVERY
)