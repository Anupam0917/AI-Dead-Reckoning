import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SIH 2026
# AI/ML Based Intelligent Dead Reckoning
#
# PROTOTYPE V3
# Extended Kalman Filter Fusion
#
# State:
#   [North Position, East Position,
#    North Velocity, East Velocity]
#
# Measurements:
#   1. GNSS position
#   2. AI speed + heading
#
# During GNSS outage:
#   GNSS measurement update is disabled.
#
# When GNSS returns:
#   EKF automatically corrects accumulated drift.
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

OUTAGE_START = 7420.42
OUTAGE_DURATION = 60.0
OUTAGE_END = OUTAGE_START + OUTAGE_DURATION

EARTH_RADIUS = 6371000.0


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("SIH PROTOTYPE V3 - EKF FUSION")
print("=" * 75)

print("\nLoading dataset...")

df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252"
)

print(
    f"Raw rows: {len(df)}"
)


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


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(keyword):

    matches = [
        c for c in df.columns
        if keyword.lower() in c.lower()
    ]

    if not matches:

        raise ValueError(
            f"Column not found: {keyword}"
        )

    return matches[0]


LAT_COL = find_column(
    "GPS_LATITUDE"
)

LON_COL = find_column(
    "GPS_LONGITUDE"
)

GPS_SPEED_COL = find_column(
    "GPS_SPEED"
)

GPS_HEADING_COL = find_column(
    "GPS_ORIENTATION"
)

PHONE_HEADING_COL = find_column(
    "ORIENTATION_Yaw"
)

TIME_COL = find_column(
    "DATE"
)


print("\nDetected columns:")

print(
    "Latitude       :", LAT_COL
)

print(
    "Longitude      :", LON_COL
)

print(
    "GPS speed      :", GPS_SPEED_COL
)

print(
    "GPS heading    :", GPS_HEADING_COL
)

print(
    "Phone heading  :", PHONE_HEADING_COL
)

print(
    "Timestamp      :", TIME_COL
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
    subset=["timestamp"]
).copy()

df["time_seconds"] = (
    df["timestamp"]
    - df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# NUMERIC DATA
# ============================================================

for col in [
    LAT_COL,
    LON_COL,
    GPS_SPEED_COL,
    GPS_HEADING_COL,
    PHONE_HEADING_COL
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# LOAD AI SPEED
# ============================================================

print("\nLoading AI speed predictions...")

ai_df = pd.read_csv(
    AI_FILE
)

print(
    "AI result columns:"
)

print(
    ai_df.columns.tolist()
)


AI_SPEED_COL = "ai_speed_kmh"

if AI_SPEED_COL not in ai_df.columns:

    raise ValueError(
        "ai_speed_kmh not found in AI results."
    )


# ============================================================
# ALIGN AI SPEED
# ============================================================

if "time_s" in ai_df.columns:

    ai_time = pd.to_numeric(
        ai_df["time_s"],
        errors="coerce"
    ).values

else:

    ai_time = np.linspace(
        df["time_seconds"].iloc[0],
        df["time_seconds"].iloc[-1],
        len(ai_df)
    )


ai_speed_values = pd.to_numeric(
    ai_df[AI_SPEED_COL],
    errors="coerce"
).values


valid_ai = (
    np.isfinite(ai_time)
    &
    np.isfinite(ai_speed_values)
)


ai_time = ai_time[valid_ai]

ai_speed_values = ai_speed_values[
    valid_ai
]


df["ai_speed_kmh"] = np.interp(
    df["time_seconds"].values,
    ai_time,
    ai_speed_values
)


# ============================================================
# HEADING
# ============================================================

phone_heading = pd.to_numeric(
    df[PHONE_HEADING_COL],
    errors="coerce"
).values

gps_heading = pd.to_numeric(
    df[GPS_HEADING_COL],
    errors="coerce"
).values


phone_heading = (
    pd.Series(phone_heading)
    .interpolate(
        limit_direction="both"
    )
    .values
)

gps_heading = (
    pd.Series(gps_heading)
    .interpolate(
        limit_direction="both"
    )
    .values
)


# ============================================================
# HEADING CALIBRATION
#
# Use GNSS heading BEFORE outage only.
# GNSS heading is never used during the outage.
# ============================================================

pre_outage = (
    (df["time_seconds"].values >=
     OUTAGE_START - 60)
    &
    (df["time_seconds"].values <
     OUTAGE_START)
    &
    (df[GPS_SPEED_COL].values > 2.0)
    &
    np.isfinite(phone_heading)
    &
    np.isfinite(gps_heading)
)


def angle_difference(a, b):

    return (
        (a - b + 180.0)
        % 360.0
    ) - 180.0


heading_diff = angle_difference(
    gps_heading[pre_outage],
    phone_heading[pre_outage]
)

heading_diff = heading_diff[
    np.isfinite(heading_diff)
]


if len(heading_diff) > 0:

    heading_offset = np.median(
        heading_diff
    )

else:

    heading_offset = 0.0


heading = (
    phone_heading
    + heading_offset
) % 360.0


print(
    f"\nHeading calibration offset: "
    f"{heading_offset:.2f} degrees"
)


# ============================================================
# GPS LAT/LON → LOCAL NORTH/EAST
# ============================================================

lat = df[LAT_COL].values
lon = df[LON_COL].values

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
    * np.cos(lat0_rad)
)


# ============================================================
# OUTAGE INDICES
# ============================================================

time = df["time_seconds"].values

start_idx = np.searchsorted(
    time,
    OUTAGE_START
)

end_idx = np.searchsorted(
    time,
    OUTAGE_END
)


print(
    f"\nGNSS outage:"
)

print(
    f"Start index: {start_idx}"
)

print(
    f"End index  : {end_idx}"
)


# ============================================================
# EKF STATE
#
# x =
# [North position,
#  East position,
#  North velocity,
#  East velocity]
# ============================================================

state = np.array([
    gps_north[start_idx],
    gps_east[start_idx],
    0.0,
    0.0
])


# ============================================================
# INITIAL COVARIANCE
# ============================================================

P = np.diag([
    5.0 ** 2,
    5.0 ** 2,
    2.0 ** 2,
    2.0 ** 2
])


# ============================================================
# MEASUREMENT MATRICES
#
# GNSS position measurement
# z = [north, east]
# ============================================================

H_GPS = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
])


# GNSS measurement covariance
R_GPS = np.diag([
    5.0 ** 2,
    5.0 ** 2
])


# ============================================================
# STORAGE
# ============================================================

fused_north = np.full(
    len(df),
    np.nan
)

fused_east = np.full(
    len(df),
    np.nan
)

fused_speed = np.full(
    len(df),
    np.nan
)

position_uncertainty = np.full(
    len(df),
    np.nan
)

gnss_used = np.zeros(
    len(df),
    dtype=bool
)


# ============================================================
# SAVE INITIAL STATE
# ============================================================

fused_north[start_idx] = state[0]

fused_east[start_idx] = state[1]

fused_speed[start_idx] = 0.0

position_uncertainty[start_idx] = np.sqrt(
    P[0, 0] + P[1, 1]
)


# ============================================================
# EKF LOOP
# ============================================================

print("\nRunning EKF...")

for i in range(
    start_idx + 1,
    len(df)
):

    dt = (
        time[i]
        - time[i - 1]
    )


    # --------------------------------------------------------
    # Protect against timestamp gaps
    # --------------------------------------------------------

    if (
        dt <= 0
        or dt > 0.5
    ):

        dt = 0.1


    # --------------------------------------------------------
    # AI SPEED + HEADING
    # --------------------------------------------------------

    speed_ms = (
        df["ai_speed_kmh"].iloc[i]
        / 3.6
    )

    heading_rad = np.radians(
        heading[i]
    )


    ai_v_north = (
        speed_ms
        * np.cos(heading_rad)
    )

    ai_v_east = (
        speed_ms
        * np.sin(heading_rad)
    )


    # --------------------------------------------------------
    # PREDICTION
    #
    # Constant velocity model
    # --------------------------------------------------------

    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


    # Process noise
    acceleration_noise = 1.5

    G = np.array([
        [0.5 * dt ** 2, 0.0],
        [0.0, 0.5 * dt ** 2],
        [dt, 0.0],
        [0.0, dt]
    ])


    Q = (
        G
        @ np.diag([
            acceleration_noise ** 2,
            acceleration_noise ** 2
        ])
        @ G.T
    )


    # State prediction
    state = F @ state


    # Covariance prediction
    P = (
        F
        @ P
        @ F.T
        + Q
    )


    # --------------------------------------------------------
    # AI VELOCITY MEASUREMENT
    #
    # z = [north velocity, east velocity]
    # --------------------------------------------------------

    z_velocity = np.array([
        ai_v_north,
        ai_v_east
    ])


    H_velocity = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


    # AI velocity uncertainty
    velocity_noise = 1.5

    R_velocity = np.diag([
        velocity_noise ** 2,
        velocity_noise ** 2
    ])


    innovation = (
        z_velocity
        - H_velocity @ state
    )


    S = (
        H_velocity
        @ P
        @ H_velocity.T
        + R_velocity
    )


    K = (
        P
        @ H_velocity.T
        @ np.linalg.inv(S)
    )


    state = (
        state
        + K @ innovation
    )


    P = (
        np.eye(4)
        - K @ H_velocity
    ) @ P


    # --------------------------------------------------------
    # GNSS UPDATE
    #
    # DISABLED DURING OUTAGE
    # --------------------------------------------------------

    gnss_available = not (
        OUTAGE_START
        <= time[i]
        <= OUTAGE_END
    )


    if gnss_available:

        z_gps = np.array([
            gps_north[i],
            gps_east[i]
        ])


        innovation = (
            z_gps
            - H_GPS @ state
        )


        S = (
            H_GPS
            @ P
            @ H_GPS.T
            + R_GPS
        )


        K = (
            P
            @ H_GPS.T
            @ np.linalg.inv(S)
        )


        state = (
            state
            + K @ innovation
        )


        P = (
            np.eye(4)
            - K @ H_GPS
        ) @ P


        gnss_used[i] = True


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    fused_north[i] = state[0]

    fused_east[i] = state[1]


    fused_speed[i] = (
        np.sqrt(
            state[2] ** 2
            +
            state[3] ** 2
        )
        * 3.6
    )


    position_uncertainty[i] = np.sqrt(
        P[0, 0]
        +
        P[1, 1]
    )


# ============================================================
# EVALUATE OUTAGE
# ============================================================

outage_mask = (
    (time >= OUTAGE_START)
    &
    (time <= OUTAGE_END)
)


raw_dr_north = np.full(
    len(df),
    np.nan
)

raw_dr_east = np.full(
    len(df),
    np.nan
)


# ------------------------------------------------------------
# Generate raw DR trajectory using same AI measurements
# ------------------------------------------------------------

raw_dr_north[start_idx] = gps_north[start_idx]

raw_dr_east[start_idx] = gps_east[start_idx]


for i in range(
    start_idx + 1,
    end_idx + 1
):

    dt = (
        time[i]
        - time[i - 1]
    )

    if (
        dt <= 0
        or dt > 0.5
    ):

        dt = 0.1


    speed_ms = (
        df["ai_speed_kmh"].iloc[i]
        / 3.6
    )

    heading_rad = np.radians(
        heading[i]
    )


    raw_dr_north[i] = (
        raw_dr_north[i - 1]
        +
        speed_ms
        * np.cos(heading_rad)
        * dt
    )


    raw_dr_east[i] = (
        raw_dr_east[i - 1]
        +
        speed_ms
        * np.sin(heading_rad)
        * dt
    )


# ============================================================
# ERROR METRICS
# ============================================================

raw_error = np.sqrt(
    (
        raw_dr_north[outage_mask]
        -
        gps_north[outage_mask]
    ) ** 2
    +
    (
        raw_dr_east[outage_mask]
        -
        gps_east[outage_mask]
    ) ** 2
)


ekf_error = np.sqrt(
    (
        fused_north[outage_mask]
        -
        gps_north[outage_mask]
    ) ** 2
    +
    (
        fused_east[outage_mask]
        -
        gps_east[outage_mask]
    ) ** 2
)


raw_error = raw_error[
    np.isfinite(raw_error)
]

ekf_error = ekf_error[
    np.isfinite(ekf_error)
]


raw_final = raw_error[-1]

ekf_final = ekf_error[-1]

raw_mean = np.mean(
    raw_error
)

ekf_mean = np.mean(
    ekf_error
)

raw_max = np.max(
    raw_error
)

ekf_max = np.max(
    ekf_error
)


# ============================================================
# DRIFT IMPROVEMENT
# ============================================================

if raw_final > 0:

    improvement = (
        1.0
        -
        ekf_final / raw_final
    ) * 100.0

else:

    improvement = np.nan


# ============================================================
# GNSS DISTANCE
# ============================================================

gps_n = gps_north[
    start_idx:end_idx + 1
]

gps_e = gps_east[
    start_idx:end_idx + 1
]


gps_distance = np.sum(
    np.sqrt(
        np.diff(gps_n) ** 2
        +
        np.diff(gps_e) ** 2
    )
)


print("\n" + "=" * 75)
print("V3 EKF RESULTS")
print("=" * 75)

print(
    f"GNSS outage duration : "
    f"{OUTAGE_DURATION:.1f} s"
)

print(
    f"GNSS distance        : "
    f"{gps_distance:.2f} m"
)

print("\nRAW DEAD RECKONING")

print(
    f"Mean error           : "
    f"{raw_mean:.2f} m"
)

print(
    f"Final error          : "
    f"{raw_final:.2f} m"
)

print(
    f"Maximum error        : "
    f"{raw_max:.2f} m"
)


print("\nEKF FUSION")

print(
    f"Mean error           : "
    f"{ekf_mean:.2f} m"
)

print(
    f"Final error          : "
    f"{ekf_final:.2f} m"
)

print(
    f"Maximum error        : "
    f"{ekf_max:.2f} m"
)

print(
    f"\nFinal-error change   : "
    f"{improvement:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

result = pd.DataFrame({

    "timestamp":
        df["timestamp"].iloc[
            start_idx:
        ].values,

    "time_seconds":
        time[start_idx:],

    "gps_north_m":
        gps_north[start_idx:],

    "gps_east_m":
        gps_east[start_idx:],

    "raw_dr_north_m":
        raw_dr_north[start_idx:],

    "raw_dr_east_m":
        raw_dr_east[start_idx:],

    "ekf_north_m":
        fused_north[start_idx:],

    "ekf_east_m":
        fused_east[start_idx:],

    "gps_speed_kmh":
        df[GPS_SPEED_COL].iloc[
            start_idx:
        ].values,

    "ai_speed_kmh":
        df["ai_speed_kmh"].iloc[
            start_idx:
        ].values,

    "ekf_speed_kmh":
        fused_speed[start_idx:],

    "heading_deg":
        heading[start_idx:],

    "gnss_used":
        gnss_used[start_idx:],

    "position_uncertainty_m":
        position_uncertainty[start_idx:]

})


result_file = os.path.join(
    PROCESSED_DIR,
    "prototype_v3_ekf_results.csv"
)

result.to_csv(
    result_file,
    index=False
)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(13, 15)
)


# ============================================================
# TRAJECTORY
# ============================================================

axes[0].plot(
    gps_east[outage_mask],
    gps_north[outage_mask],
    label="GNSS Reference",
    linewidth=2
)

axes[0].plot(
    raw_dr_east[outage_mask],
    raw_dr_north[outage_mask],
    label="Raw AI Dead Reckoning",
    linewidth=2
)

axes[0].plot(
    fused_east[outage_mask],
    fused_north[outage_mask],
    label="EKF Fusion",
    linewidth=2
)


axes[0].scatter(
    gps_east[start_idx],
    gps_north[start_idx],
    s=70,
    label="Outage Start"
)


axes[0].scatter(
    gps_east[end_idx],
    gps_north[end_idx],
    s=70,
    label="Outage End"
)


axes[0].set_title(
    "60-Second GNSS Outage: Trajectory Comparison"
)

axes[0].set_xlabel(
    "East displacement (m)"
)

axes[0].set_ylabel(
    "North displacement (m)"
)

axes[0].legend()

axes[0].grid(True)


# ============================================================
# POSITION ERROR
# ============================================================

outage_time = time[
    outage_mask
]


axes[1].plot(
    outage_time,
    raw_error,
    label="Raw DR Error",
    linewidth=2
)


axes[1].plot(
    outage_time,
    ekf_error,
    label="EKF Error",
    linewidth=2
)


axes[1].set_title(
    "Position Error During GNSS Outage"
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "Position error (m)"
)

axes[1].legend()

axes[1].grid(True)


# ============================================================
# SPEED
# ============================================================

axes[2].plot(
    time[outage_mask],
    df[GPS_SPEED_COL].values[
        outage_mask
    ],
    label="GNSS Speed",
    linewidth=2
)


axes[2].plot(
    time[outage_mask],
    df["ai_speed_kmh"].values[
        outage_mask
    ],
    label="AI Speed",
    linewidth=2
)


axes[2].plot(
    time[outage_mask],
    fused_speed[outage_mask],
    label="EKF Speed",
    linewidth=2
)


axes[2].set_title(
    "Speed Estimation During GNSS Outage"
)

axes[2].set_xlabel(
    "Time (s)"
)

axes[2].set_ylabel(
    "Speed (km/h)"
)

axes[2].legend()

axes[2].grid(True)


plt.tight_layout()


plot_file = os.path.join(
    OUTPUT_DIR,
    "prototype_v3_ekf_fusion.png"
)

plt.savefig(
    plot_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\nFiles created:")

print(
    result_file
)

print(
    plot_file
)

print("\nV3 EKF prototype completed successfully.")