import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SIH 2026
# PROTOTYPE V6
#
# AI SPEED + IMU/MAGNETOMETER HEADING
#                    ↓
#             DEAD RECKONING
#                    ↓
#          SIMULATED GNSS OUTAGE
#                    ↓
#          POSITION ERROR ANALYSIS
#
# Heading source:
#   Prototype V5
#
# Speed source:
#   Prototype V1
#
# GNSS:
#   Reference/evaluation only during outage
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

OUTAGE_START = 7420.42

OUTAGE_DURATION = 60.0

OUTAGE_END = (
    OUTAGE_START
    + OUTAGE_DURATION
)

EARTH_RADIUS = 6371000.0


# ============================================================
# LOAD RAW DATA
# ============================================================

print("\n" + "=" * 75)
print("SIH PROTOTYPE V6 - AI DEAD RECKONING")
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
# CLEAN COLUMNS
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


time = (
    df["time_seconds"]
    .values
)


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
    np.isfinite(ai_time)
    &
    np.isfinite(ai_speed)
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
    heading_df["time_seconds"],
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
# INTERPOLATE HEADING
#
# Important:
# Heading is circular, so interpolate sin/cos.
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
# FIND OUTAGE
# ============================================================

start_idx = np.searchsorted(
    time,
    OUTAGE_START
)

end_idx = np.searchsorted(
    time,
    OUTAGE_END
)


print(
    "\nGNSS outage:"
)

print(
    "Start index:",
    start_idx
)

print(
    "End index:",
    end_idx
)


# ============================================================
# DEAD RECKONING ARRAYS
# ============================================================

dr_north = np.full(
    len(df),
    np.nan
)

dr_east = np.full(
    len(df),
    np.nan
)


# ============================================================
# INITIALIZE FROM GNSS
# ============================================================

dr_north[start_idx] = (
    gps_north[start_idx]
)

dr_east[start_idx] = (
    gps_east[start_idx]
)


# ============================================================
# DEAD RECKONING
#
# AI speed
#     +
# V5 heading
#     ↓
# velocity
#     ↓
# integrate
# ============================================================

for i in range(
    start_idx + 1,
    end_idx + 1
):

    dt = (
        time[i]
        -
        time[i - 1]
    )

    # Handle timestamp gaps
    if (
        dt <= 0
        or dt > 0.5
    ):

        dt = 0.1


    # AI speed → m/s
    speed_ms = (
        df[
            "ai_speed_kmh"
        ].iloc[i]
        / 3.6
    )


    heading_rad = np.radians(
        heading[i]
    )


    # --------------------------------------------------------
    # North/East velocity
    # --------------------------------------------------------

    velocity_north = (
        speed_ms
        *
        np.cos(
            heading_rad
        )
    )


    velocity_east = (
        speed_ms
        *
        np.sin(
            heading_rad
        )
    )


    # --------------------------------------------------------
    # Position propagation
    # --------------------------------------------------------

    dr_north[i] = (
        dr_north[i - 1]
        +
        velocity_north
        * dt
    )


    dr_east[i] = (
        dr_east[i - 1]
        +
        velocity_east
        * dt
    )


# ============================================================
# POSITION ERROR
# ============================================================

outage_mask = (
    (time >= OUTAGE_START)
    &
    (time <= OUTAGE_END)
)


position_error = np.sqrt(
    (
        dr_north[outage_mask]
        -
        gps_north[outage_mask]
    ) ** 2
    +
    (
        dr_east[outage_mask]
        -
        gps_east[outage_mask]
    ) ** 2
)


position_error = (
    position_error[
        np.isfinite(
            position_error
        )
    ]
)


# ============================================================
# METRICS
# ============================================================

mean_error = np.mean(
    position_error
)

median_error = np.median(
    position_error
)

final_error = position_error[-1]

max_error = np.max(
    position_error
)


# ============================================================
# GPS DISTANCE
# ============================================================

gps_n = gps_north[
    start_idx:
    end_idx + 1
]

gps_e = gps_east[
    start_idx:
    end_idx + 1
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


# ============================================================
# DRIFT %
# ============================================================

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


# ============================================================
# COMPARE AGAINST V2
# ============================================================

V2_FINAL_ERROR = 426.01

if V2_FINAL_ERROR > 0:

    improvement_vs_v2 = (
        1.0
        -
        final_error
        /
        V2_FINAL_ERROR
    ) * 100.0

else:

    improvement_vs_v2 = np.nan


# ============================================================
# RESULTS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "V6 DEAD RECKONING RESULTS"
)

print(
    "=" * 75
)

print(
    f"Outage duration:"
    f" {OUTAGE_DURATION:.1f} s"
)

print(
    f"GNSS distance:"
    f" {gps_distance:.2f} m"
)

print(
    f"\nMean position error:"
    f" {mean_error:.2f} m"
)

print(
    f"Median position error:"
    f" {median_error:.2f} m"
)

print(
    f"Final position error:"
    f" {final_error:.2f} m"
)

print(
    f"Maximum position error:"
    f" {max_error:.2f} m"
)

print(
    f"Relative drift:"
    f" {drift_percent:.2f}%"
)

print(
    f"\nImprovement vs V2:"
    f" {improvement_vs_v2:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

result = pd.DataFrame({

    "timestamp":
        df[
            "timestamp"
        ].iloc[
            start_idx:
            end_idx + 1
        ].values,

    "time_seconds":
        time[
            start_idx:
            end_idx + 1
        ],

    "gps_north_m":
        gps_north[
            start_idx:
            end_idx + 1
        ],

    "gps_east_m":
        gps_east[
            start_idx:
            end_idx + 1
        ],

    "dr_north_m":
        dr_north[
            start_idx:
            end_idx + 1
        ],

    "dr_east_m":
        dr_east[
            start_idx:
            end_idx + 1
        ],

    "gps_speed_kmh":
        gps_speed[
            start_idx:
            end_idx + 1
        ],

    "ai_speed_kmh":
        df[
            "ai_speed_kmh"
        ].iloc[
            start_idx:
            end_idx + 1
        ].values,

    "heading_deg":
        heading[
            start_idx:
            end_idx + 1
        ],

    "position_error_m":
        np.sqrt(
            (
                dr_north[
                    start_idx:
                    end_idx + 1
                ]
                -
                gps_north[
                    start_idx:
                    end_idx + 1
                ]
            ) ** 2
            +
            (
                dr_east[
                    start_idx:
                    end_idx + 1
                ]
                -
                gps_east[
                    start_idx:
                    end_idx + 1
                ]
            ) ** 2
        )

})


result_file = os.path.join(
    PROCESSED_DIR,
    "prototype_v6_ai_dr_results.csv"
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
    gps_east[
        start_idx:
        end_idx + 1
    ],
    gps_north[
        start_idx:
        end_idx + 1
    ],
    label="GNSS Reference",
    linewidth=2
)

axes[0].plot(
    dr_east[
        start_idx:
        end_idx + 1
    ],
    dr_north[
        start_idx:
        end_idx + 1
    ],
    label="AI + IMU Dead Reckoning",
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
    "60-Second GNSS Outage: AI Dead Reckoning"
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
    position_error,
    linewidth=2
)

axes[1].set_title(
    f"Position Error "
    f"(Final = {final_error:.2f} m)"
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "Error (m)"
)

axes[1].grid(True)


# ============================================================
# SPEED
# ============================================================

axes[2].plot(
    outage_time,
    gps_speed[
        outage_mask
    ],
    label="GNSS Speed",
    linewidth=2
)

axes[2].plot(
    outage_time,
    df[
        "ai_speed_kmh"
    ].values[
        outage_mask
    ],
    label="AI Speed",
    linewidth=2
)

axes[2].set_title(
    "AI Speed During GNSS Outage"
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
    "prototype_v6_ai_dead_reckoning.png"
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

print(
    "\nFiles created:"
)

print(
    result_file
)

print(
    plot_file
)

print(
    "\nV6 completed successfully."
)