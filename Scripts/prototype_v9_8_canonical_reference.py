from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    ROOT
    / "data"
    / "raw"
    / "Synchronised V abd S datasets"
    / "Categorised IOVNB Dataset"
    / "M (Driver B)"
    / "S-M.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "canonical_gnss_reference.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EARTH_RADIUS = 6371000.0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("V9.8-A CANONICAL GNSS REFERENCE")
print("=" * 80)

print("\nLoading:")
print(RAW_FILE)

df = pd.read_csv(
    RAW_FILE,
    encoding="cp1252",
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print("\nRaw rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# FIND REQUIRED COLUMNS
# ============================================================

LAT = "GPS LATITUDE (degrees)"
LON = "GPS LONGITUDE (degrees)"
SPEED = "GPS SPEED (Kmh)"
TIME = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"


required = [
    LAT,
    LON,
    SPEED,
    TIME,
]

for column in required:

    if column not in df.columns:

        raise ValueError(
            f"Required column not found: {column}"
        )


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df[TIME],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce",
)

df = df[
    df["timestamp"].notna()
].copy()

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# NUMERIC GNSS DATA
# ============================================================

df["latitude"] = pd.to_numeric(
    df[LAT],
    errors="coerce",
)

df["longitude"] = pd.to_numeric(
    df[LON],
    errors="coerce",
)

df["gps_speed_kmh"] = pd.to_numeric(
    df[SPEED],
    errors="coerce",
)


# Remove rows without coordinates

df = df[
    df["latitude"].notna()
    &
    df["longitude"].notna()
].copy()

df = df.reset_index(drop=True)


# ============================================================
# GLOBAL TIME
# ============================================================

df["time_s"] = (
    df["timestamp"]
    - df["timestamp"].iloc[0]
).dt.total_seconds()


# ============================================================
# LOCAL NORTH/EAST COORDINATES
# ============================================================

lat0 = df["latitude"].iloc[0]
lon0 = df["longitude"].iloc[0]

lat_rad = np.radians(
    df["latitude"].to_numpy()
)

lon_rad = np.radians(
    df["longitude"].to_numpy()
)

lat0_rad = np.radians(lat0)
lon0_rad = np.radians(lon0)

north_m = (
    lat_rad - lat0_rad
) * EARTH_RADIUS

east_m = (
    lon_rad - lon0_rad
) * EARTH_RADIUS * np.cos(
    lat0_rad
)

df["north_m"] = north_m
df["east_m"] = east_m


# ============================================================
# GNSS POSITION CHANGE
# ============================================================

df["dn_m"] = (
    df["north_m"]
    .diff()
)

df["de_m"] = (
    df["east_m"]
    .diff()
)

df["step_distance_m"] = np.sqrt(
    df["dn_m"] ** 2
    +
    df["de_m"] ** 2
)

df["step_distance_m"] = (
    df["step_distance_m"]
    .fillna(0.0)
)


# ============================================================
# SAVE
# ============================================================

output = df[
    [
        "timestamp",
        "time_s",
        "latitude",
        "longitude",
        "gps_speed_kmh",
        "north_m",
        "east_m",
        "dn_m",
        "de_m",
        "step_distance_m",
    ]
].copy()


output.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 80)
print("CANONICAL REFERENCE CREATED")
print("=" * 80)

print("\nRows:", len(output))

print(
    "Time:",
    output["time_s"].min(),
    "->",
    output["time_s"].max(),
)

print(
    "\nLatitude:",
    output["latitude"].min(),
    "->",
    output["latitude"].max(),
)

print(
    "Longitude:",
    output["longitude"].min(),
    "->",
    output["longitude"].max(),
)

print(
    "\nTotal coordinate-path distance:",
    f"{output['step_distance_m'].sum():.3f} m",
)

print(
    "Maximum single position step:",
    f"{output['step_distance_m'].max():.3f} m",
)

print(
    "\nOrigin:"
)

print(
    f"Latitude : {lat0:.8f}"
)

print(
    f"Longitude: {lon0:.8f}"
)

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    "\nV9.8-A COMPLETE"
)