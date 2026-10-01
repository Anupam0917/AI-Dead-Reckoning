import numpy as np
import pandas as pd


# ============================================================
# V9 GPS DIAGNOSTIC
# ============================================================

DATA_PATH = (
    "data/raw/"
    "Synchronised V abd S datasets/"
    "Categorised IOVNB Dataset/"
    "M (Driver B)/"
    "S-M.csv"
)

EARTH_RADIUS_M = 6371000.0


print("=" * 70)
print("V9 GPS COORDINATE DIAGNOSTIC")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    encoding="cp1252"
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# COLUMNS
# ============================================================

LAT_COL = "GPS LATITUDE (degrees)"
LON_COL = "GPS LONGITUDE (degrees)"
SPEED_COL = "GPS SPEED (Kmh)"
TIME_COL = "DATE (YYYY-MO-DD HH-MI-SS_SSS)"


# ============================================================
# CONVERT
# ============================================================

df["lat"] = pd.to_numeric(
    df[LAT_COL],
    errors="coerce"
)

df["lon"] = pd.to_numeric(
    df[LON_COL],
    errors="coerce"
)

df["gps_speed"] = pd.to_numeric(
    df[SPEED_COL],
    errors="coerce"
)

df["timestamp"] = pd.to_datetime(
    df[TIME_COL],
    format="%Y-%m-%d %H:%M:%S:%f",
    errors="coerce"
)


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\nLATITUDE")

print(
    "min:",
    df["lat"].min()
)

print(
    "max:",
    df["lat"].max()
)

print(
    "mean:",
    df["lat"].mean()
)

print(
    "unique:",
    df["lat"].nunique()
)


print("\nLONGITUDE")

print(
    "min:",
    df["lon"].min()
)

print(
    "max:",
    df["lon"].max()
)

print(
    "mean:",
    df["lon"].mean()
)

print(
    "unique:",
    df["lon"].nunique()
)


print("\nGPS SPEED")

print(
    "min:",
    df["gps_speed"].min()
)

print(
    "max:",
    df["gps_speed"].max()
)

print(
    "mean:",
    df["gps_speed"].mean()
)

print(
    "median:",
    df["gps_speed"].median()
)


# ============================================================
# FIRST 10 GPS VALUES
# ============================================================

print("\nFIRST 10 GPS SAMPLES")

print(
    df[
        [
            "timestamp",
            "lat",
            "lon",
            "gps_speed"
        ]
    ].head(10).to_string()
)


# ============================================================
# LAST 10 GPS VALUES
# ============================================================

print("\nLAST 10 GPS SAMPLES")

print(
    df[
        [
            "timestamp",
            "lat",
            "lon",
            "gps_speed"
        ]
    ].tail(10).to_string()
)


# ============================================================
# TIME DIFFERENCE
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

df["dt"] = (
    df["timestamp"]
    -
    df["timestamp"].iloc[0]
).dt.total_seconds()

df["delta_t"] = df["dt"].diff()


print("\nTIME INTERVAL")

print(
    "median dt:",
    df["delta_t"].median()
)

print(
    "mean dt:",
    df["delta_t"].mean()
)

print(
    "minimum dt:",
    df["delta_t"].min()
)

print(
    "maximum dt:",
    df["delta_t"].max()
)


# ============================================================
# COORDINATE DIFFERENCES
# ============================================================

df["delta_lat"] = df["lat"].diff()

df["delta_lon"] = df["lon"].diff()


print("\nCOORDINATE DIFFERENCES")

print(
    "mean |delta lat|:",
    df["delta_lat"].abs().mean()
)

print(
    "mean |delta lon|:",
    df["delta_lon"].abs().mean()
)

print(
    "max |delta lat|:",
    df["delta_lat"].abs().max()
)

print(
    "max |delta lon|:",
    df["delta_lon"].abs().max()
)


# ============================================================
# LOCAL NORTH/EAST
# ============================================================

lat0 = df["lat"].iloc[0]
lon0 = df["lon"].iloc[0]

lat0_rad = np.deg2rad(
    lat0
)


df["north_m"] = (
    np.deg2rad(
        df["lat"] - lat0
    )
    *
    EARTH_RADIUS_M
)


df["east_m"] = (
    np.deg2rad(
        df["lon"] - lon0
    )
    *
    EARTH_RADIUS_M
    *
    np.cos(lat0_rad)
)


# ============================================================
# DISPLACEMENT
# ============================================================

df["delta_north"] = (
    df["north_m"].diff()
)

df["delta_east"] = (
    df["east_m"].diff()
)


print("\nLOCAL DISPLACEMENT")

print(
    "mean |north displacement|:",
    df["delta_north"].abs().mean()
)

print(
    "mean |east displacement|:",
    df["delta_east"].abs().mean()
)

print(
    "max |north displacement|:",
    df["delta_north"].abs().max()
)

print(
    "max |east displacement|:",
    df["delta_east"].abs().max()
)


# ============================================================
# VELOCITY
# ============================================================

df["vn"] = (
    df["delta_north"]
    /
    df["delta_t"]
)

df["ve"] = (
    df["delta_east"]
    /
    df["delta_t"]
)


df["velocity"] = np.sqrt(
    df["vn"] ** 2
    +
    df["ve"] ** 2
)


df["velocity_kmh"] = (
    df["velocity"]
    *
    3.6
)


print("\nDERIVED VELOCITY")

print(
    "mean:",
    df["velocity_kmh"].mean(),
    "km/h"
)

print(
    "median:",
    df["velocity_kmh"].median(),
    "km/h"
)

print(
    "max:",
    df["velocity_kmh"].max(),
    "km/h"
)


# ============================================================
# COMPARE DERIVED SPEED WITH GPS SPEED
# ============================================================

valid = (
    df["velocity_kmh"].notna()
    &
    df["gps_speed"].notna()
    &
    np.isfinite(df["velocity_kmh"])
)


if valid.sum() > 0:

    error = (
        df.loc[
            valid,
            "velocity_kmh"
        ]
        -
        df.loc[
            valid,
            "gps_speed"
        ]
    )

    print(
        "\nDERIVED VS GPS SPEED"
    )

    print(
        "samples:",
        valid.sum()
    )

    print(
        "mean error:",
        error.mean(),
        "km/h"
    )

    print(
        "MAE:",
        np.abs(error).mean(),
        "km/h"
    )


# ============================================================
# SAMPLE VELOCITY ROWS
# ============================================================

print("\nFIRST 20 DERIVED VELOCITY ROWS")

print(
    df[
        [
            "timestamp",
            "lat",
            "lon",
            "gps_speed",
            "delta_t",
            "delta_north",
            "delta_east",
            "vn",
            "ve",
            "velocity_kmh"
        ]
    ]
    .head(20)
    .to_string()
)


print("\n")
print("=" * 70)
print("GPS DIAGNOSTIC COMPLETE")
print("=" * 70)