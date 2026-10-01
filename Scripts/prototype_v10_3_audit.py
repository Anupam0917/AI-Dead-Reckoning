import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

V93_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v9_dead_reckoning_results.csv"
)

TARGET_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "prototype_v9_velocity_targets.csv"
)

CANONICAL_FILE = os.path.join(
    BASE,
    "data",
    "processed",
    "canonical_gnss_reference.csv"
)

print("=" * 70)
print("V10.3 BASELINE AUDIT")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load files
# ---------------------------------------------------------

print("\n[1] Loading files...")

v93 = pd.read_csv(V93_FILE)
target = pd.read_csv(TARGET_FILE)
canonical = pd.read_csv(CANONICAL_FILE)

print(f"V9.3 results : {v93.shape}")
print(f"Velocity target : {target.shape}")
print(f"Canonical GNSS : {canonical.shape}")

# ---------------------------------------------------------
# 2. Show V9.3 structure
# ---------------------------------------------------------

print("\n[2] V9.3 result columns:")
for col in v93.columns:
    print("  ", col)

print("\nV9.3 results:")
print(v93.to_string(index=False))

# ---------------------------------------------------------
# 3. Examine the target trajectory
# ---------------------------------------------------------

print("\n[3] Target velocity columns:")

required_target = [
    "time_s",
    "vn_mps",
    "ve_mps",
    "velocity_mps",
    "velocity_kmh"
]

for col in required_target:
    if col in target.columns:
        print(f"  OK: {col}")
    else:
        print(f"  MISSING: {col}")

# ---------------------------------------------------------
# 4. Select the exact outage
# ---------------------------------------------------------

OUTAGE_START = 8500.0
OUTAGE_END = 8560.0

t = target["time_s"].to_numpy()

mask = (
    (t >= OUTAGE_START) &
    (t <= OUTAGE_END)
)

out = target.loc[mask].copy()

print("\n[4] Target outage window")
print(f"Start requested : {OUTAGE_START}s")
print(f"End requested   : {OUTAGE_END}s")
print(f"Samples         : {len(out)}")

if len(out) < 2:
    raise RuntimeError("Not enough target samples in outage window.")

# ---------------------------------------------------------
# 5. Integrate target velocity
# ---------------------------------------------------------

time = out["time_s"].to_numpy()
vn = out["vn_mps"].to_numpy()
ve = out["ve_mps"].to_numpy()

dt = np.diff(time)

north = np.zeros(len(out))
east = np.zeros(len(out))

for i in range(1, len(out)):
    north[i] = north[i - 1] + 0.5 * (vn[i - 1] + vn[i]) * dt[i - 1]
    east[i] = east[i - 1] + 0.5 * (ve[i - 1] + ve[i]) * dt[i - 1]

distance = np.sqrt(
    np.diff(north) ** 2 +
    np.diff(east) ** 2
).sum()

print("\n[5] Target-derived reference trajectory")
print(f"North displacement : {north[-1]:.3f} m")
print(f"East displacement  : {east[-1]:.3f} m")
print(f"Path distance      : {distance:.3f} m")
print(f"Mean speed         : {np.mean(np.sqrt(vn**2 + ve**2)):.3f} m/s")

# ---------------------------------------------------------
# 6. Compare with canonical GNSS
# ---------------------------------------------------------

print("\n[6] Canonical GNSS reference")

ct = canonical["time_s"].to_numpy()

cmask = (
    (ct >= OUTAGE_START) &
    (ct <= OUTAGE_END)
)

cout = canonical.loc[cmask].copy()

print(f"Canonical samples : {len(cout)}")

if len(cout) >= 2:

    canonical_distance = cout["step_distance_m"].sum()

    canonical_north = (
        cout["north_m"].iloc[-1] -
        cout["north_m"].iloc[0]
    )

    canonical_east = (
        cout["east_m"].iloc[-1] -
        cout["east_m"].iloc[0]
    )

    print(f"North displacement : {canonical_north:.3f} m")
    print(f"East displacement  : {canonical_east:.3f} m")
    print(f"Path distance      : {canonical_distance:.3f} m")

else:
    canonical_distance = np.nan

# ---------------------------------------------------------
# 7. Compare reference definitions
# ---------------------------------------------------------

print("\n[7] REFERENCE COMPARISON")

if not np.isnan(canonical_distance):

    ratio = canonical_distance / max(distance, 1e-9)

    print(f"Target reference distance    : {distance:.3f} m")
    print(f"Canonical GNSS distance     : {canonical_distance:.3f} m")
    print(f"Canonical / target ratio    : {ratio:.3f}x")

    if ratio > 2:
        print("\nWARNING:")
        print("The two evaluation references are substantially different.")
        print("This means V9.3 and later versions may NOT be directly comparable.")

# ---------------------------------------------------------
# 8. Check V9.3's reported reference distance
# ---------------------------------------------------------

print("\n[8] V9.3 reported benchmark")

v93_match = v93[
    (v93["start_s"] == OUTAGE_START) &
    (v93["duration_s"] == 60)
]

if len(v93_match) == 0:

    # Floating point tolerant match
    v93_match = v93[
        np.isclose(v93["start_s"], OUTAGE_START) &
        np.isclose(v93["duration_s"], 60)
    ]

if len(v93_match):

    row = v93_match.iloc[0]

    print(f"V9.3 final error     : {row['final_error_m']:.3f} m")
    print(f"V9.3 mean error      : {row['mean_error_m']:.3f} m")
    print(f"V9.3 max error       : {row['max_error_m']:.3f} m")
    print(f"V9.3 reference dist  : {row['true_distance_m']:.3f} m")
    print(f"V9.3 drift           : {row['drift_percent']:.3f}%")

    print("\nDifference from target-derived reference:")

    print(
        f"  V9.3 reported : {row['true_distance_m']:.3f} m"
    )

    print(
        f"  Reconstructed : {distance:.3f} m"
    )

    print(
        f"  Difference    : "
        f"{abs(row['true_distance_m'] - distance):.3f} m"
    )

else:
    print("Could not find V9.3 8500s / 60s result.")

# ---------------------------------------------------------
# 9. Check timestep quality
# ---------------------------------------------------------

print("\n[9] Target timestep")

print(f"Median dt : {np.median(dt):.6f}s")
print(f"Mean dt   : {np.mean(dt):.6f}s")
print(f"Min dt    : {np.min(dt):.6f}s")
print(f"Max dt    : {np.max(dt):.6f}s")

# ---------------------------------------------------------
# 10. Final conclusion
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("V10.3 AUDIT COMPLETE")
print("=" * 70)

print("""
Next decision:

A) If V9.3 reference matches the reconstructed target trajectory:
   -> V9.3 benchmark is internally consistent.
   -> We should rebuild later versions using the SAME reference.

B) If V9.3 reference does not match:
   -> We found a benchmark/implementation discrepancy.
   -> Fix the evaluation before further navigation development.
""")