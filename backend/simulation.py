from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRODUCTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "production_real_sensor_predictions.csv"
)


# ============================================================
# NAVIGATION SIMULATION
# ============================================================

class NavigationSimulation:

    def __init__(self):
        print("Loading navigation simulation data...")

        if not PRODUCTION_FILE.exists():
            raise FileNotFoundError(
                f"Production prediction file not found:\n"
                f"{PRODUCTION_FILE}"
            )

        self.data = pd.read_csv(PRODUCTION_FILE)

        required_columns = [
            "timestamp",
            "time_s",
            "latitude",
            "longitude",
            "vn_ai_mps",
            "ve_ai_mps",
            "ai_speed_mps",
            "ai_speed_kmh",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns in production "
                "prediction file:\n"
                f"{missing_columns}"
            )

        self.data["time_s"] = pd.to_numeric(
            self.data["time_s"],
            errors="coerce"
        )

        self.data["latitude"] = pd.to_numeric(
            self.data["latitude"],
            errors="coerce"
        )

        self.data["longitude"] = pd.to_numeric(
            self.data["longitude"],
            errors="coerce"
        )

        self.data["vn_ai_mps"] = pd.to_numeric(
            self.data["vn_ai_mps"],
            errors="coerce"
        )

        self.data["ve_ai_mps"] = pd.to_numeric(
            self.data["ve_ai_mps"],
            errors="coerce"
        )

        self.data["ai_speed_mps"] = pd.to_numeric(
            self.data["ai_speed_mps"],
            errors="coerce"
        )

        self.data["ai_speed_kmh"] = pd.to_numeric(
            self.data["ai_speed_kmh"],
            errors="coerce"
        )

        self.data = self.data.dropna(
            subset=[
                "time_s",
                "latitude",
                "longitude",
                "vn_ai_mps",
                "ve_ai_mps",
                "ai_speed_mps",
                "ai_speed_kmh",
            ]
        )

        self.data = (
            self.data
            .sort_values("time_s")
            .reset_index(drop=True)
        )

        print(
            f"Simulation data loaded: "
            f"{len(self.data):,} rows"
        )

        print(
            f"Simulation time range: "
            f"{self.data['time_s'].min():.3f}s "
            f"to "
            f"{self.data['time_s'].max():.3f}s"
        )


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    def get_info(self):

        return {
            "rows": int(len(self.data)),

            "start_time_s": float(
                self.data["time_s"].min()
            ),

            "end_time_s": float(
                self.data["time_s"].max()
            ),

            "sample_rate_hz": 10.0,

            "features": [
                "latitude",
                "longitude",
                "vn_ai_mps",
                "ve_ai_mps",
                "ai_speed_mps",
                "ai_speed_kmh",
            ],
        }


    # ========================================================
    # GET SINGLE SAMPLE
    # ========================================================

    def get_sample(self, time_s: float):

        if self.data.empty:
            raise ValueError(
                "Simulation dataset is empty."
            )

        differences = (
            self.data["time_s"] - time_s
        ).abs()

        index = differences.idxmin()

        row = self.data.loc[index]

        return {
            "timestamp": str(row["timestamp"]),

            "time_s": float(
                row["time_s"]
            ),

            "latitude": float(
                row["latitude"]
            ),

            "longitude": float(
                row["longitude"]
            ),

            "vn_mps": float(
                row["vn_ai_mps"]
            ),

            "ve_mps": float(
                row["ve_ai_mps"]
            ),

            "speed_mps": float(
                row["ai_speed_mps"]
            ),

            "speed_kmh": float(
                row["ai_speed_kmh"]
            ),
        }


    # ========================================================
    # GET RANGE
    # ========================================================

    def get_range(
        self,
        start_time: float,
        end_time: float,
        step: int = 10,
    ):

        filtered = self.data[
            (self.data["time_s"] >= start_time)
            & (self.data["time_s"] < end_time)
        ].copy()

        if filtered.empty:
            return []

        step = max(1, int(step))

        filtered = filtered.iloc[::step]

        samples = []

        for _, row in filtered.iterrows():

            samples.append(
                {
                    "timestamp": str(
                        row["timestamp"]
                    ),

                    "time_s": float(
                        row["time_s"]
                    ),

                    "latitude": float(
                        row["latitude"]
                    ),

                    "longitude": float(
                        row["longitude"]
                    ),

                    "vn_mps": float(
                        row["vn_ai_mps"]
                    ),

                    "ve_mps": float(
                        row["ve_ai_mps"]
                    ),

                    "speed_mps": float(
                        row["ai_speed_mps"]
                    ),

                    "speed_kmh": float(
                        row["ai_speed_kmh"]
                    ),
                }
            )

        return samples


# ============================================================
# GLOBAL SIMULATION INSTANCE
# ============================================================

simulation = NavigationSimulation()