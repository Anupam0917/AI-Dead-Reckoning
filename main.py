from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


from backend.config import MODEL_PATH
from backend.model import NavigationModel
from backend.preprocessing import SensorPreprocessor
from backend.dead_reckoning import DeadReckoning
from backend.simulation import simulation


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Intelligent Dead Reckoning API",
    description=(
        "AI-ML based GNSS-denied navigation system "
        "using Temporal GRU and dead reckoning."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING PRODUCTION NAVIGATION MODEL")
print("=" * 60)

model = NavigationModel(MODEL_PATH)

preprocessor = SensorPreprocessor(model)

print("Production model ready.")


# ============================================================
# SENSOR INPUT MODEL
# ============================================================

class SensorSample(BaseModel):

    gyro_yaw: float
    gyro_pitch: float
    gyro_roll: float

    mag_x: float
    mag_y: float
    mag_z: float


# ============================================================
# PREDICTION REQUEST
# ============================================================

class PredictionRequest(BaseModel):

    samples: List[SensorSample] = Field(
        ...,
        min_length=20,
        max_length=20,
        description=(
            "Exactly 20 consecutive IMU/magnetometer samples."
        ),
    )

    latitude: float = Field(
        ...,
        description="Starting latitude in degrees.",
    )

    longitude: float = Field(
        ...,
        description="Starting longitude in degrees.",
    )

    dt: float = Field(
        default=0.1,
        gt=0,
        description="Time interval between samples in seconds.",
    )


# ============================================================
# PREDICTION RESPONSE
# ============================================================

class PredictionResponse(BaseModel):

    latitude: float
    longitude: float

    vn_mps: float
    ve_mps: float

    speed_mps: float
    speed_kmh: float

    north_displacement_m: float
    east_displacement_m: float


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "AI Intelligent Dead Reckoning API",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(model.device),
        "sequence_length": model.sequence_length,
        "input_features": model.feature_columns,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model_type": "Temporal GRU",
        "sequence_length": model.sequence_length,
        "input_features": model.feature_columns,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "device": str(model.device),
    }


# ============================================================
# AI PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    try:

        samples = [
            sample.model_dump()
            for sample in request.samples
        ]

        sequence = (
            preprocessor.prepare_sequence(
                samples
            )
        )

        prediction = model.predict(
            sequence
        )

        dead_reckoning = DeadReckoning(
            latitude=request.latitude,
            longitude=request.longitude,
        )

        updated_position = dead_reckoning.update(
            vn_mps=prediction["vn_mps"],
            ve_mps=prediction["ve_mps"],
            dt=request.dt,
        )

        return PredictionResponse(

            latitude=updated_position["latitude"],
            longitude=updated_position["longitude"],

            vn_mps=prediction["vn_mps"],
            ve_mps=prediction["ve_mps"],

            speed_mps=prediction["speed_mps"],
            speed_kmh=prediction["speed_kmh"],

            north_displacement_m=(
                updated_position[
                    "north_displacement_m"
                ]
            ),

            east_displacement_m=(
                updated_position[
                    "east_displacement_m"
                ]
            ),
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(error)}"
            ),
        )


# ============================================================
# SIMULATION INFORMATION
# ============================================================

@app.get("/simulation/info")
def simulation_info():

    return simulation.get_info()


# ============================================================
# SINGLE SIMULATION SAMPLE
# ============================================================

@app.get("/simulation/sample")
def simulation_sample(
    time_s: float = Query(
        ...,
        description="Dataset time in seconds.",
    )
):

    info = simulation.get_info()

    if time_s < info["start_time_s"]:

        raise HTTPException(
            status_code=400,
            detail=(
                f"time_s must be >= "
                f"{info['start_time_s']:.3f}"
            ),
        )

    if time_s > info["end_time_s"]:

        raise HTTPException(
            status_code=400,
            detail=(
                f"time_s must be <= "
                f"{info['end_time_s']:.3f}"
            ),
        )

    return simulation.get_sample(
        time_s
    )


# ============================================================
# SIMULATION RANGE
# ============================================================

@app.get("/simulation/range")
def simulation_range(
    start_time: float = Query(
        ...,
        description="Start time in seconds.",
    ),

    end_time: float = Query(
        ...,
        description="End time in seconds.",
    ),

    step: int = Query(
        default=10,
        ge=1,
        description=(
            "Return every Nth row. "
            "10 means approximately 1 sample/second "
            "for the 10 Hz dataset."
        ),
    ),
):

    if end_time <= start_time:

        raise HTTPException(
            status_code=400,
            detail=(
                "end_time must be greater "
                "than start_time."
            ),
        )

    info = simulation.get_info()

    if start_time < info["start_time_s"]:

        raise HTTPException(
            status_code=400,
            detail=(
                f"start_time must be >= "
                f"{info['start_time_s']:.3f}"
            ),
        )

    if end_time > info["end_time_s"]:

        raise HTTPException(
            status_code=400,
            detail=(
                f"end_time must be <= "
                f"{info['end_time_s']:.3f}"
            ),
        )

    samples = simulation.get_range(
        start_time=start_time,
        end_time=end_time,
        step=step,
    )

    return {
        "start_time_s": start_time,
        "end_time_s": end_time,
        "step": step,
        "sample_count": len(samples),
        "samples": samples,
    }