from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .simulation import simulation


app = FastAPI(
    title="AI-ML Intelligent Dead Reckoning API",
    description="SIH prototype API for AI-based GNSS-denied navigation",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "AI-ML Intelligent Dead Reckoning",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "service": "dead-reckoning-api",
    }


# ============================================================
# DATASET INFORMATION
# ============================================================

@app.get("/api/info")
def info():

    try:
        return simulation.get_info()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# SINGLE NAVIGATION SAMPLE
# ============================================================

@app.get("/api/sample")
def sample(time_s: float):

    try:
        return simulation.get_sample(time_s)

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ============================================================
# NAVIGATION TRAJECTORY RANGE
# ============================================================

@app.get("/api/range")
def data_range(
    start_time: float,
    end_time: float,
    step: int = 10,
):

    try:

        return simulation.get_range(
            start_time=start_time,
            end_time=end_time,
            step=step,
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )