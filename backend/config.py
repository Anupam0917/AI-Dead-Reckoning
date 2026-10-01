from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Model directory
MODEL_DIR = (
    PROJECT_ROOT /
    "model"
)


# Processed data
DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)


# Final model path
MODEL_PATH = (
    MODEL_DIR /
    "navigation_model.pt"
)


# Sampling frequency
SAMPLE_RATE_HZ = 10.0

DEFAULT_DT = (
    1.0 /
    SAMPLE_RATE_HZ
)


# Geographic Earth radius
EARTH_RADIUS_M = 6371000.0