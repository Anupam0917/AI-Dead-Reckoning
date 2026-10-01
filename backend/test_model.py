import sys
from pathlib import Path

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.model import NavigationModel
from backend.preprocessing import SensorPreprocessor


MODEL_PATH = PROJECT_ROOT / "model" / "navigation_model.pt"


def main():
    print("=" * 60)
    print("AI DEAD RECKONING MODEL TEST")
    print("=" * 60)

    # Load model
    model = NavigationModel(MODEL_PATH)

    # Create preprocessor
    preprocessor = SensorPreprocessor(model)

    print("\nPreprocessor initialized successfully.")

    # Create 20 dummy sensor samples
    samples = []

    for _ in range(20):
        samples.append(
            {
                "gyro_yaw": 0.01,
                "gyro_pitch": 0.02,
                "gyro_roll": 0.01,
                "mag_x": 30.0,
                "mag_y": 5.0,
                "mag_z": 40.0,
            }
        )

    # Prepare sequence
    sequence = preprocessor.prepare(samples)

    print("\nPrepared sequence:")
    print("Shape:", sequence.shape)

    print("\nExpected shape:")
    print(
        "(",
        model.sequence_length,
        ",",
        len(model.feature_columns),
        ")",
    )

    # Run AI prediction
    prediction = model.predict(sequence)

    print("\nAI prediction:")
    print(f"North velocity : {prediction['vn_mps']:.4f} m/s")
    print(f"East velocity  : {prediction['ve_mps']:.4f} m/s")
    print(f"Speed          : {prediction['speed_mps']:.4f} m/s")
    print(f"Speed          : {prediction['speed_kmh']:.4f} km/h")

    print("\n" + "=" * 60)
    print("MODEL TEST SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()