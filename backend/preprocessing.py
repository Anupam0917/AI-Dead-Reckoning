import numpy as np


class SensorPreprocessor:
    """
    Prepares raw gyroscope + magnetometer data
    for the V10.18 Temporal GRU model.

    Expected feature order:

    0 -> gyro_yaw
    1 -> gyro_pitch
    2 -> gyro_roll
    3 -> mag_x
    4 -> mag_y
    5 -> mag_z
    """

    REQUIRED_FEATURES = [
        "gyro_yaw",
        "gyro_pitch",
        "gyro_roll",
        "mag_x",
        "mag_y",
        "mag_z",
    ]

    def __init__(self, navigation_model):
        self.model = navigation_model

        if self.model.feature_columns != self.REQUIRED_FEATURES:
            raise ValueError(
                "Model feature columns do not match the expected "
                "V10.18 feature order.\n"
                f"Expected: {self.REQUIRED_FEATURES}\n"
                f"Found: {self.model.feature_columns}"
            )

        self.sequence_length = self.model.sequence_length

    def validate_sample(self, sample):
        """
        Validate one sensor sample.

        Expected format:

        {
            "gyro_yaw": float,
            "gyro_pitch": float,
            "gyro_roll": float,
            "mag_x": float,
            "mag_y": float,
            "mag_z": float
        }
        """

        missing = [
            feature
            for feature in self.REQUIRED_FEATURES
            if feature not in sample
        ]

        if missing:
            raise ValueError(
                f"Missing sensor features: {missing}"
            )

        values = []

        for feature in self.REQUIRED_FEATURES:
            value = sample[feature]

            if value is None:
                raise ValueError(
                    f"Sensor value '{feature}' cannot be None."
                )

            value = float(value)

            if not np.isfinite(value):
                raise ValueError(
                    f"Sensor value '{feature}' must be finite."
                )

            values.append(value)

        return np.asarray(values, dtype=np.float32)

    def create_sequence(self, samples):
        """
        Convert 20 sensor samples into the model input matrix.

        Returns:

        shape = (20, 6)
        """

        if len(samples) != self.sequence_length:
            raise ValueError(
                f"Expected exactly {self.sequence_length} sensor samples, "
                f"received {len(samples)}."
            )

        sequence = np.array(
            [self.validate_sample(sample) for sample in samples],
            dtype=np.float32,
        )

        expected_shape = (
            self.sequence_length,
            len(self.REQUIRED_FEATURES),
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                f"Invalid sequence shape. "
                f"Expected {expected_shape}, "
                f"received {sequence.shape}."
            )

        return sequence

    def prepare(self, samples):
        """
        Prepare raw sensor samples for NavigationModel.predict().

        Returns a NumPy array with shape:

        (20, 6)
        """

        sequence = self.create_sequence(samples)

        return sequence