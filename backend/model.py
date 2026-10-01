import torch
import torch.nn as nn
import numpy as np


class TemporalGRU(nn.Module):
    """
    Exact V10.18 Temporal GRU architecture.

    Input:
        20 time steps
        6 sensor features

    Features:
        0 -> gyro_yaw
        1 -> gyro_pitch
        2 -> gyro_roll
        3 -> mag_x
        4 -> mag_y
        5 -> mag_z

    Output:
        0 -> north velocity (m/s)
        1 -> east velocity (m/s)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout=0.2
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Exact checkpoint-compatible prediction head:
        #
        # head.0 -> Linear
        # head.1 -> ReLU
        # head.2 -> Dropout
        # head.3 -> Linear
        #
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        """
        Input:
            (batch_size, sequence_length, input_size)

        Example:
            (1, 20, 6)

        Output:
            (batch_size, 2)
        """

        output, _ = self.gru(x)

        # Take output from the final time step
        last_output = output[:, -1, :]

        return self.head(last_output)


class NavigationModel:
    """
    Production wrapper for the trained V10.18 Temporal GRU.
    """

    def __init__(self, model_path, device=None):

        # =========================================================
        # DEVICE
        # =========================================================

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )

        print(
            f"Loading navigation model on {self.device}"
        )

        # =========================================================
        # LOAD CHECKPOINT
        # =========================================================

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Invalid navigation model checkpoint."
            )

        # =========================================================
        # CHECK REQUIRED INFORMATION
        # =========================================================

        required_keys = [
            "model_state_dict",
            "feature_mean",
            "feature_std",
            "sequence_length",
            "feature_columns",
            "hidden_size",
            "num_layers"
        ]

        missing = [
            key
            for key in required_keys
            if key not in checkpoint
        ]

        if missing:
            raise ValueError(
                f"Checkpoint is missing required fields: {missing}"
            )

        # =========================================================
        # MODEL METADATA
        # =========================================================

        self.sequence_length = int(
            checkpoint["sequence_length"]
        )

        self.feature_columns = list(
            checkpoint["feature_columns"]
        )

        self.feature_mean = np.asarray(
            checkpoint["feature_mean"],
            dtype=np.float32
        )

        self.feature_std = np.asarray(
            checkpoint["feature_std"],
            dtype=np.float32
        )

        self.hidden_size = int(
            checkpoint["hidden_size"]
        )

        self.num_layers = int(
            checkpoint["num_layers"]
        )

        self.input_size = len(
            self.feature_columns
        )

        # =========================================================
        # VALIDATE NORMALIZATION DATA
        # =========================================================

        if len(self.feature_mean) != self.input_size:
            raise ValueError(
                "feature_mean length does not match "
                "feature_columns."
            )

        if len(self.feature_std) != self.input_size:
            raise ValueError(
                "feature_std length does not match "
                "feature_columns."
            )

        # Prevent division by zero
        self.feature_std = np.where(
            self.feature_std < 1e-6,
            1.0,
            self.feature_std
        )

        # =========================================================
        # BUILD MODEL
        # =========================================================

        self.model = TemporalGRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2
        ).to(self.device)

        # =========================================================
        # LOAD TRAINED WEIGHTS
        # =========================================================

        state_dict = checkpoint[
            "model_state_dict"
        ]

        self.model.load_state_dict(
            state_dict
        )

        # Evaluation mode
        self.model.eval()

        # =========================================================
        # DISPLAY MODEL INFORMATION
        # =========================================================

        print(
            "Model loaded successfully."
        )

        print(
            "Sequence length:",
            self.sequence_length
        )

        print(
            "Input features:",
            self.input_size
        )

        print(
            "Features:",
            self.feature_columns
        )

        print(
            "Hidden size:",
            self.hidden_size
        )

        print(
            "GRU layers:",
            self.num_layers
        )

    # =============================================================
    # PREPROCESSING
    # =============================================================

    def preprocess(self, sequence):
        """
        Normalize the sensor sequence using the exact
        mean/std values stored in the V10.18 checkpoint.

        Input:
            shape = (20, 6)

        Output:
            shape = (20, 6)
        """

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        expected_shape = (
            self.sequence_length,
            self.input_size
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                "Invalid sensor sequence shape. "
                f"Expected {expected_shape}, "
                f"received {sequence.shape}"
            )

        normalized = (
            sequence - self.feature_mean
        ) / self.feature_std

        return normalized.astype(
            np.float32
        )

    # =============================================================
    # PREDICTION
    # =============================================================

    def predict(self, sequence):
        """
        Run AI inference.

        Input:
            sequence shape = (20, 6)

        Returns:
            vn_mps
            ve_mps
            speed_mps
            speed_kmh
        """

        # Normalize
        normalized = self.preprocess(
            sequence
        )

        # Add batch dimension
        tensor = torch.from_numpy(
            normalized
        ).unsqueeze(0).to(
            self.device
        )

        # AI inference
        with torch.no_grad():

            prediction = self.model(
                tensor
            )

        # Convert to NumPy
        prediction = (
            prediction
            .cpu()
            .numpy()[0]
        )

        # North velocity
        vn_mps = float(
            prediction[0]
        )

        # East velocity
        ve_mps = float(
            prediction[1]
        )

        # Horizontal speed
        speed_mps = float(
            np.sqrt(
                vn_mps ** 2 +
                ve_mps ** 2
            )
        )

        # Convert m/s → km/h
        speed_kmh = float(
            speed_mps * 3.6
        )

        return {
            "vn_mps": vn_mps,
            "ve_mps": ve_mps,
            "speed_mps": speed_mps,
            "speed_kmh": speed_kmh
        }