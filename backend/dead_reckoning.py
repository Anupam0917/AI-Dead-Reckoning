import math

from .config import EARTH_RADIUS_M


class DeadReckoning:
    """
    Converts estimated North/East velocity into
    updated geographic position.

    Coordinate system:

        North velocity -> latitude
        East velocity  -> longitude
    """

    def __init__(
        self,
        latitude,
        longitude
    ):
        self.latitude = float(latitude)
        self.longitude = float(longitude)

        # Total local displacement from the starting point
        self.north_m = 0.0
        self.east_m = 0.0

    def update(
        self,
        vn_mps,
        ve_mps,
        dt
    ):
        """
        Update position using North/East velocity.

        Parameters
        ----------
        vn_mps : float
            North velocity in meters/second.

        ve_mps : float
            East velocity in meters/second.

        dt : float
            Time interval in seconds.

        Returns
        -------
        dict
            Updated navigation state.
        """

        vn_mps = float(vn_mps)
        ve_mps = float(ve_mps)
        dt = float(dt)

        if dt < 0:
            raise ValueError(
                "dt cannot be negative."
            )

        # ---------------------------------------------------------
        # Calculate local displacement
        # ---------------------------------------------------------

        north_displacement = vn_mps * dt
        east_displacement = ve_mps * dt

        # ---------------------------------------------------------
        # Accumulate displacement
        # ---------------------------------------------------------

        self.north_m += north_displacement
        self.east_m += east_displacement

        # ---------------------------------------------------------
        # Convert North displacement to latitude
        #
        # Arc length:
        #
        # distance = radius × angle
        #
        # angle = distance / radius
        # ---------------------------------------------------------

        delta_latitude = (
            north_displacement /
            EARTH_RADIUS_M
        )

        # ---------------------------------------------------------
        # Convert East displacement to longitude
        #
        # Longitude distance depends on latitude.
        #
        # east_distance =
        #     R × cos(latitude) × longitude_angle
        # ---------------------------------------------------------

        latitude_radians = math.radians(
            self.latitude
        )

        cos_latitude = math.cos(
            latitude_radians
        )

        # Protect against division by an extremely small value
        if abs(cos_latitude) < 1e-12:
            raise ValueError(
                "Longitude update is unstable near the poles."
            )

        delta_longitude = (
            east_displacement /
            (
                EARTH_RADIUS_M *
                cos_latitude
            )
        )

        # ---------------------------------------------------------
        # Convert radians to degrees
        # ---------------------------------------------------------

        self.latitude += math.degrees(
            delta_latitude
        )

        self.longitude += math.degrees(
            delta_longitude
        )

        # ---------------------------------------------------------
        # Return updated navigation state
        # ---------------------------------------------------------

        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "north_m": self.north_m,
            "east_m": self.east_m,
            "north_displacement_m": north_displacement,
            "east_displacement_m": east_displacement,
        }

    def reset(
        self,
        latitude,
        longitude
    ):
        """
        Reset dead-reckoning state to a new position.
        """

        self.latitude = float(latitude)
        self.longitude = float(longitude)

        self.north_m = 0.0
        self.east_m = 0.0