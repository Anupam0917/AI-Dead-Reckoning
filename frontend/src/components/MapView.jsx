import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
  useMap,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";

function MapController({ position }) {
  const map = useMap();

  useEffect(() => {
    if (position) {
      map.setView(position, map.getZoom(), {
        animate: false,
      });
    }
  }, [position, map]);

  return null;
}

function MapView({ data, gnssLost, simulationTime }) {
  const initialPosition = [52.464752, -1.482770];

  const [position, setPosition] = useState(initialPosition);
  const [trajectory, setTrajectory] = useState([initialPosition]);
  const [lastTime, setLastTime] = useState(null);

  useEffect(() => {
    if (!data) return;

    const currentTime = Number(data.time_s);

    // Reset simulation
    if (simulationTime <= 8500.01) {
      const start = [
        Number(data.latitude),
        Number(data.longitude),
      ];

      setPosition(start);
      setTrajectory([start]);
      setLastTime(currentTime);
      return;
    }

    // GNSS available: use dataset position
    if (!gnssLost) {
      const gpsPosition = [
        Number(data.latitude),
        Number(data.longitude),
      ];

      setPosition(gpsPosition);

      setTrajectory((previous) => {
        const updated = [...previous, gpsPosition];

        return updated.slice(-300);
      });

      setLastTime(currentTime);
      return;
    }

    // GNSS lost: continue using AI velocity
    if (lastTime !== null) {
      const dt = Math.max(
        0,
        Math.min(currentTime - lastTime, 1)
      );

      const earthRadius = 6371000;

      const vn = Number(data.vn_mps);
      const ve = Number(data.ve_mps);

      const northMovement = vn * dt;
      const eastMovement = ve * dt;

      const currentLat = position[0];
      const currentLon = position[1];

      const newLat =
        currentLat +
        (northMovement / earthRadius) *
          (180 / Math.PI);

      const cosLat = Math.cos(
        (currentLat * Math.PI) / 180
      );

      const newLon =
        currentLon +
        (eastMovement /
          (earthRadius * cosLat)) *
          (180 / Math.PI);

      const aiPosition = [newLat, newLon];

      setPosition(aiPosition);

      setTrajectory((previous) => {
        const updated = [...previous, aiPosition];

        return updated.slice(-300);
      });
    }

    setLastTime(currentTime);
  }, [
    data,
    gnssLost,
    simulationTime,
    lastTime,
    position,
  ]);

  return (
    <MapContainer
      center={initialPosition}
      zoom={15}
      scrollWheelZoom={true}
      className="real-map"
    >
      <TileLayer
        attribution="&copy; OpenStreetMap contributors"
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      <MapController position={position} />

      <Polyline
        positions={trajectory}
        pathOptions={{
          color: gnssLost ? "#dc2626" : "#2563eb",
          weight: 4,
        }}
      />

      <Marker position={position}>
        <Popup>
          <strong>AI Dead Reckoning</strong>
          <br />
          {gnssLost
            ? "GNSS LOST • AI navigation active"
            : "GNSS navigation active"}
        </Popup>
      </Marker>
    </MapContainer>
  );
}

export default MapView;