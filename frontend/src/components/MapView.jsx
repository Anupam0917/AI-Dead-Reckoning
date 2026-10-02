import { useEffect, useRef, useState } from "react";
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

  // Keep navigation state in refs so updating it
  // does not continuously retrigger the main effect.
  const positionRef = useRef(initialPosition);
  const lastTimeRef = useRef(null);

  useEffect(() => {
    if (!data) return;

    const currentTime = Number(data.time_s);

    // Reset simulation
    if (simulationTime <= 8500.01) {
      const start = [
        Number(data.latitude),
        Number(data.longitude),
      ];

      positionRef.current = start;
      lastTimeRef.current = currentTime;

      setPosition(start);
      setTrajectory([start]);

      return;
    }

    // GNSS available
    if (!gnssLost) {
      const gpsPosition = [
        Number(data.latitude),
        Number(data.longitude),
      ];

      positionRef.current = gpsPosition;
      lastTimeRef.current = currentTime;

      setPosition(gpsPosition);

      setTrajectory((previous) => {
        const updated = [...previous, gpsPosition];
        return updated.slice(-300);
      });

      return;
    }

    // GNSS lost: continue using AI velocity
    const lastTime = lastTimeRef.current;

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

      const currentLat = positionRef.current[0];
      const currentLon = positionRef.current[1];

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

      positionRef.current = aiPosition;

      setPosition(aiPosition);

      setTrajectory((previous) => {
        const updated = [...previous, aiPosition];
        return updated.slice(-300);
      });
    }

    lastTimeRef.current = currentTime;
  }, [data, gnssLost, simulationTime]);

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