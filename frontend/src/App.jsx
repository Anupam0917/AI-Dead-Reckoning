import { useEffect, useState } from "react";
import "./App.css";
import MapView from "./components/MapView";

const API_URL = "https://ai-dead-reckoning-production.up.railway.app";

function App() {
  const [data, setData] = useState(null);
  const [simulationTime, setSimulationTime] = useState(8500);
  const [running, setRunning] = useState(false);
  const [gnssLost, setGnssLost] = useState(false);
  const [error, setError] = useState("");

  // Fetch one sample from FastAPI
  const fetchSample = async (time) => {
    try {
      const url = `${API_URL}/api/sample?time_s=${time.toFixed(3)}`;

      console.log("Fetching:", url);

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(
          `API request failed: ${response.status} ${response.statusText}`
        );
      }

      const result = await response.json();

      console.log("Fetched sample:", result);

      setData(result);
      setError("");
    } catch (err) {
      console.error("Backend error:", err);
      setError("Backend connection failed");
    }
  };

  // Load initial sample
  useEffect(() => {
    fetchSample(8500);
  }, []);

  // Run simulation
  useEffect(() => {
    if (!running) {
      return;
    }

    console.log("Simulation started");

    const timer = setInterval(() => {
      setSimulationTime((previous) => {
        const next = previous + 0.5;

        console.log("Simulation time:", next);

        if (next >= 8560) {
          setRunning(false);
          return 8560;
        }

        return next;
      });
    }, 500);

    return () => {
      console.log("Simulation timer stopped");
      clearInterval(timer);
    };
  }, [running]);

  // Fetch new sensor/model data whenever simulation time changes
  useEffect(() => {
    console.log("Simulation time changed:", simulationTime);

    fetchSample(simulationTime);
  }, [simulationTime]);

  // Start normal navigation
  const startNavigation = () => {
    console.log("START NAVIGATION clicked");

    setGnssLost(false);
    setRunning(true);
  };

  // Simulate GNSS outage
  const simulateGnssLoss = () => {
    console.log("SIMULATE GNSS LOSS clicked");

    setGnssLost(true);
    setRunning(true);
  };

  // Reset simulation
  const resetSimulation = () => {
    console.log("RESET clicked");

    setRunning(false);
    setGnssLost(false);
    setSimulationTime(8500);
    fetchSample(8500);
  };

  // Safe values
  const speed = data?.speed_kmh ?? 0;
  const northVelocity = data?.vn_mps ?? 0;
  const eastVelocity = data?.ve_mps ?? 0;
  const latitude = data?.latitude ?? 0;
  const longitude = data?.longitude ?? 0;

  return (
    <div className="app">

      {/* HEADER */}
      <header className="header">
        <div>
          <h1>AI Intelligent Dead Reckoning</h1>
          <p>GNSS-Denied Navigation System</p>
        </div>

        <div className="system-badge">
          <span
            className={`status-dot ${
              error ? "offline" : "online"
            }`}
          ></span>

          {error ? "BACKEND ERROR" : "SYSTEM READY"}
        </div>
      </header>

      {/* MAIN DASHBOARD */}
      <main className="dashboard">

        {/* LEFT PANEL */}
        <section className="left-panel">

          {/* SYSTEM STATUS */}
          <div className="card">

            <div className="card-title">
              <span>System Status</span>

              <span className="live-label">
                {running ? "RUNNING" : "READY"}
              </span>
            </div>

            <div className="status-row">
              <span>GNSS</span>

              <span className="status-value">
                <span
                  className={`status-dot ${
                    gnssLost ? "offline" : "online"
                  }`}
                ></span>

                {gnssLost ? "SIGNAL LOST" : "CONNECTED"}
              </span>
            </div>

            <div className="status-row">
              <span>IMU</span>

              <span className="status-value online-text">
                <span className="status-dot online"></span>
                ACTIVE
              </span>
            </div>

            <div className="status-row">
              <span>AI MODEL</span>

              <span className="status-value online-text">
                <span className="status-dot online"></span>
                ACTIVE
              </span>
            </div>

            <div className="status-row">
              <span>DEAD RECKONING</span>

              <span className="status-value">
                {gnssLost ? "RUNNING" : "STANDBY"}
              </span>
            </div>

          </div>

          {/* NAVIGATION DATA */}
          <div className="card">

            <div className="card-title">
              <span>Navigation Data</span>
            </div>

            <div className="metrics-grid">

              <div className="metric">
                <span className="metric-label">
                  SPEED
                </span>

                <strong>
                  {speed.toFixed(2)}
                </strong>

                <small>
                  km/h
                </small>
              </div>

              <div className="metric">
                <span className="metric-label">
                  TIME
                </span>

                <strong>
                  {simulationTime.toFixed(1)}
                </strong>

                <small>
                  s
                </small>
              </div>

              <div className="metric">
                <span className="metric-label">
                  NORTH VELOCITY
                </span>

                <strong>
                  {northVelocity.toFixed(2)}
                </strong>

                <small>
                  m/s
                </small>
              </div>

              <div className="metric">
                <span className="metric-label">
                  EAST VELOCITY
                </span>

                <strong>
                  {eastVelocity.toFixed(2)}
                </strong>

                <small>
                  m/s
                </small>
              </div>

            </div>

          </div>

          {/* CURRENT POSITION */}
          <div className="card">

            <div className="card-title">
              <span>Current Position</span>
            </div>

            <div className="coordinate">
              <span>LATITUDE</span>

              <strong>
                {latitude.toFixed(6)}°
              </strong>
            </div>

            <div className="coordinate">
              <span>LONGITUDE</span>

              <strong>
                {longitude.toFixed(6)}°
              </strong>
            </div>

          </div>

        </section>

        {/* MAP */}
        <section className="map-card">

          <div className="map-header">

            <div>
              <h2>Navigation Map</h2>

              <span>
                Real sensor dataset simulation
              </span>
            </div>

            <div className="gps-indicator">

              <span
                className={`status-dot ${
                  gnssLost ? "offline" : "online"
                }`}
              ></span>

              {gnssLost
                ? "GNSS LOST • AI DR ACTIVE"
                : "GNSS ACTIVE"}

            </div>

          </div>

          <MapView
            data={data}
            gnssLost={gnssLost}
            simulationTime={simulationTime}
          />

        </section>

      </main>

      {/* NAVIGATION CONTROL */}
      <section className="control-panel">

        <div className="control-info">

          <span className="control-label">
            NAVIGATION CONTROL
          </span>

          <div className="control-status">

            <span
              className={`status-dot ${
                gnssLost ? "offline" : "online"
              }`}
            ></span>

            {gnssLost
              ? "AI DEAD RECKONING ACTIVE"
              : running
              ? "NAVIGATION RUNNING"
              : "GNSS SIGNAL AVAILABLE"}

          </div>

        </div>

        <div className="controls">

          <button
            className="btn primary"
            onClick={startNavigation}
          >
            ▶ START NAVIGATION
          </button>

          <button
            className="btn danger"
            onClick={simulateGnssLoss}
          >
            SIMULATE GNSS LOSS
          </button>

          <button
            className="btn secondary"
            onClick={resetSimulation}
          >
            RESET
          </button>

        </div>

      </section>

      {/* FOOTER */}
      <footer>

        <span>
          AI-ML Intelligent Dead Reckoning
        </span>

        <span>
          Temporal GRU • IMU + Magnetometer • 10 Hz
        </span>

      </footer>

    </div>
  );
}

export default App;