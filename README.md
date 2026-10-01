\# AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation



\## SIH 2026 Prototype



\### Problem Statement



GNSS/GPS signals can become unavailable or unreliable in environments such as tunnels, underground areas, dense urban regions, and areas affected by signal interference.



When GNSS is unavailable, a navigation system needs another way to estimate vehicle movement and maintain a continuous navigation trajectory.



This project develops an AI-assisted dead reckoning prototype that estimates vehicle velocity from onboard motion sensors and continues navigation during simulated GNSS-denied conditions.



\---



\## Proposed Solution



The system uses:



\- Accelerometer and gyroscope measurements from the IMU

\- Magnetometer measurements

\- A Temporal GRU based deep learning model

\- North and East velocity estimation

\- Dead reckoning based position propagation

\- GNSS outage simulation

\- A FastAPI backend

\- A React + Leaflet navigation dashboard



The AI model estimates the vehicle's northward and eastward velocity from a sequence of sensor measurements.



These velocity estimates are integrated over time to estimate the vehicle's position when GNSS is unavailable.



\---



\## System Architecture



```text

&#x20;                   IO-VNBD Dataset

&#x20;                          |

&#x20;                          v

&#x20;               IMU + Magnetometer Data

&#x20;                          |

&#x20;                          v

&#x20;                   Preprocessing

&#x20;                          |

&#x20;                          v

&#x20;                Temporal GRU Model

&#x20;                   /           \\

&#x20;                  /             \\

&#x20;                 v               v

&#x20;         North Velocity     East Velocity

&#x20;                 \\             /

&#x20;                  \\           /

&#x20;                   v         v

&#x20;                 Dead Reckoning

&#x20;                        |

&#x20;                        v

&#x20;               Estimated Position

&#x20;                        |

&#x20;                        v

&#x20;                   FastAPI API

&#x20;                        |

&#x20;                        v

&#x20;                React Dashboard

&#x20;                        |

&#x20;                        v

&#x20;                   Leaflet Map





\## Project Overview



This project is an AI-assisted navigation system designed to maintain vehicle position estimation during GNSS-denied conditions.



The system uses IMU and magnetometer sensor data as input to a Temporal GRU model. The model estimates north and east velocity, which is then integrated using dead reckoning to estimate the vehicle's position.



The current prototype uses recorded sensor data from the IO-VNBD dataset and simulates GNSS signal loss to demonstrate continued navigation.



\---



\## Technology Stack



\### AI / Machine Learning

\- Python

\- PyTorch

\- Pandas

\- NumPy

\- Scikit-learn

\- Temporal GRU



\### Backend

\- FastAPI

\- Uvicorn



\### Frontend

\- React

\- Vite

\- Leaflet

\- React-Leaflet



\### Dataset

\- IO-VNBD Dataset



\---



\## Prototype Workflow



1\. Recorded IMU and magnetometer data is loaded from the dataset.

2\. Sensor data is preprocessed into sequences of 20 samples.

3\. The Temporal GRU receives the sensor sequence.

4\. The model predicts north and east velocity.

5\. The velocity is integrated using dead reckoning.

6\. The estimated position is sent through the FastAPI backend.

7\. The React dashboard displays the navigation state and trajectory.

8\. GNSS loss can be simulated from the dashboard.



\---



\## Prototype Demonstration



The prototype dashboard provides:



\- GNSS status

\- IMU status

\- AI model status

\- Dead reckoning status

\- Current speed

\- North velocity

\- East velocity

\- Latitude and longitude

\- Interactive navigation map

\- GNSS loss simulation



When GNSS loss is simulated, the system continues navigation using the AI-predicted velocity and dead reckoning.



\---



\## Current Status



The prototype successfully demonstrates the complete pipeline:



Dataset → Sensor Data → AI Model → Velocity Estimation → Dead Reckoning → FastAPI → React Dashboard → Map



The current implementation uses recorded IO-VNBD sensor data. Future development will connect the same inference pipeline to live smartphone or vehicle sensors.

