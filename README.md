# AI-Driven Multi-Threat Signal Integrity Management for Resilient NavIC-Based Railway Positioning

An AI-driven research framework for monitoring, forecasting and detecting GNSS signal integrity threats in NavIC-enabled railway positioning systems.

The project integrates machine learning, deep learning, signal analysis and geospatial visualisation to demonstrate a unified framework for identifying and responding to signal degradation caused by ionospheric scintillation, multipath and spoofing.

The framework is developed with a focus on Indian Railways and railway safety applications such as **Kavach (Automatic Train Protection System)**.

---

## Project Overview

Satellite-based positioning systems are increasingly important for railway localisation. However, GNSS signals are vulnerable to several environmental and intentional disruptions, including:

- Ionospheric scintillation
- Multipath propagation
- Signal blockage
- GNSS spoofing

These disturbances can affect signal reliability and positioning availability.

This project proposes an AI-driven multi-threat signal integrity framework that combines forecasting, anomaly detection and railway corridor visualisation to support resilient NavIC-based railway positioning.

The framework is organised into four major objectives.

## Project Objectives

### Objective 1 – Ionospheric Scintillation Forecasting

Develop a time-series forecasting framework to predict signal degradation using GNSS observation data.

**Key components:**
- GNSS RINEX observation data processing.
- NavIC L5 signal feature extraction.
- Signal-quality and carrier-phase-derived feature engineering.
- Temporal feature generation and historical observation analysis.
- DeepAR-based time-series forecasting.
- 10-minute-ahead signal degradation prediction.

**Dataset:**
- Source: NASA CDDIS.
- Observation format: RINEX / Compact RINEX.
- Sampling interval: 30 seconds.
- Reference stations: IISC00IND, LCK400IND and JDPR00IND.
- Observation period: 2022–2024.

The current implementation uses `sigma_phi_proxy` as its forecasting target. This is a proxy derived from available observations and should not be interpreted as a directly measured conventional scintillation index.

### Objective 2 – Multi-Class Signal Threat Detection

Develop a machine learning-based classification system to identify different GNSS signal conditions.

**Target classes:**
1. Genuine
2. Multipath
3. Spoofing

**Key components:**
- Signal feature engineering.
- C/N0 and carrier-phase analysis.
- Pseudorange and code-carrier consistency indicators.
- Carrier-phase variation and rate features.
- Synthetic multipath and spoofing scenario generation.
- LightGBM-based classification.
- Threat prediction and confidence estimation.

The model is trained and evaluated using genuine observations and synthetically injected threat scenarios.

### Objective 3 – NavIC Railway Corridor Digital Twin

Integrate the forecasting and threat detection pipelines into a unified railway integrity monitoring dashboard.

**Technology stack:**
- Python
- Streamlit
- CesiumJS
- Leaflet
- JavaScript
- HTML
- Pandas and NumPy

**Dashboard features:**

1. **Integrated Monitoring Dashboard**
   - Unified interface for O1 and O2 model outputs.
   - Station and satellite selection.
   - Signal integrity status and prediction summaries.

2. **3D Digital Twin**
   - Interactive 3D globe visualisation.
   - Selected reference station and satellite representation.
   - Schematic satellite-to-station signal visualisation.

3. **Railway GIS Visualisation**
   - Railway corridor visualisation using GIS data.
   - Train movement animation along the railway route.
   - Visual indication of affected railway sections.
   - Integration of signal threat information into the railway map.

4. **Scenario-Based Threat Simulation**
   - Genuine signal scenario.
   - Multipath scenario.
   - Spoofing scenario.
   - Model-based prediction of induced scenarios.

5. **10-Minute Forecasting**
   - Integration of the O1 DeepAR forecasting pipeline.
   - Display of predicted signal degradation.

6. **Threat Intelligence**
   - Integration of the O2 LightGBM classifier.
   - Predicted threat category.
   - Classification confidence.
   - Signal integrity alerts.

7. **Railway Integrity Decision Support**
   - Combines O1 forecasting and O2 threat classification.
   - Provides signal monitoring and handling recommendations.
   - Demonstrates an integrated integrity monitoring workflow.

### Objective 4 – Deployment Readiness Evaluation

Evaluate the integrated framework using held-out GNSS observations and synthetic threat scenarios.

**Evaluation areas:**
- Signal availability and usability analysis.
- Comparison of reactive and proactive integrity policies.
- Signal-consistency proxy compliance.
- Held-out synthetic multipath detection.
- Held-out synthetic spoofing detection.
- Seasonal robustness analysis across different ionospheric conditions.
- Railway corridor performance considerations.

The current evaluation uses visibility and signal-consistency proxies. It does not constitute a complete positioning availability assessment or certified protection-level evaluation using HPL/VPL.

---

## System Architecture

```text
                NASA CDDIS RINEX DATA
                         |
                         v
              DATA PREPROCESSING
                         |
                         v
             FEATURE ENGINEERING
                         |
              +----------+----------+
              |                     |
              v                     v
       OBJECTIVE 1             OBJECTIVE 2
       DeepAR Model            LightGBM Model
              |                     |
              v                     v
       10-Minute Signal       Multi-Class Threat
       Degradation Forecast   Classification
              |                     |
              +----------+----------+
                         |
                         v
                 OBJECTIVE 3
             INTEGRATED FRAMEWORK
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
          Streamlit   CesiumJS    Railway GIS
          Dashboard   3D Twin     Visualisation
             |           |           |
             +-----------+-----------+
                         |
                         v
              INTEGRITY DECISION
                   SUPPORT
                         |
                         v
                 OBJECTIVE 4
              EVALUATION FRAMEWORK
                         |
                         v
             PERFORMANCE REPORTS
```

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | LightGBM |
| Deep Learning | DeepAR |
| Data Processing | Pandas, NumPy, Xarray |
| GNSS Data Processing | GeoRinex, RINEX / CRINEX tools |
| Dashboard | Streamlit |
| 3D Visualisation | CesiumJS |
| GIS Visualisation | Leaflet, GeoJSON |
| Frontend | HTML, CSS, JavaScript |
| Data Storage | Parquet, CSV |
| Model Persistence | Joblib |
| Version Control | Git, GitHub |

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Cesium ion

The 3D visualisation requires a valid Cesium ion access token.

Create the following file:

```text
.streamlit/secrets.toml
```

Add:

```toml
CESIUM_ION_TOKEN = "YOUR_CESIUM_ION_ACCESS_TOKEN"
```

Do not commit your access token or other credentials to GitHub.

### 5. Configure dataset and model paths

Ensure the required processed datasets and saved model artefacts are available at the paths expected by the application.

The framework reuses the trained O1 and O2 models and their associated preprocessing artefacts.

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the dashboard.

---

## Running Objective 4

To execute the deployment-readiness evaluation:

```bash
python objective_4/run_o4.py
```

The evaluation script reuses saved O1 and O2 outputs rather than retraining the models.

Generated reports and evaluation results are saved under:

```text
objective_4/results/
```

---

## Dataset Information

The project uses GNSS observation data obtained from NASA's Crustal Dynamics Data Information System (CDDIS).

**Reference stations:**
- IISC00IND
- LCK400IND
- JDPR00IND

**NavIC observables:**
- C5A – Pseudorange observable.
- L5A – Carrier-phase observable.
- S5A – Signal strength observable.

The available single-frequency observations impose limitations on the direct calculation of conventional dual-frequency ionospheric metrics. Consequently, selected signal degradation indicators are represented using derived proxies.

The project uses synthetic signal perturbations to demonstrate multipath and spoofing detection.

---

## Important Research Limitations

This project is a research prototype and decision-support demonstration. The following distinctions are important:

- The 3D satellite representations are illustrative and are not generated from live NavIC ephemeris propagation.
- The train tracker demonstrates simulated train movement along the railway GIS route; it is not connected to a live railway tracking feed.
- The affected railway route section is a visualisation of the selected threat scenario, not a field-validated physical threat location.
- Multipath and spoofing detection results are based on synthetic injected scenarios and do not establish field performance against real attacks.
- `sigma_phi_proxy` is a derived proxy and should not be interpreted as a directly measured conventional S4 or sigma-phi index.
- Signal-handling recommendations are decision-support outputs. The prototype does not directly control a GNSS receiver or railway safety system.
- Signal-consistency compliance is not equivalent to certified HPL/VPL protection-level compliance.
- Four-satellite visibility is not equivalent to a successfully computed and validated positioning solution.

The framework is intended for research, experimentation and demonstration. Further validation with real receiver data, satellite geometry, positioning solutions and operational railway systems is required before safety-critical deployment.

---

## Future Enhancements

- Integration of real-time NavIC receiver observations.
- Satellite orbit and ephemeris-based visualisation.
- Real-time railway tracking integration.
- Improved multipath and spoofing detection using field-collected datasets.
- GNSS positioning solution and protection-level computation.
- Advanced integrity monitoring and adaptive satellite selection.
- Deployment and performance validation in representative railway environments.

---

## Applications

Potential research and application areas include:

- NavIC-enabled railway localisation.
- Indian Railways and Kavach-related positioning research.
- GNSS signal integrity monitoring.
- Intelligent transportation systems.
- Satellite navigation resilience.
- AI-based anomaly detection and forecasting.

---

## Acknowledgements

- **NASA CDDIS** – GNSS observation data.
- **Amrita Vishwa Vidyapeetham** – Academic and research support.
- **CesiumJS** – 3D geospatial visualisation platform.
- **Streamlit** – Interactive dashboard development.
- **Leaflet** – Interactive GIS mapping.

---

## Author

M.Tech – Artificial Intelligence  
Amrita Vishwa Vidyapeetham, Coimbatore

**Project:** AI-Driven Multi-Threat Signal Integrity Management for Resilient NavIC-Based Railway Positioning

---

*This repository contains the research implementation, model pipelines, integration framework and evaluation components developed as part of the M.Tech project.*
