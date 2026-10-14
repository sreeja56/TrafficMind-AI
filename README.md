# 🚦 TrafficMind AI: Intelligent Urban Traffic Congestion Prediction & Dynamic Rerouting Engine

> **PromptWars x Creative Community VVITU Hackathon Project**  
> An intelligent transportation system (ITS) prototype that predicts metropolitan congestion levels, explains telemetry drivers, visualizes dynamic road network friction, and plans congestion-bypassing and emergency green-corridor routes.

---

## 📌 Executive Summary

Urban traffic congestion accounts for billions of dollars in lost productivity, inflated fuel waste, and increased carbon emissions each year. Even more critically, emergency response vehicles (ambulances, fire engines) often lose precious minutes of the **"Golden Hour"** stuck behind static GPS directions that cannot anticipate dynamic gridlocks.

**TrafficMind AI** solves this by unifying:
1. **Machine Learning Predictive Modeling**: A Scikit-Learn Random Forest Classifier that predicts congestion severity (`Low`, `Moderate`, `Severe`) from multi-modal sensor inputs (vehicle volume, average speed, time of day, weather, and road incidents).
2. **Transparent Decision Explanations**: Real-time feature attribution explaining *why* traffic is slowing down (e.g., volume-to-capacity saturation vs. weather-induced pavement friction).
3. **Dynamic Network Rerouting (NetworkX)**: An interactive 12-node metropolitan road network where edge weights dynamically reflect real-time travel time rather than static distance.
4. **Emergency Green-Corridor Dispatch**: Signal preemption and priority lane clearance simulation saving critical minutes during medical and fire emergencies.

---

## 🚀 Key Features

| Feature | Description | Tech Used |
| :--- | :--- | :--- |
| **Live Congestion Predictor** | Real-time classification into *Low*, *Moderate*, or *Severe* congestion with confidence probabilities. | `scikit-learn`, `RandomForestClassifier` |
| **What-If Telemetry Simulator** | Interactive controls for Hour (0–23), Day of Week, Vehicle Volume (veh/hr/lane), Observed Speed (km/h), Weather (Clear/Rain/Fog/Storm), and Incidents. | `Streamlit` |
| **Short-Term Forecast** | 12-hour illustrative congestion and speed projection highlighting morning and evening rush cycles. | `Plotly`, `Pandas` |
| **Dynamic City Network Map** | Interactive graph of 12 urban hubs and 20 major transport corridors color-coded by real-time congestion state. | `NetworkX`, `Plotly` |
| **AI Alternative Route Planner** | Dijkstra-based comparative routing: Baseline Shortest Distance vs. AI Congestion-Aware Dynamic Route with travel time savings. | `NetworkX` |
| **Emergency Green-Corridor** | Priority routing with simulated traffic light preemption and emergency clearance for Trauma Ambulances and Fire Engines. | `NetworkX`, `Plotly` |
| **Model Evaluation & Transparency** | Confusion matrix heatmap, precision/recall/F1 metrics, and Gini feature importances on a held-out test split. | `scikit-learn`, `Plotly` |
| **Quick Scenario Presets** | 1-click presets for immediate hackathon demonstration (e.g., Morning Commute Gridlock, Major Expressway Accident). | `Streamlit` |

---

## 🛠️ Technology Stack

- **Language:** Python 3.12
- **Dashboard & UI:** [Streamlit](https://streamlit.io/) (Dark Command-Center Theme, Custom CSS)
- **Data Engineering:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (Random Forest Classifier, Model Evaluation)
- **Data Visualization:** [Plotly](https://plotly.com/python/) (Interactive Scatter/Line Networks, Heatmaps, Bar Charts)
- **Graph Algorithms:** [NetworkX](https://networkx.org/) (Dijkstra Shortest Path, Dynamic Edge Cost Functions)

---

## 📁 Repository Structure

```text
TrafficMind-AI/
├── .gitignore           # Excludes virtual environments, caches, OS files, and secrets
├── requirements.txt     # Python package dependencies
├── app.py               # Complete Streamlit application with ML engine and network algorithms
└── README.md            # Comprehensive documentation, setup guide, and hackathon details
```

---

## ⚙️ Installation & Setup (Windows PowerShell)

Follow these exact steps in Windows PowerShell to set up and launch TrafficMind AI:

### 1. Open Windows PowerShell and navigate to the project directory:
```powershell
cd c:\Users\sreej\OneDrive\Documents\promptwars\TrafficMind-AI
```

### 2. (Optional but Recommended) Create and activate a clean virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> *Note: If PowerShell execution policy restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`*

### 3. Install required Python packages:
```powershell
pip install -r requirements.txt
```

### 4. Run the TrafficMind AI application:
```powershell
python -m streamlit run app.py
```

Streamlit will automatically launch the application in your default web browser at `http://localhost:8501`.

---

## 🎮 Step-by-Step Hackathon Demo Walkthrough

To showcase the application to judges in under 3 minutes:

1. **Load a Preset Scenario**:
   - In the left sidebar under **⚡ Quick Scenario Presets**, select:  
     `Preset 1: Morning Commute Gridlock (8:30 AM, High Volume, Rain)`.
   - Observe how the dashboard immediately updates the **Predicted Congestion** card to **🔴 Severe**, shows confidence breakdown, and provides explainability notes on volume squeeze and weather degradation.
2. **Inspect the Short-Term Forecast**:
   - Scroll down in the **🚦 Congestion Predictor & Simulator** tab to see the 12-hour illustrative projection showing the morning rush peak and post-commute dissipation.
3. **Explore the Dynamic Network Map**:
   - Switch to the **🗺️ Dynamic City Network Map** tab.
   - Hover over corridors to inspect real-time live speed, delay, and congestion ratings across all 20 urban arterials.
4. **Demonstrate AI Congestion Rerouting**:
   - Switch to the **🧭 AI Alternative Route Planner** tab.
   - Select Origin: `West End Industrial Park` and Destination: `International Airport Link`.
   - Click **🚀 Re-calculate Optimal Path**.
   - Review the comparison: notice how the AI route avoids congested downtown arterials, routing along less congested bypasses to save significant travel minutes.
5. **Demonstrate Emergency Green-Corridor**:
   - Switch to the **🚑 Emergency Green-Corridor** tab.
   - Select `🚑 Critical Trauma Ambulance` departing from `North Harbor Crossing` to `City General Trauma Hospital`.
   - Observe the **Golden Hour Minutes Saved** metric and the neon emerald green corridor with automated signal preemption.
6. **Verify Model Rigor & Transparency**:
   - Switch to the **📊 Model Evaluation & Transparency** tab.
   - Walk through the Confusion Matrix heatmap, balanced F1-scores, and Gini Feature Importances.

---

## 🔬 Limitations & Synthetic Data Transparency

In accordance with responsible AI practices and hackathon integrity guidelines:

- **Synthetic Telemetry Disclaimer**: Real-time municipal induction-loop datasets and commercial vehicle GPS probe streams are typically proprietary, geo-restricted, or anonymized behind strict NDAs. To guarantee **100% reproducible execution and offline self-containment**, TrafficMind AI generates a physics-inspired synthetic dataset based on the **Greenshields fundamental traffic diagram** and **Bureau of Public Roads (BPR) delay equations**.
- **Model Accuracy Interpretation**: The test set accuracy (~95%+) reflects classification fidelity against this synthetic benchmark distribution. These metrics demonstrate algorithmic correctness and architectural feasibility, but must not be presented as validated real-world municipal operational results without on-site sensor calibration.
- **Future Field Calibration**: For real-world deployment, the pipeline can be connected via MQTT / Kafka to inductive loop counters, CCTV computer-vision density estimators, and V2X (Vehicle-to-Everything) telemetry streams.

---

## ⚖️ Hackathon Compliance Checklist

- [x] Lightweight repository (well below 10 MB limit, no large binary blobs or uncompressed model weights).
- [x] All work committed to the existing `main` branch.
- [x] Zero hardcoded API keys, secrets, or cloud credentials.
- [x] Comprehensive `.gitignore` excluding `.venv/`, `__pycache__/`, and local environment files.
- [x] Verified launch command: `python -m streamlit run app.py`.
- [x] Fully self-contained without external paid API requirements.

---

## 👨‍💻 Author & Acknowledgments

- **Participant:** Individual Entry (Intermediate Developer)
- **Event:** PromptWars x Creative Community VVITU Hackathon
- **Frameworks:** Streamlit, Scikit-Learn, NetworkX, Plotly, Pandas, NumPy
