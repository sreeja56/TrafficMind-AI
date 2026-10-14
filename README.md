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
├── vercel.json              # Vercel deployment config (API rewrites to Python serverless)
├── package.json             # Next.js & React dependencies
├── next.config.mjs          # Next.js configuration with development API rewrites
├── tailwind.config.ts       # Tailwind CSS theme configuration
├── postcss.config.mjs       # PostCSS configuration
├── tsconfig.json            # TypeScript configuration
├── requirements.txt         # Python dependencies for Vercel Serverless Function
├── app.py                   # Root FastAPI entrypoint (resolves Vercel top-level export)
├── app.py.bak               # Original backup preserved
├── api/
│   └── index.py             # FastAPI Serverless backend for Vercel
├── src/
│   └── app/
│       ├── layout.tsx       # Root Next.js layout & metadata
│       ├── globals.css      # Cyber dark theme & glassmorphism utilities
│       └── page.tsx         # Full TrafficMind AI interactive dashboard
├── backend/
│   ├── predictor.py         # ML dataset generator, Random Forest training & inference
│   └── routing.py           # NetworkX graph topology, dynamic cost & green-corridor engine
├── frontend/
│   └── app.py               # Original Streamlit dashboard (preserved)
└── README.md                # Comprehensive documentation, setup guide, and hackathon details
```

---

## ☁️ Deploying on Vercel (Step-by-Step)

The project is now pre-configured for seamless 1-click deployment on Vercel:

1. **Commit and push your changes to GitHub:**
   ```powershell
   git add .
   git commit -m "feat: Vercel deployment with Next.js frontend and FastAPI serverless backend"
   git push origin main
   ```

2. **Deploy on Vercel:**
   - Log into [Vercel](https://vercel.com) and click **"Add New Project"**.
   - Import `sreeja56/TrafficMind-AI`.
   - Vercel will automatically detect **Next.js** as the framework.
   - Leave the default settings (**Build Command:** `next build`, **Output Directory:** `.next`).
   - Click **Deploy**!

Vercel will build the Next.js frontend and automatically compile `api/index.py` into a high-performance Python Serverless Function.

---

## ⚙️ Running Locally (Windows PowerShell)

To test both the Next.js frontend and FastAPI backend locally:

### 1. Terminal 1: Run the Next.js Frontend
```powershell
cd c:\Users\sreej\OneDrive\Documents\promptwars\TrafficMind-AI
npm install
npm run dev
```
Open **`http://localhost:3000`** in your browser.

### 2. (Optional) Terminal 2: Run the FastAPI Python Backend locally
```powershell
cd c:\Users\sreej\OneDrive\Documents\promptwars\TrafficMind-AI
.\.venv\Scripts\Activate.ps1
python app.py
```
FastAPI will start on `http://localhost:8000`. Next.js proxies all `/api/*` requests to this backend during development!

*(To run the original Streamlit interface locally instead: `python -m streamlit run frontend/app.py`)*

---

## 🎮 Step-by-Step Hackathon Demo Walkthrough

To showcase the application to judges in under 3 minutes:

1. **Load a Preset Scenario**:
   - In the left sidebar under **⚡ Quick Scenario Presets**, select:  
     `Preset 1: Morning Commute Gridlock (8:30 AM, High Volume, Rain)`.
   - Observe how the dashboard immediately updates the **Predicted Congestion** card to **🔴 Severe**, shows confidence breakdown, and provides explainability notes on volume squeeze and weather degradation.
2. **Inspect the Short-Term Forecast**:
   - Scroll down in the **🚦 Predictor & Simulator** tab to see the 12-hour illustrative projection showing the morning rush peak and post-commute dissipation.
3. **Explore the Dynamic Network Map**:
   - Switch to the **🗺️ Dynamic City Map** tab.
   - Hover over corridors and nodes to inspect real-time live speed, delay, and congestion ratings across all 20 urban arterials.
4. **Demonstrate AI Congestion Rerouting**:
   - Switch to the **🧭 AI Alternative Route** tab.
   - Select Origin: `West End Industrial Park` and Destination: `International Airport Link`.
   - Review the comparison: notice how the AI route avoids congested downtown arterials, routing along less congested bypasses to save significant travel minutes.
5. **Demonstrate Emergency Green-Corridor**:
   - Switch to the **🚑 Emergency Corridor** tab.
   - Select `🚑 Critical Trauma Ambulance` departing from `North Harbor Crossing` to `City General Trauma Hospital`.
   - Observe the **Golden Hour Minutes Saved** metric and the neon emerald green corridor with automated signal preemption.
6. **Verify Model Rigor & Transparency**:
   - Switch to the **📊 Model Evaluation** tab.
   - Walk through the Confusion Matrix heatmap, balanced F1-scores, and Gini Feature Importances.

---

## 🔬 Limitations & Synthetic Data Transparency

In accordance with responsible AI practices and hackathon integrity guidelines:

- **Synthetic Telemetry Disclaimer**: Real-time municipal induction-loop datasets and commercial vehicle GPS probe streams are typically proprietary, geo-restricted, or anonymized behind strict NDAs. To guarantee **100% reproducible execution and offline self-containment**, TrafficMind AI generates a physics-inspired synthetic dataset based on the **Greenshields fundamental traffic diagram** and **Bureau of Public Roads (BPR) delay equations**.
- **Model Accuracy Interpretation**: The test set accuracy (~94.3%) reflects classification fidelity against this synthetic benchmark distribution. These metrics demonstrate algorithmic correctness and architectural feasibility, but must not be presented as validated real-world municipal operational results without on-site sensor calibration.
- **Future Field Calibration**: For real-world deployment, the pipeline can be connected via MQTT / Kafka to inductive loop counters, CCTV computer-vision density estimators, and V2X (Vehicle-to-Everything) telemetry streams.

---

## ⚖️ Hackathon Compliance Checklist

- [x] Lightweight repository (well below 10 MB limit, excludes `.venv` and `node_modules`).
- [x] Zero hardcoded API keys, secrets, or cloud credentials.
- [x] Comprehensive `.gitignore` excluding `.venv/`, `node_modules/`, `.next/`, and local caches.
- [x] Vercel-ready with Next.js App Router and Python FastAPI Serverless backend.
- [x] Fully self-contained without external paid API requirements.

---

## 👨‍💻 Author & Acknowledgments

- **Participant:** Individual Entry (Intermediate Developer)
- **Event:** PromptWars x Creative Community VVITU Hackathon
- **Frameworks:** Streamlit, Scikit-Learn, NetworkX, Plotly, Pandas, NumPy
