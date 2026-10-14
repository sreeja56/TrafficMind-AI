"""
TrafficMind AI - FastAPI Serverless Backend for Vercel
Serves predictive machine learning inference, network telemetry,
AI route optimization, and emergency green-corridor planning.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure repository root is on Python sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.predictor import (
    generate_synthetic_traffic_dataset,
    train_traffic_model,
    encode_single_input,
    predict_congestion
)
from backend.routing import (
    build_metropolitan_network,
    simulate_network_conditions,
    compute_route_comparison,
    compute_emergency_green_corridor
)

# Initialize FastAPI app
app = FastAPI(
    title="TrafficMind AI API",
    description="Intelligent Urban Traffic Congestion Prediction & Dynamic Rerouting Engine",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for model and graph
_MODEL_CACHE: Dict[str, Any] = {}
_NETWORK_GRAPH = None


def get_model_and_network():
    """Lazily loads and caches the ML model and base network topology."""
    global _MODEL_CACHE, _NETWORK_GRAPH
    if "model" not in _MODEL_CACHE:
        df = generate_synthetic_traffic_dataset(num_samples=2200, seed=42)
        clf, feature_names, metrics = train_traffic_model(df)
        _MODEL_CACHE["model"] = clf
        _MODEL_CACHE["feature_names"] = feature_names
        _MODEL_CACHE["metrics"] = metrics
        _MODEL_CACHE["sample_count"] = len(df)

    if _NETWORK_GRAPH is None:
        _NETWORK_GRAPH = build_metropolitan_network()

    return (
        _MODEL_CACHE["model"],
        _MODEL_CACHE["feature_names"],
        _MODEL_CACHE["metrics"],
        _NETWORK_GRAPH
    )


# --- Request & Response Schemas ---

class ScenarioInput(BaseModel):
    hour: int = Field(8, ge=0, le=23)
    day_of_week: str = "Monday"
    is_weekend: bool = False
    road_type: str = "Highway Expressway"
    vehicle_count: int = Field(1850, ge=50, le=3000)
    avg_speed: float = Field(34.0, ge=1.0, le=140.0)
    weather: str = "Rain"
    incident: str = "Minor Accident"


class RouteRequest(BaseModel):
    origin: str
    destination: str
    scenario: ScenarioInput


class EmergencyRequest(BaseModel):
    origin: str
    destination: str = "City General Trauma Hospital"
    vehicle_type: str = "🚑 Critical Trauma Ambulance"
    scenario: ScenarioInput


# --- API Endpoints ---

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TrafficMind AI Engine",
        "timestamp": datetime.now().isoformat(),
        "runtime": "Vercel Python Serverless"
    }


@app.get("/api/network")
def get_network():
    """Returns static metropolitan nodes and corridors information."""
    _, _, _, G = get_model_and_network()
    
    nodes_data = []
    for node, data in G.nodes(data=True):
        nodes_data.append({
            "id": node,
            "x": data["pos"][0],
            "y": data["pos"][1],
            "type": data.get("type", "Urban Hub"),
            "icon": data.get("icon", "📍")
        })

    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "from": u,
            "to": v,
            "road_name": data["road_name"],
            "distance_km": data["distance_km"],
            "speed_limit": data["speed_limit"],
            "base_time_min": data["base_time_min"]
        })

    return {
        "nodes": nodes_data,
        "edges": edges_data
    }


@app.post("/api/predict")
def predict_traffic(input_data: ScenarioInput):
    """
    Performs real-time congestion classification, feature explanations,
    12-hour illustrative forecast projection, and dynamic network state calculation.
    """
    clf, feature_names, _, G_base = get_model_and_network()

    # Encode single input
    X_user = encode_single_input(
        hour=input_data.hour,
        is_weekend=input_data.is_weekend,
        vehicle_count=input_data.vehicle_count,
        avg_speed=input_data.avg_speed,
        weather=input_data.weather,
        incident=input_data.incident,
        road_type=input_data.road_type,
        feature_names=feature_names
    )

    pred_res = predict_congestion(clf, X_user)
    pred_class = pred_res["prediction"]
    confidence = pred_res["confidence"]
    prob_map = pred_res["prob_map"]

    # Generate Explainability Drivers
    explanations = []
    if input_data.vehicle_count > 1800:
        explanations.append(f"High Volume Squeeze: Influx of {input_data.vehicle_count} veh/hr approaches corridor design capacity.")
    elif input_data.vehicle_count < 800:
        explanations.append(f"Light Traffic Demand: Low volume ({input_data.vehicle_count} veh/hr) leaves surplus road capacity.")

    if input_data.avg_speed < 30:
        explanations.append(f"Severe Velocity Drop: Speed degraded to {input_data.avg_speed} km/h, signaling bottleneck queueing.")
    elif input_data.avg_speed > 60:
        explanations.append(f"Healthy Flow Velocity: Velocity ({input_data.avg_speed} km/h) indicates free flow.")

    if input_data.weather != "Clear":
        explanations.append(f"Weather Factor ({input_data.weather}): Elevated pavement friction & reduced headway buffer.")

    if input_data.incident != "None":
        explanations.append(f"Incident Impact ({input_data.incident}): Lane closure restriction creating physical constriction.")

    if not explanations:
        explanations.append("Parameters operating within balanced nominal tolerances.")

    # 12-Hour Illustrative Forecast
    now = datetime.now()
    forecast_points = []
    for step in range(13):
        h = (input_data.hour + step) % 24
        t_label = (now + timedelta(hours=step)).strftime("%I:%M %p")

        if 7 <= h <= 9:
            rush = 42.0 + (9 - abs(h - 8.2)) * 6.0
        elif 16 <= h <= 19:
            rush = 48.0 + (10 - abs(h - 17.5)) * 5.5
        elif 10 <= h <= 15:
            rush = 24.0
        else:
            rush = 8.0

        if step == 0:
            c_idx = 25.0 if pred_class == "Low" else (55.0 if pred_class == "Moderate" else 85.0)
            spd = input_data.avg_speed
        else:
            c_idx = max(5.0, min(95.0, rush))
            spd = max(12.0, min(105.0, 85.0 - (c_idx * 0.75)))

        forecast_points.append({
            "step": step,
            "hour": h,
            "time_label": t_label,
            "congestion_index": round(c_idx, 1),
            "projected_speed": round(spd, 1)
        })

    # Dynamic Network Simulation
    scenario_dict = input_data.model_dump()
    G_dyn = simulate_network_conditions(G_base, scenario_dict)

    corridors = []
    speeds = []
    severe_count = 0
    moderate_count = 0
    low_count = 0

    for u, v, d in G_dyn.edges(data=True):
        c_state = d["congestion_state"]
        speeds.append(d["effective_speed"])
        if c_state == "Severe":
            severe_count += 1
        elif c_state == "Moderate":
            moderate_count += 1
        else:
            low_count += 1

        corridors.append({
            "from": u,
            "to": v,
            "road_name": d["road_name"],
            "distance_km": d["distance_km"],
            "speed_limit": d["speed_limit"],
            "effective_speed": d["effective_speed"],
            "dynamic_time_min": d["dynamic_time_min"],
            "congestion_state": c_state,
            "color": d["color"],
            "congestion_multiplier": d["congestion_multiplier"]
        })

    return {
        "prediction": pred_class,
        "confidence": round(confidence, 1),
        "probabilities": {k: round(v * 100, 1) for k, v in prob_map.items()},
        "explanations": explanations,
        "forecast_12h": forecast_points,
        "network_corridors": corridors,
        "network_summary": {
            "avg_speed": round(float(sum(speeds) / len(speeds)), 1) if speeds else 0.0,
            "severe_corridors": severe_count,
            "moderate_corridors": moderate_count,
            "low_corridors": low_count,
            "total_corridors": len(corridors)
        }
    }


@app.post("/api/route")
def plan_route(req: RouteRequest):
    """
    Computes comparative shortest path (baseline distance vs. AI dynamic travel time).
    """
    _, _, _, G_base = get_model_and_network()
    scenario_dict = req.scenario.model_dump()
    G_dyn = simulate_network_conditions(G_base, scenario_dict)

    res = compute_route_comparison(G_dyn, req.origin, req.destination)
    if res is None:
        raise HTTPException(status_code=400, detail="Invalid origin or destination.")

    return res


@app.post("/api/emergency")
def plan_emergency_route(req: EmergencyRequest):
    """
    Computes priority green corridor clearance with signal preemption and golden-hour savings.
    """
    _, _, _, G_base = get_model_and_network()
    scenario_dict = req.scenario.model_dump()
    G_dyn = simulate_network_conditions(G_base, scenario_dict)

    res = compute_emergency_green_corridor(G_dyn, req.origin, req.destination)
    if res is None:
        raise HTTPException(status_code=400, detail="Emergency origin and receiving facility cannot be identical.")

    return res


@app.get("/api/evaluation")
def get_evaluation():
    """
    Returns held-out test evaluation metrics and synthetic data disclosure.
    """
    _, _, metrics, _ = get_model_and_network()

    return {
        "test_accuracy": round(metrics["test_acc"] * 100, 2),
        "train_accuracy": round(metrics["train_acc"] * 100, 2),
        "macro_f1": round(float(sum(metrics["f1"].values()) / len(metrics["f1"])), 3),
        "classes": metrics["class_names"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "precision": {k: round(v, 3) for k, v in metrics["precision"].items()},
        "recall": {k: round(v, 3) for k, v in metrics["recall"].items()},
        "f1": {k: round(v, 3) for k, v in metrics["f1"].items()},
        "feature_importances": metrics["feature_importances"].to_dict(orient="records"),
        "synthetic_data_disclosure": (
            "Model is evaluated on a synthetic benchmark dataset generated using Greenshields & BPR "
            "macroscopic traffic equations. Metrics demonstrate architectural fidelity and must be "
            "calibrated with real induction loop telemetry for municipal field deployment."
        )
    }
