"""
TrafficMind AI - Backend Machine Learning & Predictor Module
Handles synthetic data generation, model training, feature encoding,
and congestion classification inference.
"""

import math
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


def generate_synthetic_traffic_dataset(num_samples: int = 3200, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, physics-inspired synthetic traffic dataset.
    
    Principles:
    - BPR (Bureau of Public Roads) delay formulation and Greenshields fundamental traffic diagram:
      Speed drops non-linearly as Volume / Capacity ratio increases.
    - Commute Rush Hours: Distinct morning peak (07:30 - 09:30) and evening peak (16:30 - 19:00).
    - Incidents & Weather: Add friction penalties reducing road capacity.
    
    Classes:
    - Low Congestion: Free flow or minor deceleration (Volume/Capacity < 0.55).
    - Moderate Congestion: Stop-and-go queueing (0.55 <= Volume/Capacity < 0.85).
    - Severe Congestion: Gridlock or major breakdown (Volume/Capacity >= 0.85 or Major Incident).
    """
    np.random.seed(seed)
    random.seed(seed)

    hours = np.random.randint(0, 24, size=num_samples)
    days_of_week = np.random.randint(0, 7, size=num_samples)  # 0=Mon, 6=Sun
    is_weekend = np.isin(days_of_week, [5, 6]).astype(int)

    weathers = np.random.choice(
        ['Clear', 'Rain', 'Fog', 'Storm'],
        size=num_samples,
        p=[0.55, 0.25, 0.12, 0.08]
    )

    incidents = np.random.choice(
        ['None', 'Minor Accident', 'Road Work', 'Vehicle Breakdown', 'Major Accident'],
        size=num_samples,
        p=[0.70, 0.12, 0.08, 0.07, 0.03]
    )

    road_types = np.random.choice(
        ['Highway Expressway', 'Urban Arterial', 'Downtown Grid', 'Harbor Bridge'],
        size=num_samples,
        p=[0.35, 0.35, 0.20, 0.10]
    )

    # Road baseline capacity (vehicles/hour/lane) and free-flow speed (km/h)
    capacity_map = {
        'Highway Expressway': 2200,
        'Urban Arterial': 1600,
        'Downtown Grid': 1100,
        'Harbor Bridge': 1700
    }
    speed_free_map = {
        'Highway Expressway': 105,
        'Urban Arterial': 65,
        'Downtown Grid': 45,
        'Harbor Bridge': 80
    }

    base_capacities = np.array([capacity_map[rt] for rt in road_types], dtype=float)
    base_free_speeds = np.array([speed_free_map[rt] for rt in road_types], dtype=float)

    # Commute demand multiplier based on hour of day
    rush_multiplier = np.ones(num_samples, dtype=float)
    for i, h in enumerate(hours):
        if is_weekend[i]:
            # Weekend peak between 11:00 and 18:00
            rush_multiplier[i] = 1.0 + 0.35 * np.exp(-0.5 * ((h - 14) / 3.0) ** 2)
        else:
            # Morning rush (8:00) and evening rush (17:30)
            morning = 0.85 * np.exp(-0.5 * ((h - 8.2) / 1.5) ** 2)
            evening = 0.95 * np.exp(-0.5 * ((h - 17.5) / 1.8) ** 2)
            rush_multiplier[i] = 0.5 + morning + evening

    # Effective vehicle count with stochastic variation
    raw_volume = base_capacities * rush_multiplier * np.random.uniform(0.35, 1.15, size=num_samples)
    vehicle_counts = np.clip(np.round(raw_volume).astype(int), 120, 2600)

    # Capacity reduction factors for weather and incidents
    weather_cap_factor = np.array([{
        'Clear': 1.0,
        'Rain': 0.82,
        'Fog': 0.75,
        'Storm': 0.60
    }[w] for w in weathers])

    incident_cap_factor = np.array([{
        'None': 1.0,
        'Minor Accident': 0.72,
        'Road Work': 0.80,
        'Vehicle Breakdown': 0.70,
        'Major Accident': 0.35
    }[inc] for inc in incidents])

    effective_capacity = base_capacities * weather_cap_factor * incident_cap_factor
    v_c_ratio = vehicle_counts / effective_capacity

    # Greenshields & BPR speed degradation curve with observation noise
    # S = S_free / (1 + alpha * (V/C)^beta)
    alpha, beta = 0.20, 3.2
    model_speed = base_free_speeds / (1.0 + alpha * np.power(v_c_ratio, beta))
    speed_noise = np.random.normal(0, 3.5, size=num_samples)
    avg_speeds = np.clip(np.round(model_speed + speed_noise, 1), 6.0, 120.0)

    # Compute continuous congestion score (0 to 100)
    # Higher score = worse congestion
    congestion_score = (
        (v_c_ratio * 42.0) +
        ((1.0 - (avg_speeds / base_free_speeds)) * 38.0) +
        ((1.0 - weather_cap_factor) * 12.0) +
        ((1.0 - incident_cap_factor) * 25.0) +
        np.random.normal(0, 2.0, size=num_samples)
    )

    # Discretize into 3 classes: Low, Moderate, Severe
    labels = []
    for score in congestion_score:
        if score < 42.0:
            labels.append('Low')
        elif score < 72.0:
            labels.append('Moderate')
        else:
            labels.append('Severe')

    df = pd.DataFrame({
        'hour': hours,
        'day_of_week': days_of_week,
        'is_weekend': is_weekend,
        'road_type': road_types,
        'vehicle_count': vehicle_counts,
        'avg_speed': avg_speeds,
        'weather': weathers,
        'incident': incidents,
        'v_c_ratio': np.round(v_c_ratio, 2),
        'congestion_level': labels
    })

    return df


def train_traffic_model(df: pd.DataFrame):
    """
    Prepares features, trains a Random Forest Classifier on the synthetic dataset,
    and returns model artifacts alongside held-out test evaluation metrics.
    """
    feature_cols = [
        'hour', 'is_weekend', 'vehicle_count', 'avg_speed',
        'weather', 'incident', 'road_type'
    ]
    X_raw = df[feature_cols].copy()
    y = df['congestion_level'].copy()

    # One-hot encode categorical features with known dummy columns
    X_encoded = pd.get_dummies(
        X_raw,
        columns=['weather', 'incident', 'road_type'],
        drop_first=False
    )

    feature_names = list(X_encoded.columns)

    # 80/20 train/test split with stratification for balanced representation
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, clf.predict(X_train))

    class_names = ['Low', 'Moderate', 'Severe']
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=class_names, average=None
    )

    metrics = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'class_names': class_names,
        'confusion_matrix': cm,
        'precision': {c: p for c, p in zip(class_names, precision)},
        'recall': {c: r for c, r in zip(class_names, recall)},
        'f1': {c: f for c, f in zip(class_names, f1)},
        'classification_report_str': classification_report(y_test, y_pred, labels=class_names),
        'feature_importances': pd.DataFrame({
            'feature': feature_names,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
    }

    return clf, feature_names, metrics


def encode_single_input(hour, is_weekend, vehicle_count, avg_speed, weather, incident, road_type, feature_names):
    """Encodes a single user input row to match the Random Forest trained feature schema."""
    row_dict = {col: 0 for col in feature_names}
    row_dict['hour'] = hour
    row_dict['is_weekend'] = int(is_weekend)
    row_dict['vehicle_count'] = vehicle_count
    row_dict['avg_speed'] = avg_speed

    w_col = f"weather_{weather}"
    if w_col in row_dict:
        row_dict[w_col] = 1

    inc_col = f"incident_{incident}"
    if inc_col in row_dict:
        row_dict[inc_col] = 1

    rt_col = f"road_type_{road_type}"
    if rt_col in row_dict:
        row_dict[rt_col] = 1

    return pd.DataFrame([row_dict], columns=feature_names)


def predict_congestion(model, inputs):
    """
    Runs model inference on an encoded input vector.
    Returns predicted class label, confidence percentage, and class-wise probabilities.
    """
    prediction = model.predict(inputs)[0]
    probabilities = model.predict_proba(inputs)[0]
    classes = list(model.classes_)
    prob_map = {c: float(p) for c, p in zip(classes, probabilities)}

    return {
        "prediction": prediction,
        "confidence": float(max(probabilities)) * 100.0,
        "probabilities": probabilities,
        "prob_map": prob_map,
        "classes": classes
    }
