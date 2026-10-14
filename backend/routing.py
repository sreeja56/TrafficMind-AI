"""
TrafficMind AI - Backend Routing & Graph Theory Module
Handles metropolitan network graph construction, dynamic traffic friction simulation,
AI congestion-aware route optimization, and emergency green-corridor planning.
"""

import numpy as np
import networkx as nx


def build_metropolitan_network() -> nx.Graph:
    """
    Constructs a representative 12-node Metropolitan Road Network.
    
    Nodes represent strategic urban locations (CBD, Industrial, Airport, Hospital, Fire HQ).
    Edges represent road corridors with realistic distances, speed limits, and base capacities.
    """
    G = nx.Graph()

    # Node positions on an urban grid (x, y)
    nodes_info = {
        'Central Metro CBD': {'pos': (50, 50), 'type': 'Commercial Hub', 'icon': '🏢'},
        'Tech Innovation Corridor': {'pos': (76, 74), 'type': 'Business Park', 'icon': '💻'},
        'East Suburb Interchange': {'pos': (88, 46), 'type': 'Residential Hub', 'icon': '🏡'},
        'North Harbor Crossing': {'pos': (44, 86), 'type': 'Port & Logistics', 'icon': '🚢'},
        'West End Industrial Park': {'pos': (14, 56), 'type': 'Manufacturing', 'icon': '🏭'},
        'International Airport Link': {'pos': (92, 86), 'type': 'Airport Gateway', 'icon': '✈️'},
        'Riverside Expressway': {'pos': (28, 76), 'type': 'Transit Arterial', 'icon': '🌉'},
        'South Gateway Junction': {'pos': (56, 14), 'type': 'Regional Highway', 'icon': '🛣️'},
        'University Campus Circle': {'pos': (24, 26), 'type': 'Education Zone', 'icon': '🎓'},
        'Civic Center & Stadium': {'pos': (62, 38), 'type': 'Entertainment / Civic', 'icon': '🏟️'},
        'City General Trauma Hospital': {'pos': (38, 36), 'type': 'Emergency Medical Center', 'icon': '🏥'},
        'Central Fire & Emergency HQ': {'pos': (34, 58), 'type': 'Emergency Dispatch', 'icon': '🚒'},
    }

    for name, data in nodes_info.items():
        G.add_node(name, pos=data['pos'], type=data['type'], icon=data['icon'])

    # Corridors: (u, v, distance_km, base_speed_limit_kmh, road_name)
    corridors = [
        ('Central Metro CBD', 'Tech Innovation Corridor', 8.5, 65, 'Metro Tech Expressway'),
        ('Central Metro CBD', 'Civic Center & Stadium', 4.2, 50, 'Civic Boulevard'),
        ('Central Metro CBD', 'City General Trauma Hospital', 5.0, 50, 'Healthcare Arterial'),
        ('Central Metro CBD', 'Central Fire & Emergency HQ', 4.5, 50, 'Central Dispatch Way'),
        ('Central Metro CBD', 'North Harbor Crossing', 10.2, 70, 'North Harbor Causeway'),
        ('Tech Innovation Corridor', 'East Suburb Interchange', 7.8, 60, 'Silicon Link'),
        ('Tech Innovation Corridor', 'International Airport Link', 9.4, 90, 'Airport Tech Freeway'),
        ('East Suburb Interchange', 'Civic Center & Stadium', 8.2, 65, 'Eastern Parkway'),
        ('East Suburb Interchange', 'South Gateway Junction', 12.5, 80, 'Southeast Ring Road'),
        ('North Harbor Crossing', 'International Airport Link', 14.2, 85, 'Harbor Skyway'),
        ('North Harbor Crossing', 'Riverside Expressway', 8.0, 75, 'Riverbank Tollway'),
        ('Riverside Expressway', 'Central Fire & Emergency HQ', 6.0, 60, 'Riverhead Access'),
        ('Riverside Expressway', 'West End Industrial Park', 7.5, 65, 'Industrial North Spur'),
        ('West End Industrial Park', 'Central Fire & Emergency HQ', 5.8, 50, 'Freight Corridor'),
        ('West End Industrial Park', 'University Campus Circle', 9.2, 60, 'Westside Ring Road'),
        ('University Campus Circle', 'City General Trauma Hospital', 4.8, 45, 'Campus Medical Link'),
        ('University Campus Circle', 'South Gateway Junction', 11.0, 70, 'Southwest Arterial'),
        ('City General Trauma Hospital', 'Civic Center & Stadium', 5.6, 50, 'Cross-Town Connector'),
        ('City General Trauma Hospital', 'Central Fire & Emergency HQ', 3.5, 45, 'Emergency Response Route'),
        ('Civic Center & Stadium', 'South Gateway Junction', 7.2, 60, 'Southern Grand Boulevard'),
    ]

    for u, v, dist, speed, name in corridors:
        base_time_min = (dist / speed) * 60.0
        G.add_edge(
            u, v,
            distance_km=dist,
            speed_limit=speed,
            road_name=name,
            base_time_min=round(base_time_min, 1)
        )

    return G


def simulate_network_conditions(G: nx.Graph, active_scenario_inputs: dict) -> nx.Graph:
    """
    Applies real-time simulated conditions across network edges based on current
    dashboard inputs (time of day, weather, incidents) and network bottlenecks.
    Calculates dynamic travel time (minutes) and congestion status.
    """
    G_dyn = G.copy()
    hour = active_scenario_inputs['hour']
    weather = active_scenario_inputs['weather']
    incident = active_scenario_inputs['incident']

    # Known high-friction bottlenecks during peak hours
    bottleneck_edges = {
        ('Central Metro CBD', 'Tech Innovation Corridor'),
        ('Central Metro CBD', 'Civic Center & Stadium'),
        ('North Harbor Crossing', 'Riverside Expressway'),
        ('West End Industrial Park', 'Central Fire & Emergency HQ')
    }

    # Rush multiplier based on hour
    if (7 <= hour <= 9) or (16 <= hour <= 19):
        rush_factor = 1.95
    elif (10 <= hour <= 15):
        rush_factor = 1.25
    elif (20 <= hour <= 22):
        rush_factor = 1.10
    else:
        rush_factor = 0.70

    weather_slowdown = {'Clear': 1.0, 'Rain': 1.25, 'Fog': 1.40, 'Storm': 1.75}.get(weather, 1.0)

    # Deterministic pseudo-random seed per hour/weather for consistent map display
    rand_state = np.random.RandomState(int(hour * 100 + len(weather)))

    for u, v, data in G_dyn.edges(data=True):
        dist = data['distance_km']
        speed_lim = data['speed_limit']

        is_bottleneck = (u, v) in bottleneck_edges or (v, u) in bottleneck_edges
        edge_load = rush_factor * (1.4 if is_bottleneck else 0.95) * rand_state.uniform(0.85, 1.2)

        # Apply incident penalty if incident is active
        incident_penalty = 1.0
        if incident == 'Major Accident' and is_bottleneck:
            incident_penalty = 3.6
        elif incident in ['Minor Accident', 'Vehicle Breakdown'] and is_bottleneck:
            incident_penalty = 1.9
        elif incident == 'Road Work':
            incident_penalty = 1.4

        combined_multiplier = max(1.0, edge_load * weather_slowdown * incident_penalty)

        effective_speed = max(8.0, speed_lim / combined_multiplier)
        dynamic_time_min = (dist / effective_speed) * 60.0

        if combined_multiplier < 1.35:
            congestion_state = 'Low'
            color = '#00E676'  # Green
        elif combined_multiplier < 2.20:
            congestion_state = 'Moderate'
            color = '#FFB300'  # Amber
        else:
            congestion_state = 'Severe'
            color = '#FF1744'  # Red

        G_dyn[u][v]['effective_speed'] = round(effective_speed, 1)
        G_dyn[u][v]['dynamic_time_min'] = round(dynamic_time_min, 1)
        G_dyn[u][v]['congestion_state'] = congestion_state
        G_dyn[u][v]['color'] = color
        G_dyn[u][v]['congestion_multiplier'] = round(combined_multiplier, 2)

    return G_dyn


def compute_route_comparison(G_dyn: nx.Graph, origin: str, destination: str):
    """
    Computes and compares:
    1. Baseline Shortest Path: Dijkstra based strictly on distance (ignoring congestion).
    2. TrafficMind AI Dynamic Route: Dijkstra based on dynamic travel time under live congestion.
    """
    if origin == destination:
        return None

    # Baseline distance-only route
    try:
        dist_path = nx.shortest_path(G_dyn, source=origin, target=destination, weight='distance_km')
    except nx.NetworkXNoPath:
        return None

    dist_km_sum = 0.0
    dist_time_sum = 0.0
    dist_bottlenecks = 0

    for i in range(len(dist_path) - 1):
        u, v = dist_path[i], dist_path[i + 1]
        edge = G_dyn[u][v]
        dist_km_sum += edge['distance_km']
        dist_time_sum += edge['dynamic_time_min']
        if edge['congestion_state'] == 'Severe':
            dist_bottlenecks += 1

    # AI Dynamic Cost Route
    ai_path = nx.shortest_path(G_dyn, source=origin, target=destination, weight='dynamic_time_min')
    ai_km_sum = 0.0
    ai_time_sum = 0.0
    ai_bottlenecks = 0

    for i in range(len(ai_path) - 1):
        u, v = ai_path[i], ai_path[i + 1]
        edge = G_dyn[u][v]
        ai_km_sum += edge['distance_km']
        ai_time_sum += edge['dynamic_time_min']
        if edge['congestion_state'] == 'Severe':
            ai_bottlenecks += 1

    time_saved = max(0.0, dist_time_sum - ai_time_sum)

    return {
        'baseline_path': dist_path,
        'baseline_dist': round(dist_km_sum, 1),
        'baseline_time': round(dist_time_sum, 1),
        'baseline_bottlenecks': dist_bottlenecks,
        'ai_path': ai_path,
        'ai_dist': round(ai_km_sum, 1),
        'ai_time': round(ai_time_sum, 1),
        'ai_bottlenecks': ai_bottlenecks,
        'time_saved_min': round(time_saved, 1),
        'pct_time_saved': round((time_saved / max(0.1, dist_time_sum)) * 100, 1),
        'is_rerouted': dist_path != ai_path
    }


def compute_emergency_green_corridor(G_dyn: nx.Graph, origin: str, hospital_destination: str = 'City General Trauma Hospital'):
    """
    Simulates Green Corridor Traffic Signal Preemption & Priority Lane Clearance.
    
    Emergency vehicles benefit from:
    - Dedicated lane clearance (delay friction reduced by 65-80%).
    - Automatic green-light wave signal preemption.
    - Path selection prioritizing high-capacity expressways that can be cleared safely.
    """
    if origin == hospital_destination:
        return None

    # Construct weighted emergency graph with preemption factor
    G_emerg = G_dyn.copy()
    for u, v, data in G_emerg.edges(data=True):
        # Emergency clearance speed is significantly higher because cross-traffic is halted
        raw_dyn_time = data['dynamic_time_min']
        clearance_factor = 0.40 if data['congestion_state'] == 'Severe' else 0.30
        emergency_travel_time = max(data['base_time_min'] * 0.75, raw_dyn_time * clearance_factor)
        G_emerg[u][v]['emergency_time_min'] = emergency_travel_time

    # Standard civilian route to hospital under current traffic
    civilian_path = nx.shortest_path(G_dyn, source=origin, target=hospital_destination, weight='dynamic_time_min')
    civilian_time = sum(G_dyn[civilian_path[i]][civilian_path[i + 1]]['dynamic_time_min'] for i in range(len(civilian_path) - 1))

    # Priority emergency green-corridor route
    emergency_path = nx.shortest_path(G_emerg, source=origin, target=hospital_destination, weight='emergency_time_min')
    emergency_time = sum(G_emerg[emergency_path[i]][emergency_path[i + 1]]['emergency_time_min'] for i in range(len(emergency_path) - 1))
    emergency_dist = sum(G_emerg[emergency_path[i]][emergency_path[i + 1]]['distance_km'] for i in range(len(emergency_path) - 1))

    time_saved = max(0.0, civilian_time - emergency_time)
    signals_preempted = len(emergency_path) - 1

    return {
        'emergency_path': emergency_path,
        'emergency_dist': round(emergency_dist, 1),
        'emergency_time': round(emergency_time, 1),
        'civilian_time': round(civilian_time, 1),
        'time_saved_min': round(time_saved, 1),
        'signals_preempted': signals_preempted,
        'golden_hour_benefit': "High - Rapid Trauma Intervention Possible" if emergency_time < 12.0 else "Critical Corridor Monitored"
    }
