"""
Maritime Ports, Choke Points, and Major Shipping Routes Catalog.
Provides geographic coordinates, baseline transit metrics, and operational capacities.
"""

from typing import Dict, List, Any

# Top Global Maritime Ports
PORTS_CATALOG: Dict[str, Dict[str, Any]] = {
    "CNSHA": {
        "name": "Port of Shanghai",
        "city": "Shanghai",
        "country": "China",
        "region": "East Asia",
        "lat": 31.2304,
        "lon": 121.4737,
        "baseline_dwell_days": 2.8,
        "annual_teu_capacity_millions": 47.3,
        "berths": 125
    },
    "SGSIN": {
        "name": "Port of Singapore",
        "city": "Singapore",
        "country": "Singapore",
        "region": "Southeast Asia",
        "lat": 1.3521,
        "lon": 103.8198,
        "baseline_dwell_days": 2.2,
        "annual_teu_capacity_millions": 37.5,
        "berths": 67
    },
    "CNNGB": {
        "name": "Port of Ningbo-Zhoushan",
        "city": "Ningbo",
        "country": "China",
        "region": "East Asia",
        "lat": 29.8683,
        "lon": 121.5440,
        "baseline_dwell_days": 3.1,
        "annual_teu_capacity_millions": 33.3,
        "berths": 88
    },
    "CNSZX": {
        "name": "Port of Shenzhen (Yantian)",
        "city": "Shenzhen",
        "country": "China",
        "region": "East Asia",
        "lat": 22.5431,
        "lon": 114.0579,
        "baseline_dwell_days": 2.9,
        "annual_teu_capacity_millions": 28.7,
        "berths": 72
    },
    "KRPUS": {
        "name": "Port of Busan",
        "city": "Busan",
        "country": "South Korea",
        "region": "East Asia",
        "lat": 35.1796,
        "lon": 129.0756,
        "baseline_dwell_days": 2.5,
        "annual_teu_capacity_millions": 22.7,
        "berths": 58
    },
    "NLRTM": {
        "name": "Port of Rotterdam",
        "city": "Rotterdam",
        "country": "Netherlands",
        "region": "North Europe",
        "lat": 51.9244,
        "lon": 4.4777,
        "baseline_dwell_days": 3.5,
        "annual_teu_capacity_millions": 14.5,
        "berths": 92
    },
    "BEANR": {
        "name": "Port of Antwerp-Bruges",
        "city": "Antwerp",
        "country": "Belgium",
        "region": "North Europe",
        "lat": 51.2194,
        "lon": 4.4025,
        "baseline_dwell_days": 3.8,
        "annual_teu_capacity_millions": 12.5,
        "berths": 64
    },
    "DEHAM": {
        "name": "Port of Hamburg",
        "city": "Hamburg",
        "country": "Germany",
        "region": "North Europe",
        "lat": 53.5511,
        "lon": 9.9937,
        "baseline_dwell_days": 4.1,
        "annual_teu_capacity_millions": 8.7,
        "berths": 48
    },
    "USLAX": {
        "name": "Port of Los Angeles",
        "city": "Los Angeles",
        "country": "United States",
        "region": "US West Coast",
        "lat": 33.7432,
        "lon": -118.2673,
        "baseline_dwell_days": 4.8,
        "annual_teu_capacity_millions": 10.6,
        "berths": 43
    },
    "USLGB": {
        "name": "Port of Long Beach",
        "city": "Long Beach",
        "country": "United States",
        "region": "US West Coast",
        "lat": 33.7701,
        "lon": -118.1937,
        "baseline_dwell_days": 4.5,
        "annual_teu_capacity_millions": 9.1,
        "berths": 38
    },
    "USNYC": {
        "name": "Port of New York and New Jersey",
        "city": "New York",
        "country": "United States",
        "region": "US East Coast",
        "lat": 40.6840,
        "lon": -74.1485,
        "baseline_dwell_days": 4.2,
        "annual_teu_capacity_millions": 8.9,
        "berths": 41
    },
    "USHOU": {
        "name": "Port of Houston",
        "city": "Houston",
        "country": "United States",
        "region": "US Gulf Coast",
        "lat": 29.7604,
        "lon": -95.3698,
        "baseline_dwell_days": 3.4,
        "annual_teu_capacity_millions": 3.9,
        "berths": 30
    },
    "AEJEA": {
        "name": "Jebel Ali Port (Dubai)",
        "city": "Dubai",
        "country": "United Arab Emirates",
        "region": "Middle East",
        "lat": 25.0113,
        "lon": 55.0617,
        "baseline_dwell_days": 2.6,
        "annual_teu_capacity_millions": 14.0,
        "berths": 67
    },
    "LKCMB": {
        "name": "Port of Colombo",
        "city": "Colombo",
        "country": "Sri Lanka",
        "region": "South Asia",
        "lat": 6.9271,
        "lon": 79.8612,
        "baseline_dwell_days": 2.9,
        "annual_teu_capacity_millions": 7.2,
        "berths": 28
    },
    "BRSSZ": {
        "name": "Port of Santos",
        "city": "Santos",
        "country": "Brazil",
        "region": "South America",
        "lat": -23.9618,
        "lon": -46.3322,
        "baseline_dwell_days": 4.6,
        "annual_teu_capacity_millions": 4.8,
        "berths": 35
    },
    "ESVLC": {
        "name": "Port of Valencia",
        "city": "Valencia",
        "country": "Spain",
        "region": "Mediterranean",
        "lat": 39.4699,
        "lon": -0.3763,
        "baseline_dwell_days": 3.2,
        "annual_teu_capacity_millions": 5.6,
        "berths": 36
    }
}

# Critical Global Maritime Choke Points
CHOKE_POINTS: Dict[str, Dict[str, Any]] = {
    "Suez_Canal": {
        "name": "Suez Canal",
        "location": "Egypt",
        "lat": 30.5852,
        "lon": 32.2654,
        "daily_transit_capacity": 65,
        "vulnerability_factors": ["Geopolitical conflict", "Grounding / blockage", "Red Sea security threats"],
        "criticality": "Severe"
    },
    "Panama_Canal": {
        "name": "Panama Canal",
        "location": "Panama",
        "lat": 9.0800,
        "lon": -79.6800,
        "daily_transit_capacity": 36,
        "vulnerability_factors": ["Drought / freshwater levels", "Draft restrictions", "Booking auctions"],
        "criticality": "High"
    },
    "Strait_of_Malacca": {
        "name": "Strait of Malacca",
        "location": "Malaysia / Indonesia",
        "lat": 2.5000,
        "lon": 101.5000,
        "daily_transit_capacity": 230,
        "vulnerability_factors": ["High traffic density", "Collisions", "Piracy"],
        "criticality": "High"
    },
    "Bab_el_Mandeb": {
        "name": "Bab el-Mandeb Strait",
        "location": "Yemen / Djibouti",
        "lat": 12.5833,
        "lon": 43.3333,
        "daily_transit_capacity": 60,
        "vulnerability_factors": ["Houthi drone/missile strikes", "Naval rerouting via Cape of Good Hope"],
        "criticality": "Critical"
    },
    "Strait_of_Hormuz": {
        "name": "Strait of Hormuz",
        "location": "Oman / Iran",
        "lat": 26.5667,
        "lon": 56.2500,
        "daily_transit_capacity": 140,
        "vulnerability_factors": ["Oil tanker seizures", "Military posturing"],
        "criticality": "High"
    },
    "English_Channel": {
        "name": "English Channel (Strait of Dover)",
        "location": "UK / France",
        "lat": 51.0167,
        "lon": 1.4833,
        "daily_transit_capacity": 400,
        "vulnerability_factors": ["Winter storm surges", "Extreme maritime traffic"],
        "criticality": "Moderate"
    }
}

# Major Global Maritime Corridors
MAJOR_ROUTES: List[Dict[str, Any]] = [
    {
        "route_id": "R01_SHA_RTM_SUEZ",
        "route_name": "Asia - North Europe (via Suez)",
        "origin": "CNSHA",
        "destination": "NLRTM",
        "distance_nm": 10525,
        "baseline_transit_days": 28.0,
        "choke_points": ["Strait_of_Malacca", "Bab_el_Mandeb", "Suez_Canal"],
        "cape_reroute_transit_days": 38.0,
        "cape_reroute_distance_nm": 14100
    },
    {
        "route_id": "R02_SHA_LAX_PACIFIC",
        "route_name": "Transpacific Eastbound (Shanghai to Los Angeles)",
        "origin": "CNSHA",
        "destination": "USLAX",
        "distance_nm": 5700,
        "baseline_transit_days": 14.5,
        "choke_points": [],
        "cape_reroute_transit_days": 14.5,
        "cape_reroute_distance_nm": 5700
    },
    {
        "route_id": "R03_SIN_RTM_SUEZ",
        "route_name": "Southeast Asia - Europe (via Suez)",
        "origin": "SGSIN",
        "destination": "NLRTM",
        "distance_nm": 8288,
        "baseline_transit_days": 22.0,
        "choke_points": ["Bab_el_Mandeb", "Suez_Canal"],
        "cape_reroute_transit_days": 32.0,
        "cape_reroute_distance_nm": 11750
    },
    {
        "route_id": "R04_NGB_NYC_PANAMA",
        "route_name": "Asia - US East Coast (via Panama Canal)",
        "origin": "CNNGB",
        "destination": "USNYC",
        "distance_nm": 10580,
        "baseline_transit_days": 26.5,
        "choke_points": ["Panama_Canal"],
        "cape_reroute_transit_days": 36.0,
        "cape_reroute_distance_nm": 14500
    },
    {
        "route_id": "R05_SZX_HAM_SUEZ",
        "route_name": "South China - Germany (via Suez)",
        "origin": "CNSZX",
        "destination": "DEHAM",
        "distance_nm": 9850,
        "baseline_transit_days": 27.0,
        "choke_points": ["Strait_of_Malacca", "Bab_el_Mandeb", "Suez_Canal", "English_Channel"],
        "cape_reroute_transit_days": 37.5,
        "cape_reroute_distance_nm": 13700
    },
    {
        "route_id": "R06_PUS_LGB_PACIFIC",
        "route_name": "Korea - US West Coast",
        "origin": "KRPUS",
        "destination": "USLGB",
        "distance_nm": 5120,
        "baseline_transit_days": 13.0,
        "choke_points": [],
        "cape_reroute_transit_days": 13.0,
        "cape_reroute_distance_nm": 5120
    },
    {
        "route_id": "R07_RTM_NYC_ATLANTIC",
        "route_name": "Transatlantic Westbound (Rotterdam to New York)",
        "origin": "NLRTM",
        "destination": "USNYC",
        "distance_nm": 3380,
        "baseline_transit_days": 8.5,
        "choke_points": ["English_Channel"],
        "cape_reroute_transit_days": 8.5,
        "cape_reroute_distance_nm": 3380
    },
    {
        "route_id": "R08_ANR_HOU_ATLANTIC",
        "route_name": "Europe - US Gulf (Antwerp to Houston)",
        "origin": "BEANR",
        "destination": "USHOU",
        "distance_nm": 4820,
        "baseline_transit_days": 12.0,
        "choke_points": ["English_Channel"],
        "cape_reroute_transit_days": 12.0,
        "cape_reroute_distance_nm": 4820
    },
    {
        "route_id": "R09_SHA_JEA_INDIAN",
        "route_name": "East Asia - Middle East",
        "origin": "CNSHA",
        "destination": "AEJEA",
        "distance_nm": 5200,
        "baseline_transit_days": 14.0,
        "choke_points": ["Strait_of_Malacca", "Strait_of_Hormuz"],
        "cape_reroute_transit_days": 14.0,
        "cape_reroute_distance_nm": 5200
    },
    {
        "route_id": "R10_SSZ_RTM_ATLANTIC",
        "route_name": "South America - Europe (Santos to Rotterdam)",
        "origin": "BRSSZ",
        "destination": "NLRTM",
        "distance_nm": 5450,
        "baseline_transit_days": 14.0,
        "choke_points": ["English_Channel"],
        "cape_reroute_transit_days": 14.0,
        "cape_reroute_distance_nm": 5450
    },
    {
        "route_id": "R11_CMB_VLC_SUEZ",
        "route_name": "South Asia - Mediterranean (Colombo to Valencia)",
        "origin": "LKCMB",
        "destination": "ESVLC",
        "distance_nm": 5100,
        "baseline_transit_days": 13.5,
        "choke_points": ["Bab_el_Mandeb", "Suez_Canal"],
        "cape_reroute_transit_days": 25.0,
        "cape_reroute_distance_nm": 9800
    },
    {
        "route_id": "R12_SIN_LAX_PACIFIC",
        "route_name": "Southeast Asia - US West Coast",
        "origin": "SGSIN",
        "destination": "USLAX",
        "distance_nm": 7750,
        "baseline_transit_days": 19.5,
        "choke_points": [],
        "cape_reroute_transit_days": 19.5,
        "cape_reroute_distance_nm": 7750
    }
]

# Vessel Types & Capacities
VESSEL_CLASSES: Dict[str, Dict[str, Any]] = {
    "Ultra_Large_Container_Vessel": {
        "teu_range": (18000, 24000),
        "typical_cruising_knots": 18.5,
        "fuel_burn_tonnes_day": 140,
        "draft_meters": 16.0,
        "can_transit_panama": False  # Too large for Panama Canal Neopanamax locks
    },
    "New_Panamax": {
        "teu_range": (10000, 14999),
        "typical_cruising_knots": 19.5,
        "fuel_burn_tonnes_day": 105,
        "draft_meters": 15.0,
        "can_transit_panama": True
    },
    "Post_Panamax": {
        "teu_range": (5000, 9999),
        "typical_cruising_knots": 20.0,
        "fuel_burn_tonnes_day": 75,
        "draft_meters": 13.5,
        "can_transit_panama": True
    },
    "Feeder_Max": {
        "teu_range": (1500, 4999),
        "typical_cruising_knots": 17.0,
        "fuel_burn_tonnes_day": 40,
        "draft_meters": 10.5,
        "can_transit_panama": True
    }
}
