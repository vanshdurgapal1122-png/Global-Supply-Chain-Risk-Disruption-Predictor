"""
Historical Shipping and Maritime Transit Data Generator & Loader.
Generates a statistically realistic, multi-year global shipping dataset synthesizing
AIS transit logs, vessel TEU classifications, canal choke point bottlenecks,
and port congestion dwell metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config.config import RAW_DATA_DIR
from config.ports_routes import PORTS_CATALOG, CHOKE_POINTS, MAJOR_ROUTES, VESSEL_CLASSES

logger = logging.getLogger("ShippingData")

CARRIER_NAMES = [
    "Mediterranean Shipping Company (MSC)",
    "Maersk Line",
    "CMA CGM Group",
    "COSCO Shipping Lines",
    "Hapag-Lloyd",
    "Ocean Network Express (ONE)",
    "Evergreen Marine",
    "HMM Co."
]


class ShippingDataGenerator:
    """
    Synthesizes and exports realistic historical maritime voyages across global trade corridors.
    Models complex non-linear correlations between choke point status, fuel prices,
    seasonal weather, and destination port dwell bottlenecks.
    """

    def __init__(self, raw_data_dir: Optional[Path] = None):
        self.raw_data_dir = raw_data_dir or RAW_DATA_DIR
        self.output_file = self.raw_data_dir / "historical_shipping_logs.csv"

    def generate_dataset(
        self,
        n_samples: int = 5500,
        start_date: str = "2023-01-01",
        end_date: str = "2025-01-01",
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Generates N realistic voyage records fusing routing, ship dynamics, macro indicators, and sentiment.
        """
        np.random.seed(random_seed)
        logger.info(f"Synthesizing {n_samples} maritime voyage records across {len(MAJOR_ROUTES)} major corridors...")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_days = (end_dt - start_dt).days

        records = []
        vessel_class_names = list(VESSEL_CLASSES.keys())

        for i in range(n_samples):
            voyage_id = f"VOY-{2023 + (i % 2):04d}-{i+1:05d}"
            
            # Select route
            route = MAJOR_ROUTES[i % len(MAJOR_ROUTES)]
            origin_code = route["origin"]
            dest_code = route["destination"]
            origin_meta = PORTS_CATALOG[origin_code]
            dest_meta = PORTS_CATALOG[dest_code]

            # Select vessel class
            v_class_name = np.random.choice(vessel_class_names, p=[0.25, 0.40, 0.25, 0.10])
            v_meta = VESSEL_CLASSES[v_class_name]
            teu = int(np.random.uniform(v_meta["teu_range"][0], v_meta["teu_range"][1]))
            carrier = np.random.choice(CARRIER_NAMES)

            # Voyage timing
            dep_offset = np.random.randint(0, total_days)
            departure_date = start_dt + timedelta(days=int(dep_offset), hours=int(np.random.randint(0, 24)))
            dep_month = departure_date.month

            # Baseline transit metrics
            baseline_days = float(route["baseline_transit_days"])
            distance_nm = float(route["distance_nm"])
            choke_points = route["choke_points"]
            has_suez = "Suez_Canal" in choke_points or "Bab_el_Mandeb" in choke_points
            has_panama = "Panama_Canal" in choke_points

            # Macroeconomic conditions at departure
            # Seasonal/cyclical oil price fluctuation
            brent_price = float(np.clip(
                74.0 + 12.0 * np.sin(dep_offset / 60.0) + np.random.normal(0, 4.0),
                62.0, 115.0
            ))
            fuel_stress = float(np.clip((brent_price - 50.0) / 70.0, 0.05, 1.0))
            usd_cny = float(np.clip(7.15 + 0.0003 * dep_offset + np.random.normal(0, 0.04), 6.95, 7.35))
            eur_usd = float(np.clip(1.08 - 0.0001 * dep_offset + np.random.normal(0, 0.02), 1.03, 1.15))

            # External Disruption Factors:
            # 1. Choke Point Blockade / Rerouting:
            # e.g. Red Sea / Suez crisis (starting ~Nov 2023, high probability of Cape rerouting)
            is_red_sea_crisis_period = (departure_date >= pd.to_datetime("2023-11-15"))
            rerouted_via_cape = False
            if has_suez and is_red_sea_crisis_period and np.random.random() < 0.78:
                rerouted_via_cape = True

            # 2. Panama Canal Drought Restrictions (Summer 2023 - Spring 2024)
            panama_delay_days = 0.0
            if has_panama:
                is_panama_drought_period = (pd.to_datetime("2023-06-01") <= departure_date <= pd.to_datetime("2024-05-01"))
                if is_panama_drought_period:
                    panama_delay_days = float(np.random.gamma(shape=2.5, scale=2.8))  # 4 to 12 days wait
                else:
                    panama_delay_days = float(np.random.exponential(scale=1.2))

            # 3. Seasonal Weather Disruption (Typhoons July-Oct in Asia, Winter storms Nov-Feb in Atlantic/Pacific)
            is_typhoon_season = (dep_month in [7, 8, 9, 10]) and (origin_meta["region"] == "East Asia")
            is_winter_storm_season = (dep_month in [11, 12, 1, 2]) and (dest_meta["region"] in ["North Europe", "US East Coast"])
            
            weather_risk_index = 0.15
            if is_typhoon_season:
                weather_risk_index += np.random.uniform(0.20, 0.65)
            if is_winter_storm_season:
                weather_risk_index += np.random.uniform(0.15, 0.50)
            weather_risk_index = float(np.clip(weather_risk_index + np.random.normal(0, 0.05), 0.05, 0.98))

            # 4. News Disruption Sentiment Factor (FinBERT alignment)
            # Sentiment is heavily negative when crises, strikes, or reroutings occur
            base_sentiment = 0.25
            if rerouted_via_cape:
                base_sentiment += 0.55
            if panama_delay_days > 4.0:
                base_sentiment += 0.35
            if weather_risk_index > 0.60:
                base_sentiment += 0.20
            news_disruption_sentiment = float(np.clip(base_sentiment + np.random.normal(0, 0.08), 0.05, 0.99))

            # 5. Slow-Steaming Effect from High Oil Prices:
            # High fuel prices cause carriers to reduce cruising speed by 1-2 knots to save bunker fuel
            slow_steaming_days = 0.0
            if brent_price > 88.0:
                slow_steaming_days = (brent_price - 88.0) * 0.08 * (distance_nm / 10000.0)

            # 6. Actual Voyage Transit Time & Delay Calculation:
            reroute_added_days = (route["cape_reroute_transit_days"] - baseline_days) if rerouted_via_cape else 0.0
            weather_delay = weather_risk_index * np.random.uniform(1.0, 4.5)
            
            # Destination port congestion & dwell time
            dest_baseline_dwell = dest_meta["baseline_dwell_days"]
            # Seasonal congestion peak (pre-Christmas rush in Aug-Nov)
            peak_season_factor = 1.35 if dep_month in [8, 9, 10, 11] else 1.0
            
            port_congestion_noise = np.random.gamma(shape=2.0, scale=0.8)
            actual_dwell_time = float(np.clip(
                dest_baseline_dwell * peak_season_factor + port_congestion_noise + (2.0 if news_disruption_sentiment > 0.7 else 0.0),
                1.5, 18.0
            ))

            berth_waiting_hours = float(np.clip(
                (actual_dwell_time - dest_baseline_dwell) * 16.0 + np.random.normal(12, 6),
                4.0, 168.0
            ))

            # Total route delay in days (Actual voyage transit - baseline transit)
            pure_transit_delay = float(np.clip(
                reroute_added_days + panama_delay_days + weather_delay + slow_steaming_days + np.random.normal(0.4, 0.6),
                0.0, 25.0
            ))

            actual_transit_days = round(baseline_days + pure_transit_delay, 2)
            route_delay_days = round(pure_transit_delay, 2)
            arrival_date = departure_date + timedelta(days=actual_transit_days)

            # Target 2: Port Bottleneck Risk Classification
            # Multi-class classification based on destination dwell time & berth waiting:
            if actual_dwell_time < 3.2 and berth_waiting_hours < 24.0:
                bottleneck_risk = "Low"
                bottleneck_risk_code = 0
            elif actual_dwell_time < 5.0 and berth_waiting_hours < 48.0:
                bottleneck_risk = "Medium"
                bottleneck_risk_code = 1
            elif actual_dwell_time < 8.0 and berth_waiting_hours < 96.0:
                bottleneck_risk = "High"
                bottleneck_risk_code = 2
            else:
                bottleneck_risk = "Critical"
                bottleneck_risk_code = 3

            # Target 3: Disruption Severity Index (0 - 100 Continuous Score)
            # Fuses delay (scaled to 15 days), bottleneck code, fuel stress, and sentiment
            norm_delay = np.clip(route_delay_days / 14.0, 0.0, 1.0)
            norm_bottleneck = bottleneck_risk_code / 3.0
            
            dsi = (
                0.40 * norm_delay +
                0.30 * norm_bottleneck +
                0.15 * fuel_stress +
                0.15 * news_disruption_sentiment
            ) * 100.0
            dsi = round(float(np.clip(dsi, 5.0, 99.5)), 1)

            records.append({
                "voyage_id": voyage_id,
                "route_id": route["route_id"],
                "route_name": route["route_name"],
                "origin_port": origin_code,
                "dest_port": dest_code,
                "origin_region": origin_meta["region"],
                "dest_region": dest_meta["region"],
                "carrier_name": carrier,
                "vessel_class": v_class_name,
                "teu_capacity": teu,
                "distance_nm": distance_nm,
                "baseline_transit_days": baseline_days,
                "departure_date": departure_date.strftime("%Y-%m-%d %H:%M:%S"),
                "arrival_date": arrival_date.strftime("%Y-%m-%d %H:%M:%S"),
                "departure_month": dep_month,
                "departure_quarter": (dep_month - 1) // 3 + 1,
                "choke_points_count": len(choke_points),
                "has_suez_passage": int(has_suez),
                "has_panama_passage": int(has_panama),
                "rerouted_via_cape": int(rerouted_via_cape),
                "weather_risk_index": round(weather_risk_index, 3),
                "brent_oil_price": round(brent_price, 2),
                "fuel_stress_index": round(fuel_stress, 3),
                "usd_cny_rate": round(usd_cny, 4),
                "eur_usd_rate": round(eur_usd, 4),
                "news_disruption_sentiment": round(news_disruption_sentiment, 3),
                "dest_dwell_time_days": round(actual_dwell_time, 2),
                "berth_waiting_hours": round(berth_waiting_hours, 1),
                "actual_transit_days": actual_transit_days,
                "route_delay_days": route_delay_days,
                "port_bottleneck_risk": bottleneck_risk,
                "port_bottleneck_risk_code": bottleneck_risk_code,
                "disruption_severity_index": dsi
            })

        df = pd.DataFrame(records)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Generated dataset successfully saved with {len(df)} records to {self.output_file}")
        return df

    def load_dataset(self, auto_generate: bool = True) -> pd.DataFrame:
        """Loads historical shipping logs from disk or auto-generates if not yet created."""
        if self.output_file.exists():
            return pd.read_csv(self.output_file)
        if auto_generate:
            return self.generate_dataset()
        raise FileNotFoundError(f"Dataset not found at {self.output_file}")
