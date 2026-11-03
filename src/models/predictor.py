"""
Inference and Disruption Scenario Simulation Engine.
Loads trained XGBoost models to provide real-time route delay predictions,
bottleneck risk probabilities, 95% confidence intervals, and what-if stress testing.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from config.config import SAVED_MODELS_DIR, RISK_WEIGHTS
from config.ports_routes import PORTS_CATALOG, MAJOR_ROUTES, VESSEL_CLASSES
from src.feature_engineering.pipeline import SupplyChainFeaturePipeline

logger = logging.getLogger("Predictor")


class SupplyChainPredictor:
    """
    Production inference engine for maritime supply chain disruption forecasting.
    Supports single voyage assessment, risk decomposition, and macro scenario simulations.
    """

    BOTTLENECK_CLASSES = ["Low", "Medium", "High", "Critical"]

    def __init__(self, models_dir: Path = SAVED_MODELS_DIR):
        self.models_dir = models_dir
        self.feature_pipeline = SupplyChainFeaturePipeline(artifacts_dir=self.models_dir)
        self.model_delay = None
        self.model_bottleneck = None
        self.model_dsi = None
        self.metrics = {}
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized XGBoost models, scalers, and validation metrics."""
        try:
            delay_path = self.models_dir / "model_delay_regressor.joblib"
            bot_path = self.models_dir / "model_bottleneck_classifier.joblib"
            dsi_path = self.models_dir / "model_dsi_regressor.joblib"
            metrics_path = self.models_dir / "metrics.json"

            if delay_path.exists() and bot_path.exists() and dsi_path.exists():
                self.model_delay = joblib.load(delay_path)
                self.model_bottleneck = joblib.load(bot_path)
                self.model_dsi = joblib.load(dsi_path)
                
                if metrics_path.exists():
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        self.metrics = json.load(f)
                
                self.is_loaded = True
                logger.info("Successfully loaded all ML models and scalers into memory.")
            else:
                logger.warning("Model files not found. Please train models first using ModelTrainer.")
        except Exception as e:
            logger.error(f"Error loading trained models: {e}")
            self.is_loaded = False

    def predict_voyage(
        self,
        origin_port: str,
        dest_port: str,
        vessel_class: str = "New_Panamax",
        teu_capacity: Optional[int] = None,
        departure_month: int = 10,
        rerouted_via_cape: bool = False,
        weather_risk_index: float = 0.25,
        brent_oil_price: float = 85.0,
        news_disruption_sentiment: float = 0.35,
        usd_cny_rate: float = 7.22,
        eur_usd_rate: float = 1.08
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-model inference for a specific shipping transit.
        Returns predicted delay days with 95% CI, bottleneck risk probabilities, and DSI score.
        """
        if not self.is_loaded:
            raise RuntimeError("Models are not loaded. Train the models before calling predict_voyage.")

        # Route metadata lookup
        origin_meta = PORTS_CATALOG.get(origin_port, PORTS_CATALOG["CNSHA"])
        dest_meta = PORTS_CATALOG.get(dest_port, PORTS_CATALOG["NLRTM"])

        # Find matching route or estimate distance
        matched_route = None
        for r in MAJOR_ROUTES:
            if r["origin"] == origin_port and r["destination"] == dest_port:
                matched_route = r
                break

        if matched_route:
            distance_nm = matched_route["distance_nm"]
            baseline_transit = matched_route["baseline_transit_days"]
            choke_points = matched_route["choke_points"]
        else:
            # Geodesic approximate fallback
            lat1, lon1 = origin_meta["lat"], origin_meta["lon"]
            lat2, lon2 = dest_meta["lat"], dest_meta["lon"]
            # Approximate nautical miles from degrees
            deg_dist = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
            distance_nm = deg_dist * 60.0 * 1.15
            baseline_transit = distance_nm / (19.0 * 24.0)
            choke_points = ["Suez_Canal"] if "Europe" in dest_meta["region"] else []

        has_suez = int("Suez_Canal" in choke_points or "Bab_el_Mandeb" in choke_points)
        has_panama = int("Panama_Canal" in choke_points)

        v_meta = VESSEL_CLASSES.get(vessel_class, VESSEL_CLASSES["New_Panamax"])
        teu = teu_capacity or int(np.mean(v_meta["teu_range"]))

        fuel_stress = float(np.clip((brent_oil_price - 50.0) / 70.0, 0.05, 1.0))

        # Build feature dictionary
        features_dict = {
            "origin_port": origin_port,
            "dest_port": dest_port,
            "vessel_class": vessel_class,
            "teu_capacity": teu,
            "distance_nm": distance_nm,
            "baseline_transit_days": baseline_transit,
            "departure_month": departure_month,
            "choke_points_count": len(choke_points),
            "has_suez_passage": has_suez,
            "has_panama_passage": has_panama,
            "rerouted_via_cape": int(rerouted_via_cape),
            "weather_risk_index": weather_risk_index,
            "brent_oil_price": brent_oil_price,
            "fuel_stress_index": fuel_stress,
            "usd_cny_rate": usd_cny_rate,
            "eur_usd_rate": eur_usd_rate,
            "news_disruption_sentiment": news_disruption_sentiment
        }

        # Feature matrix transformation
        X_vec = self.feature_pipeline.prepare_single_inference(features_dict)

        # 1. Delay Regression
        pred_delay = float(self.model_delay.predict(X_vec)[0])
        pred_delay = max(0.0, round(pred_delay, 2))
        
        # 95% Confidence Interval based on validation RMSE
        rmse = self.metrics.get("delay_regressor", {}).get("rmse_days", 1.1)
        ci_lower = max(0.0, round(pred_delay - 1.96 * rmse, 2))
        ci_upper = round(pred_delay + 1.96 * rmse, 2)

        # 2. Port Bottleneck Classification
        bot_probs = self.model_bottleneck.predict_proba(X_vec)[0]
        bot_class_idx = int(np.argmax(bot_probs))
        predicted_bottleneck = self.BOTTLENECK_CLASSES[bot_class_idx]
        bottleneck_prob_dict = {
            cls: round(float(bot_probs[i]), 3) for i, cls in enumerate(self.BOTTLENECK_CLASSES)
        }

        # 3. Disruption Severity Index (DSI)
        pred_dsi = float(self.model_dsi.predict(X_vec)[0])
        pred_dsi = float(np.clip(pred_dsi, 0.0, 100.0))
        pred_dsi = round(pred_dsi, 1)

        # Risk Tier categorization
        if pred_dsi < 30.0:
            risk_tier = "Normal Operations"
            risk_color = "#10B981"  # Emerald Green
        elif pred_dsi < 55.0:
            risk_tier = "Elevated Risk"
            risk_color = "#F59E0B"  # Amber
        elif pred_dsi < 75.0:
            risk_tier = "High Vulnerability"
            risk_color = "#F97316"  # Orange
        else:
            risk_tier = "Critical Disruption Alert"
            risk_color = "#EF4444"  # Red

        # Identify key drivers of predicted delay
        drivers = []
        if rerouted_via_cape:
            drivers.append("Canal Rerouting via Cape of Good Hope (+8-12 days)")
        if weather_risk_index > 0.50:
            drivers.append("Severe Seasonal Weather / Typhoon Zone")
        if news_disruption_sentiment > 0.65:
            drivers.append("High Negative News Sentiment (Active Corridor Strikes/Attacks)")
        if brent_oil_price > 95.0:
            drivers.append("Elevated Bunker Fuel Inducing Carrier Slow-Steaming")
        if has_panama and pred_delay > 4.0:
            drivers.append("Panama Canal Draft / Transit Queuing Delay")
        if not drivers:
            drivers.append("Standard Maritime Operational Variations")

        return {
            "origin_port": origin_port,
            "origin_name": origin_meta["name"],
            "dest_port": dest_port,
            "dest_name": dest_meta["name"],
            "distance_nm": distance_nm,
            "baseline_transit_days": baseline_transit,
            "predicted_total_transit_days": round(baseline_transit + pred_delay, 2),
            "predicted_delay_days": pred_delay,
            "delay_ci_95": [ci_lower, ci_upper],
            "bottleneck_risk": predicted_bottleneck,
            "bottleneck_probabilities": bottleneck_prob_dict,
            "disruption_severity_index": pred_dsi,
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "primary_drivers": drivers
        }

    def run_stress_test_scenarios(
        self,
        origin_port: str = "CNSHA",
        dest_port: str = "NLRTM",
        vessel_class: str = "New_Panamax"
    ) -> List[Dict[str, Any]]:
        """
        Executes comparative stress testing across 5 major global supply chain crisis scenarios.
        """
        scenarios = [
            {
                "name": "1. Baseline (Calm Seas)",
                "desc": "Normal transit conditions, stable bunker fuel, standard port dwell.",
                "cape": False, "weather": 0.15, "oil": 78.0, "sentiment": 0.20
            },
            {
                "name": "2. Red Sea / Bab el-Mandeb Blockade",
                "desc": "Suez route closed due to drone attacks, forced Cape rerouting, intense negative sentiment.",
                "cape": True, "weather": 0.25, "oil": 92.0, "sentiment": 0.88
            },
            {
                "name": "3. Extreme Weather & Severe Congestion",
                "desc": "Category 4 typhoons in East Asia + peak winter North Atlantic storm front.",
                "cape": False, "weather": 0.85, "oil": 84.0, "sentiment": 0.65
            },
            {
                "name": "4. Global Oil Shock ($125/bbl)",
                "desc": "Middle East crisis drives bunker fuel to record highs, triggering severe slow steaming.",
                "cape": False, "weather": 0.20, "oil": 125.0, "sentiment": 0.70
            },
            {
                "name": "5. Compound Crisis (War + Strike + Fuel)",
                "desc": "Simultaneous canal rerouting, destination dock strike, and surging fuel surcharges.",
                "cape": True, "weather": 0.60, "oil": 115.0, "sentiment": 0.95
            }
        ]

        results = []
        for s in scenarios:
            res = self.predict_voyage(
                origin_port=origin_port,
                dest_port=dest_port,
                vessel_class=vessel_class,
                rerouted_via_cape=s["cape"],
                weather_risk_index=s["weather"],
                brent_oil_price=s["oil"],
                news_disruption_sentiment=s["sentiment"]
            )
            results.append({
                "scenario_name": s["name"],
                "description": s["desc"],
                "predicted_delay_days": res["predicted_delay_days"],
                "delay_ci_95": res["delay_ci_95"],
                "bottleneck_risk": res["bottleneck_risk"],
                "dsi": res["disruption_severity_index"],
                "risk_tier": res["risk_tier"],
                "risk_color": res["risk_color"]
            })

        return results
