"""
Feature Engineering and Multi-Modal Data Fusion Pipeline.
Fuses historical maritime transit variables, macroeconomic indicators, and FinBERT sentiment scores
into standardized, feature-engineered matrices ready for XGBoost modeling.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from config.config import SAVED_MODELS_DIR, PROCESSED_DATA_DIR
from config.ports_routes import PORTS_CATALOG, MAJOR_ROUTES

logger = logging.getLogger("FeaturePipeline")


class SupplyChainFeaturePipeline:
    """
    Transforms raw multi-modal shipping, macro, and sentiment data into ML feature matrices.
    Computes interaction terms, non-linear physical logistics proxies, and scales inputs.
    """

    NUMERIC_FEATURES = [
        "teu_capacity",
        "distance_nm",
        "baseline_transit_days",
        "choke_points_count",
        "has_suez_passage",
        "has_panama_passage",
        "rerouted_via_cape",
        "weather_risk_index",
        "brent_oil_price",
        "fuel_stress_index",
        "usd_cny_rate",
        "eur_usd_rate",
        "news_disruption_sentiment",
        "fuel_distance_pressure",
        "choke_sentiment_interaction",
        "weather_canal_interaction",
        "expected_knots"
    ]

    CATEGORICAL_FEATURES = [
        "origin_port",
        "dest_port",
        "vessel_class",
        "departure_month"
    ]

    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or SAVED_MODELS_DIR
        self.preprocessor_file = self.artifacts_dir / "preprocessor.joblib"
        self.feature_names_file = self.artifacts_dir / "feature_names.joblib"
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []

    def compute_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives logistical and economic interaction terms."""
        data = df.copy()

        # 1. Fuel cost pressure index ($/bbl * 1000 nautical miles)
        oil = data.get("brent_oil_price", 80.0)
        dist = data.get("distance_nm", 7000.0)
        data["fuel_distance_pressure"] = (oil * (dist / 10000.0)).round(3)

        # 2. Choke point risk amplified by negative news sentiment
        chokes = data.get("choke_points_count", 1)
        sentiment = data.get("news_disruption_sentiment", 0.3)
        data["choke_sentiment_interaction"] = (chokes * (1.0 + sentiment * 2.0)).round(3)

        # 3. Weather risk interacting with canal passage vulnerability
        weather = data.get("weather_risk_index", 0.2)
        has_canal = data.get("has_suez_passage", 0) + data.get("has_panama_passage", 0)
        data["weather_canal_interaction"] = (weather * (1.0 + has_canal)).round(3)

        # 4. Expected cruising speed in knots (Distance NM / (Transit days * 24))
        transit_days = data.get("baseline_transit_days", 14.0)
        transit_days = np.where(transit_days <= 0, 1.0, transit_days)
        data["expected_knots"] = (dist / (transit_days * 24.0)).round(2)

        return data

    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits the preprocessor on the dataset and returns transformed X and target arrays:
        Returns:
            X_transformed: np.ndarray
            y_delay: np.ndarray (Continuous route delay days)
            y_bottleneck: np.ndarray (Multi-class bottleneck risk code 0-3)
            y_dsi: np.ndarray (Composite disruption severity index 0-100)
        """
        logger.info("Fitting feature engineering pipeline and column transformers...")
        data = self.compute_interaction_features(df)

        # Preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.CATEGORICAL_FEATURES)
            ]
        )

        X_transformed = self.preprocessor.fit_transform(data)

        # Extract transformed feature names
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(self.CATEGORICAL_FEATURES))
        self.feature_names = self.NUMERIC_FEATURES + cat_feature_names

        # Save artifacts
        joblib.dump(self.preprocessor, self.preprocessor_file)
        joblib.dump(self.feature_names, self.feature_names_file)
        logger.info(f"Saved feature preprocessor to {self.preprocessor_file} (Total features: {len(self.feature_names)})")

        y_delay = data["route_delay_days"].values
        y_bottleneck = data["port_bottleneck_risk_code"].values
        y_dsi = data["disruption_severity_index"].values

        return X_transformed, y_delay, y_bottleneck, y_dsi

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms new raw dataframe using the fitted preprocessor."""
        if self.preprocessor is None:
            if self.preprocessor_file.exists():
                self.preprocessor = joblib.load(self.preprocessor_file)
                self.feature_names = joblib.load(self.feature_names_file)
            else:
                raise ValueError("Preprocessor not fitted and no saved preprocessor found on disk.")

        data = self.compute_interaction_features(df)
        return self.preprocessor.transform(data)

    def prepare_single_inference(self, params: Dict[str, Any]) -> np.ndarray:
        """Converts user dashboard parameters into a 1-row feature vector for ML inference."""
        df_single = pd.DataFrame([params])
        return self.transform(df_single)
