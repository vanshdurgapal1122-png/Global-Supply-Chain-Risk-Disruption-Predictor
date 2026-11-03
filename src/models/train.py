"""
Model Training Pipeline.
Trains XGBoost models for Route Delay Regression, Port Bottleneck Risk Classification,
and Disruption Severity Index scoring. Computes comprehensive evaluation metrics and feature importances.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from config.config import MODEL_CONFIG, SAVED_MODELS_DIR, RAW_DATA_DIR
from src.data_ingestion.shipping_data import ShippingDataGenerator
from src.feature_engineering.pipeline import SupplyChainFeaturePipeline

logger = logging.getLogger("ModelTrainer")


class ModelTrainer:
    """
    Orchestrates data preparation, cross-validation, model training,
    and serialization for supply chain disruption forecasting.
    """

    def __init__(self, models_dir: Path = SAVED_MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feature_pipeline = SupplyChainFeaturePipeline(artifacts_dir=self.models_dir)

    def train_all_models(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Trains and validates all three models:
        1. Route Delay Regressor (XGBRegressor)
        2. Port Bottleneck Classifier (XGBClassifier)
        3. Disruption Severity Regressor (XGBRegressor)
        """
        logger.info(f"Starting model training on {len(df)} voyage records...")

        # Feature transformation
        X, y_delay, y_bottleneck, y_dsi = self.feature_pipeline.fit_transform(df)
        feature_names = self.feature_pipeline.feature_names

        # Train/Test Split (80/20)
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=MODEL_CONFIG.test_size,
            random_state=MODEL_CONFIG.random_state,
            stratify=y_bottleneck
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_delay_train, y_delay_test = y_delay[train_idx], y_delay[test_idx]
        y_bot_train, y_bot_test = y_bottleneck[train_idx], y_bottleneck[test_idx]
        y_dsi_train, y_dsi_test = y_dsi[train_idx], y_dsi[test_idx]

        metrics_summary = {}

        # ----------------------------------------------------
        # 1. Route Delay Regressor (Days)
        # ----------------------------------------------------
        logger.info("Training Model 1: Route Delay XGBRegressor...")
        model_delay = XGBRegressor(
            n_estimators=MODEL_CONFIG.n_estimators,
            learning_rate=MODEL_CONFIG.learning_rate,
            max_depth=MODEL_CONFIG.max_depth,
            subsample=MODEL_CONFIG.subsample,
            colsample_bytree=MODEL_CONFIG.colsample_bytree,
            random_state=MODEL_CONFIG.random_state,
            n_jobs=-1
        )
        model_delay.fit(X_train, y_delay_train)
        preds_delay = model_delay.predict(X_test)

        mae_delay = float(mean_absolute_error(y_delay_test, preds_delay))
        rmse_delay = float(root_mean_squared_error(y_delay_test, preds_delay))
        r2_delay = float(r2_score(y_delay_test, preds_delay))

        metrics_summary["delay_regressor"] = {
            "mae_days": round(mae_delay, 3),
            "rmse_days": round(rmse_delay, 3),
            "r2_score": round(r2_delay, 4),
            "median_delay_test": round(float(np.median(y_delay_test)), 2)
        }
        logger.info(f"Delay Regressor - MAE: {mae_delay:.3f} days | RMSE: {rmse_delay:.3f} | R²: {r2_delay:.4f}")

        # ----------------------------------------------------
        # 2. Port Bottleneck Risk Classifier (Low, Medium, High, Critical)
        # ----------------------------------------------------
        logger.info("Training Model 2: Port Bottleneck XGBClassifier...")
        model_bottleneck = XGBClassifier(
            n_estimators=MODEL_CONFIG.n_estimators,
            learning_rate=MODEL_CONFIG.learning_rate,
            max_depth=MODEL_CONFIG.max_depth,
            subsample=MODEL_CONFIG.subsample,
            colsample_bytree=MODEL_CONFIG.colsample_bytree,
            random_state=MODEL_CONFIG.random_state,
            eval_metric="mlogloss",
            n_jobs=-1
        )
        model_bottleneck.fit(X_train, y_bot_train)
        preds_bot = model_bottleneck.predict(X_test)

        acc_bot = float(accuracy_score(y_bot_test, preds_bot))
        f1_bot = float(f1_score(y_bot_test, preds_bot, average="weighted"))
        cm_bot = confusion_matrix(y_bot_test, preds_bot).tolist()

        metrics_summary["bottleneck_classifier"] = {
            "accuracy": round(acc_bot, 4),
            "weighted_f1": round(f1_bot, 4),
            "confusion_matrix": cm_bot,
            "classes": ["Low", "Medium", "High", "Critical"]
        }
        logger.info(f"Bottleneck Classifier - Accuracy: {acc_bot*100:.2f}% | Weighted F1: {f1_bot:.4f}")

        # ----------------------------------------------------
        # 3. Disruption Severity Index Regressor (0 to 100)
        # ----------------------------------------------------
        logger.info("Training Model 3: Disruption Severity Index (DSI) XGBRegressor...")
        model_dsi = XGBRegressor(
            n_estimators=MODEL_CONFIG.n_estimators,
            learning_rate=MODEL_CONFIG.learning_rate,
            max_depth=MODEL_CONFIG.max_depth,
            subsample=MODEL_CONFIG.subsample,
            colsample_bytree=MODEL_CONFIG.colsample_bytree,
            random_state=MODEL_CONFIG.random_state,
            n_jobs=-1
        )
        model_dsi.fit(X_train, y_dsi_train)
        preds_dsi = model_dsi.predict(X_test)

        mae_dsi = float(mean_absolute_error(y_dsi_test, preds_dsi))
        rmse_dsi = float(root_mean_squared_error(y_dsi_test, preds_dsi))
        r2_dsi = float(r2_score(y_dsi_test, preds_dsi))

        metrics_summary["dsi_regressor"] = {
            "mae_score": round(mae_dsi, 2),
            "rmse_score": round(rmse_dsi, 2),
            "r2_score": round(r2_dsi, 4)
        }
        logger.info(f"DSI Regressor - MAE: {mae_dsi:.2f} | RMSE: {rmse_dsi:.2f} | R²: {r2_dsi:.4f}")

        # ----------------------------------------------------
        # Extract Feature Importances
        # ----------------------------------------------------
        delay_importances = model_delay.feature_importances_
        sorted_indices = np.argsort(delay_importances)[::-1]
        top_features = [
            {"feature": feature_names[i], "importance": round(float(delay_importances[i]), 4)}
            for i in sorted_indices[:15]
        ]
        metrics_summary["top_delay_features"] = top_features

        # ----------------------------------------------------
        # Serialize Artifacts
        # ----------------------------------------------------
        joblib.dump(model_delay, self.models_dir / "model_delay_regressor.joblib")
        joblib.dump(model_bottleneck, self.models_dir / "model_bottleneck_classifier.joblib")
        joblib.dump(model_dsi, self.models_dir / "model_dsi_regressor.joblib")

        with open(self.models_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        logger.info("All trained models and metadata successfully persisted to disk.")
        return metrics_summary


def train_pipeline(n_samples: int = 5500) -> Dict[str, Any]:
    """CLI and program helper to generate data and run end-to-end training."""
    data_gen = ShippingDataGenerator()
    df = data_gen.generate_dataset(n_samples=n_samples)
    trainer = ModelTrainer()
    return trainer.train_all_models(df)


if __name__ == "__main__":
    train_pipeline()
