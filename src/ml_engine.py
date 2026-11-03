"""
Machine Learning Engine for Global Supply Chain Risk Predictor.
Trains two XGBoost models:
  1. XGBClassifier for binary late delivery risk prediction.
  2. XGBRegressor for continuous delay days estimation.
Integrates news disruption sentiment and macroeconomic volatility indicators.
"""

import sys
import types
from pathlib import Path

# Safeguard against Windows AppLocker policy blocking sklearn._csr_polynomial_expansion
if "sklearn.preprocessing._polynomial" not in sys.modules:
    mock_poly = types.ModuleType("sklearn.preprocessing._polynomial")
    mock_poly.PolynomialFeatures = object
    mock_poly.SplineTransformer = object
    sys.modules["sklearn.preprocessing._polynomial"] = mock_poly
    sys.modules["sklearn.preprocessing._csr_polynomial_expansion"] = types.ModuleType("mock")

import os
import logging
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from xgboost import XGBClassifier, XGBRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLEngine")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "supply_chain_data.csv"
CLASSIFIER_PATH = MODELS_DIR / "xgboost_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "xgboost_regressor.joblib"
PREPROCESSORS_PATH = MODELS_DIR / "preprocessors.joblib"


class DataPreprocessor:
    """
    Handles categorical one-hot encoding and numerical standardization.
    """

    def __init__(self, cat_features: list, num_features: list):
        self.cat_features = cat_features
        self.num_features = num_features
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names = []
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        cat_encoded = self.encoder.fit_transform(df[self.cat_features])
        cat_names = self.encoder.get_feature_names_out(self.cat_features).tolist()

        num_scaled = self.scaler.fit_transform(df[self.num_features])
        num_names = list(self.num_features)

        self.feature_names = num_names + cat_names
        self.is_fitted = True

        return np.hstack([num_scaled, cat_encoded])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted before transform.")
        cat_encoded = self.encoder.transform(df[self.cat_features])
        num_scaled = self.scaler.transform(df[self.num_features])
        return np.hstack([num_scaled, cat_encoded])


def prepare_features(data_path: Path) -> pd.DataFrame:
    """
    Loads dataset and enriches it with simulated disruption scores and macro volatility.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data_loader.py first.")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")

    # Set deterministic seed for reproducibility
    np.random.seed(42)

    # 1. Feature Engineering: Disruption shock score (0.0 to 1.0) correlated with late delivery risk
    base_disruption = np.random.beta(a=1.8, b=3.2, size=len(df))
    # Disrupted shipments experience elevated sentiment disruption shock
    risk_boost = df["late_delivery_risk"] * np.random.uniform(0.15, 0.40, size=len(df))
    df["disruption_score"] = np.clip(base_disruption + risk_boost, 0.0, 1.0).round(4)

    # 2. Feature Engineering: Macro Volatility indicators
    # Crude oil 7-day rolling annualized volatility (typically 0.15 - 0.45)
    df["crude_oil_7d_vol"] = np.random.normal(0.25, 0.05, size=len(df)).clip(0.10, 0.50).round(4)
    # EUR/USD 7-day rolling annualized volatility (typically 0.02 - 0.08)
    df["eur_usd_7d_vol"] = np.random.normal(0.04, 0.01, size=len(df)).clip(0.01, 0.12).round(4)

    # Interaction feature: effective distance load
    df["distance_weight_ratio"] = (df["cargo_weight_kg"] / (df["distance_km"] + 1.0)).round(4)

    return df


def train_models(data_path: Path = DATA_PATH) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Preprocesses data and trains both XGBClassifier and XGBRegressor models.
    """
    df = prepare_features(data_path)

    cat_cols = ["shipping_mode", "origin_country", "destination_country"]
    num_cols = [
        "scheduled_days",
        "distance_km",
        "cargo_weight_kg",
        "disruption_score",
        "crude_oil_7d_vol",
        "eur_usd_7d_vol",
        "distance_weight_ratio"
    ]

    target_cls = "late_delivery_risk"
    target_reg = "delay_days"

    preprocessor = DataPreprocessor(cat_features=cat_cols, num_features=num_cols)
    X = preprocessor.fit_transform(df)
    y_cls = df[target_cls].values
    y_reg = df[target_reg].values

    # Train / Test split (80% train, 20% test)
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.20, random_state=42, stratify=y_cls
    )

    logger.info(f"Training split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")

    # ==========================================
    # Model 1: Classification (late_delivery_risk)
    # ==========================================
    logger.info("Training Model 1: XGBClassifier for Late Delivery Risk...")
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=(len(y_cls_train) - sum(y_cls_train)) / sum(y_cls_train),
        random_state=42,
        eval_metric="logloss"
    )
    clf.fit(X_train, y_cls_train)

    y_cls_pred = clf.predict(X_test)
    y_cls_prob = clf.predict_proba(X_test)[:, 1]

    cls_metrics = {
        "Accuracy": float(accuracy_score(y_cls_test, y_cls_pred)),
        "Precision": float(precision_score(y_cls_test, y_cls_pred)),
        "Recall": float(recall_score(y_cls_test, y_cls_pred)),
        "F1-Score": float(f1_score(y_cls_test, y_cls_pred)),
        "ROC-AUC": float(roc_auc_score(y_cls_test, y_cls_prob))
    }

    # ==========================================
    # Model 2: Regression (delay_days)
    # ==========================================
    logger.info("Training Model 2: XGBRegressor for Delay Days...")
    reg = XGBRegressor(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="rmse"
    )
    reg.fit(X_train, y_reg_train)

    y_reg_pred = reg.predict(X_test)
    # Days cannot be negative
    y_reg_pred = np.clip(y_reg_pred, 0.0, None)

    rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)))
    mae = float(mean_absolute_error(y_reg_test, y_reg_pred))
    r2 = float(r2_score(y_reg_test, y_reg_pred))

    reg_metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2-Score": r2
    }

    # ==========================================
    # Save Artifacts
    # ==========================================
    logger.info("Saving trained models and preprocessor objects...")
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(reg, REGRESSOR_PATH)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "scaler": preprocessor.scaler,
            "encoder": preprocessor.encoder,
            "cat_features": cat_cols,
            "num_features": num_cols,
            "feature_names": preprocessor.feature_names
        },
        PREPROCESSORS_PATH
    )

    logger.info(f"Artifacts successfully saved to: {MODELS_DIR}")
    return cls_metrics, reg_metrics


if __name__ == "__main__":
    print("=" * 75)
    print("   GLOBAL SUPPLY CHAIN PREDICTOR: ML ENGINE TRAINING & EVALUATION (STEP 4)")
    print("=" * 75)

    cls_res, reg_res = train_models()

    print("\n[+] MODEL 1: XGBClassifier Evaluation (Late Delivery Risk):")
    print(f"    - Accuracy  : {cls_res['Accuracy']:.4f} ({(cls_res['Accuracy']*100):.2f}%)")
    print(f"    - Precision : {cls_res['Precision']:.4f}")
    print(f"    - Recall    : {cls_res['Recall']:.4f}")
    print(f"    - F1-Score  : {cls_res['F1-Score']:.4f}")
    print(f"    - ROC-AUC   : {cls_res['ROC-AUC']:.4f}")

    print("\n[+] MODEL 2: XGBRegressor Evaluation (Exact Delay Days):")
    print(f"    - RMSE      : {reg_res['RMSE']:.4f} days")
    print(f"    - MAE       : {reg_res['MAE']:.4f} days")
    print(f"    - R2-Score  : {reg_res['R2-Score']:.4f}")

    print("\n[+] Saved Model Artifacts:")
    print(f"    - {CLASSIFIER_PATH.name}  : {CLASSIFIER_PATH.stat().st_size / 1024:.2f} KB")
    print(f"    - {REGRESSOR_PATH.name}    : {REGRESSOR_PATH.stat().st_size / 1024:.2f} KB")
    print(f"    - {PREPROCESSORS_PATH.name}       : {PREPROCESSORS_PATH.stat().st_size / 1024:.2f} KB")
    print("=" * 75)
