"""
Configuration module for the Global Supply Chain Risk & Disruption Predictor.
Centralizes paths, hyperparameters, model constants, and feature definitions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Hyperparameters and configuration for ML models."""
    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 250
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    early_stopping_rounds: int = 20

@dataclass
class NLPConfig:
    """FinBERT sentiment analysis configuration."""
    model_name: str = "ProsusAI/finbert"
    max_seq_length: int = 128
    batch_size: int = 16
    device: str = "cpu"  # Will dynamically use cuda if available
    sentiment_labels: List[str] = field(default_factory=lambda: ["positive", "negative", "neutral"])

@dataclass
class MacroConfig:
    """Macroeconomic data fetching tickers and parameters."""
    tickers: Dict[str, str] = field(default_factory=lambda: {
        "crude_oil_brent": "BZ=F",       # Brent Crude Oil Front Month
        "crude_oil_wti": "CL=F",         # WTI Crude Oil Front Month
        "eur_usd": "EURUSD=X",           # Euro to US Dollar
        "usd_cny": "CNY=X",              # US Dollar to Chinese Yuan
        "usd_jpy": "JPY=X",              # US Dollar to Japanese Yen
        "freight_etf": "BDRY"            # Breakwave Dry Bulk Shipping ETF
    })
    period: str = "2y"
    interval: str = "1d"
    default_oil_price: float = 82.50
    default_fx_eur_usd: float = 1.0850
    default_fx_usd_cny: float = 7.2300

@dataclass
class RiskScoringWeights:
    """Weights used for calculating the composite Disruption Severity Index (0-100)."""
    delay_weight: float = 0.40
    bottleneck_weight: float = 0.30
    macro_stress_weight: float = 0.15
    news_sentiment_weight: float = 0.15

# Global instances
MODEL_CONFIG = ModelConfig()
NLP_CONFIG = NLPConfig()
MACRO_CONFIG = MacroConfig()
RISK_WEIGHTS = RiskScoringWeights()
