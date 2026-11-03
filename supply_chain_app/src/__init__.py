"""
Supply Chain Intelligence Core Source Package.
Exposes data loaders, FinBERT NLP engine, and XGBoost ML modeling engine.
"""

from supply_chain_app.src.data_loader import (
    generate_shipping_data,
    fetch_macro_data,
    save_dataset,
    DEFAULT_OUTPUT_FILE
)
from supply_chain_app.src.nlp_engine import NLPEngine, get_nlp_engine
from supply_chain_app.src.ml_engine import MLEngine, get_ml_engine

__all__ = [
    "generate_shipping_data",
    "fetch_macro_data",
    "save_dataset",
    "DEFAULT_OUTPUT_FILE",
    "NLPEngine",
    "get_nlp_engine",
    "MLEngine",
    "get_ml_engine"
]
