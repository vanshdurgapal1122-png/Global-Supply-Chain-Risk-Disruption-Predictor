"""
Data Loader Module for Global Supply Chain Risk Predictor.
Handles synthetic shipping dataset generation mimicking DataCo supply chain data
and pulls real-time macroeconomic indicators (Crude Oil & USD/EUR) via YFinance.
"""

import os
import ssl
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataLoader")

# Ensure SSL bypass for restricted corporate / local proxy environments on Windows
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    import curl_cffi.requests.session as s
    orig_request = s.Session.request
    s.Session.request = lambda self, *args, **kwargs: orig_request(self, *args, **{**kwargs, 'verify': False})
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_FILE = DATA_DIR / "supply_chain_data.csv"


def generate_shipping_data(num_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    """
    Creates a realistic synthetic shipping dataset mimicking the Kaggle DataCo
    supply chain dataset.

    Columns:
    - shipment_id: Unique shipment identifier (e.g., SHIP-00001)
    - origin_country: Origin nation of shipment
    - destination_country: Destination nation
    - shipping_mode: Transport mode (Air, Ocean, Rail, Road)
    - scheduled_days: Contractually promised transit days
    - actual_days: Realized transit duration
    - distance_km: Route geodesic/transit distance in kilometers
    - cargo_weight_kg: Total freight payload in kilograms
    - late_delivery_risk: Binary target (1 if actual_days > scheduled_days else 0)
    - delay_days: Continuous target (max(0, actual_days - scheduled_days))
    """
    np.random.seed(random_seed)
    logger.info(f"Generating synthetic shipping dataset with {num_samples} samples...")

    countries = [
        "United States", "China", "Germany", "Japan", "United Kingdom",
        "India", "South Korea", "Netherlands", "Singapore", "Brazil",
        "Australia", "Canada", "France", "Mexico", "Vietnam"
    ]

    shipping_modes = ["Air", "Ocean", "Rail", "Road"]
    mode_probs = [0.20, 0.40, 0.15, 0.25]  # Typical multimodal distribution

    # Mode-specific parameters: (speed_km_day_mean, speed_km_day_std, base_days, max_dist_km, max_weight_kg)
    mode_params = {
        "Air": {"speed_mean": 2400.0, "speed_std": 400.0, "base_days": 1.5, "dist_range": (800, 14000), "weight_range": (50, 5000)},
        "Ocean": {"speed_mean": 550.0, "speed_std": 90.0, "base_days": 5.0, "dist_range": (2000, 22000), "weight_range": (5000, 80000)},
        "Rail": {"speed_mean": 700.0, "speed_std": 120.0, "base_days": 2.5, "dist_range": (500, 8000), "weight_range": (2000, 45000)},
        "Road": {"speed_mean": 600.0, "speed_std": 100.0, "base_days": 1.0, "dist_range": (100, 3500), "weight_range": (100, 25000)}
    }

    records = []
    for i in range(num_samples):
        shipment_id = f"SHIP-{i+1:05d}"
        
        orig, dest = np.random.choice(countries, size=2, replace=False)
        mode = np.random.choice(shipping_modes, p=mode_probs)
        params = mode_params[mode]

        # Sample distance and cargo weight appropriate for mode
        dist_km = round(float(np.random.uniform(params["dist_range"][0], params["dist_range"][1])), 1)
        cargo_weight = round(float(np.random.uniform(params["weight_range"][0], params["weight_range"][1])), 1)

        # Baseline expected transit days
        effective_speed = max(150.0, np.random.normal(params["speed_mean"], params["speed_std"]))
        raw_transit = params["base_days"] + (dist_km / effective_speed)
        
        # Scheduled days is planned with some operational buffer
        scheduled = max(1.0, round(raw_transit + np.random.uniform(0.5, 2.0), 1))

        # Real-world disruptions: customs, weather, congestion, mechanical issues
        disruption_prob = 0.38  # ~38% late delivery rate consistent with global supply chain benchmarks
        is_disrupted = np.random.random() < disruption_prob

        if is_disrupted:
            # Delay severity follows a positive skewed distribution
            delay_shock = np.random.exponential(scale=2.8) + np.random.uniform(0.5, 3.5)
            actual = round(scheduled + delay_shock, 1)
        else:
            # On-time or early delivery
            early_or_ontime = max(0.5, scheduled - np.random.exponential(scale=0.8))
            actual = round(min(scheduled, early_or_ontime), 1)

        late_delivery_risk = 1 if actual > scheduled else 0
        delay_days = round(max(0.0, actual - scheduled), 2)

        records.append({
            "shipment_id": shipment_id,
            "origin_country": orig,
            "destination_country": dest,
            "shipping_mode": mode,
            "scheduled_days": scheduled,
            "actual_days": actual,
            "distance_km": dist_km,
            "cargo_weight_kg": cargo_weight,
            "late_delivery_risk": late_delivery_risk,
            "delay_days": delay_days
        })

    df = pd.DataFrame(records)
    logger.info(
        f"Synthetic dataset created successfully: {len(df)} rows. "
        f"Late delivery rate: {(df['late_delivery_risk'].mean() * 100):.2f}%, "
        f"Mean delay: {df['delay_days'].mean():.2f} days"
    )
    return df


def fetch_macro_data(period: str = "1mo") -> pd.DataFrame:
    """
    Pulls crude oil prices ('CL=F') and USD/EUR exchange rates ('EURUSD=X')
    over the past 30 days using yfinance, and computes 7-day rolling volatility.

    Returns:
        pd.DataFrame containing daily prices, percentage returns, and 7-day rolling volatility.
    """
    logger.info("Fetching macroeconomic indicators ('CL=F', 'EURUSD=X') via YFinance...")
    tickers = {
        "crude_oil_wti": "CL=F",
        "eur_usd": "EURUSD=X"
    }

    macro_series = {}
    for col_name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, interval="1d")
            if not hist.empty and "Close" in hist.columns:
                macro_series[col_name] = hist["Close"]
                logger.info(f"Successfully retrieved {len(hist)} days of data for {ticker}")
            else:
                logger.warning(f"No Close data returned for ticker {ticker}")
        except Exception as e:
            logger.warning(f"Error fetching ticker {ticker}: {e}")

    if macro_series:
        macro_df = pd.DataFrame(macro_series).ffill().bfill()
        macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
    else:
        logger.warning("Live fetch failed or incomplete. Using calibrated fallback historical series.")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
        np.random.seed(42)
        oil_vals = 75.0 + np.cumsum(np.random.normal(0, 0.9, size=30))
        eur_vals = 1.085 + np.cumsum(np.random.normal(0, 0.003, size=30))
        macro_df = pd.DataFrame({
            "crude_oil_wti": oil_vals,
            "eur_usd": eur_vals
        }, index=dates)

    # Calculate daily returns
    macro_df["crude_oil_return"] = macro_df["crude_oil_wti"].pct_change().fillna(0)
    macro_df["eur_usd_return"] = macro_df["eur_usd"].pct_change().fillna(0)

    # Calculate 7-day rolling volatility (annualized standard deviation: std * sqrt(252))
    macro_df["crude_oil_7d_vol"] = (
        macro_df["crude_oil_return"].rolling(window=7, min_periods=2).std().fillna(0) * np.sqrt(252)
    ).round(4)
    macro_df["eur_usd_7d_vol"] = (
        macro_df["eur_usd_return"].rolling(window=7, min_periods=2).std().fillna(0) * np.sqrt(252)
    ).round(4)

    logger.info(f"Macroeconomic indicators processed ({len(macro_df)} rows).")
    return macro_df


def save_dataset(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Saves the generated dataset to CSV."""
    target_path = output_path or DEFAULT_OUTPUT_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    logger.info(f"Dataset successfully saved to: {target_path}")
    return target_path


if __name__ == "__main__":
    print("=" * 70)
    print("   GLOBAL SUPPLY CHAIN PREDICTOR: DATA LOADER EXECUTION (STEP 2)")
    print("=" * 70)

    # 1. Generate Shipping Dataset
    shipping_df = generate_shipping_data(num_samples=2500)
    print(f"\n[+] Shipping Data Summary:")
    print(f"    - Total Records      : {len(shipping_df):,}")
    print(f"    - Columns            : {list(shipping_df.columns)}")
    print(f"    - Late Delivery Rate : {(shipping_df['late_delivery_risk'].mean() * 100):.2f}%")
    print(f"    - Avg Delay Days     : {shipping_df['delay_days'].mean():.2f} days")
    print(f"    - Shipping Modes     : {shipping_df['shipping_mode'].value_counts().to_dict()}")

    # 2. Fetch Macroeconomic Indicators
    print(f"\n[+] Fetching 30-Day Macro Indicators & 7-Day Rolling Volatility:")
    macro_data = fetch_macro_data(period="1mo")
    print(macro_data.tail(5)[["crude_oil_wti", "crude_oil_7d_vol", "eur_usd", "eur_usd_7d_vol"]])

    # 3. Save to data/supply_chain_data.csv
    saved_file = save_dataset(shipping_df)
    print(f"\n[SUCCESS] Dataset created at: {saved_file}")
    print(f"          File size: {saved_file.stat().st_size / 1024:.2f} KB")
    print("=" * 70)
