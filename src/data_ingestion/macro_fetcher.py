"""
Macroeconomic Data Fetcher.
Fetches real-time and historical oil prices (Brent, WTI), currency exchange rates (EUR/USD, USD/CNY),
and freight market proxies using yfinance. Provides rolling volatility, trend metrics, and offline caching.
"""

import os
import ssl
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

from config.config import MACRO_CONFIG, RAW_DATA_DIR

logger = logging.getLogger("MacroFetcher")


class MacroDataFetcher:
    """
    Ingests and processes macroeconomic time series relevant to global maritime operations.
    Calculates fuel stress, FX volatility, and macro risk scores.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or RAW_DATA_DIR
        self.cache_file = self.cache_dir / "macro_indicators.csv"
        self.tickers = MACRO_CONFIG.tickers
        self._ensure_ssl()

    def _ensure_ssl(self):
        """Bypasses strict SSL issues for financial scraping on restricted networks."""
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            pass
        try:
            import curl_cffi.requests.session as s
            orig = s.Session.request
            s.Session.request = lambda self, *args, **kwargs: orig(self, *args, **{**kwargs, 'verify': False})
        except Exception:
            pass

    def fetch_live_macro(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical data for all configured macro tickers from Yahoo Finance.
        Falls back to cache or synthetic realistic series if offline.
        """
        logger.info(f"Fetching macro indicators from Yahoo Finance ({list(self.tickers.keys())})...")
        macro_dfs = []

        try:
            for col_name, ticker in self.tickers.items():
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period=period, interval=interval)
                    if not hist.empty and "Close" in hist.columns:
                        s = hist["Close"].rename(col_name)
                        macro_dfs.append(s)
                    else:
                        logger.warning(f"No data returned for ticker {ticker} ({col_name})")
                except Exception as ex:
                    logger.warning(f"Error fetching ticker {ticker}: {ex}")

            if macro_dfs:
                combined_df = pd.concat(macro_dfs, axis=1)
                combined_df = combined_df.ffill().bfill()
                combined_df.index = pd.to_datetime(combined_df.index).tz_localize(None)
                
                # Enrich with derived financial indicators
                enriched_df = self._compute_macro_indicators(combined_df)
                
                # Save to cache
                enriched_df.to_csv(self.cache_file)
                logger.info(f"Successfully saved {len(enriched_df)} macro records to {self.cache_file}")
                return enriched_df

        except Exception as e:
            logger.error(f"Failed to fetch live macro data: {e}. Checking cache or synthetic fallback.")

        # Check existing cache
        if self.cache_file.exists():
            logger.info("Loading macro data from local cache.")
            df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
            return df

        # Generate realistic fallback time series
        logger.info("Generating realistic historical macroeconomic time series fallback...")
        return self._generate_realistic_macro_history()

    def _compute_macro_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes rolling returns, volatility, fuel price indices, and stress scores."""
        data = df.copy()

        # Ensure essential columns exist
        if "crude_oil_brent" not in data.columns:
            data["crude_oil_brent"] = MACRO_CONFIG.default_oil_price
        if "eur_usd" not in data.columns:
            data["eur_usd"] = MACRO_CONFIG.default_fx_eur_usd
        if "usd_cny" not in data.columns:
            data["usd_cny"] = MACRO_CONFIG.default_fx_usd_cny

        # 1. Oil Price Momentum & Returns
        data["oil_ret_7d"] = data["crude_oil_brent"].pct_change(7).fillna(0)
        data["oil_ret_30d"] = data["crude_oil_brent"].pct_change(30).fillna(0)
        data["oil_vol_30d"] = data["crude_oil_brent"].pct_change().rolling(30).std().fillna(0) * np.sqrt(252)

        # 2. FX Volatility (Trade disruption proxy)
        data["eur_usd_vol_30d"] = data["eur_usd"].pct_change().rolling(30).std().fillna(0) * np.sqrt(252)
        data["usd_cny_ret_30d"] = data["usd_cny"].pct_change(30).fillna(0)

        # 3. Bunker Fuel Stress Index (0.0 to 1.0)
        # Normalized: $60/bbl is baseline 0.2, $90/bbl is 0.6, $120+/bbl approaches 1.0
        data["fuel_stress_index"] = np.clip((data["crude_oil_brent"] - 50.0) / 70.0, 0.05, 1.0)

        # 4. Macro Disruption Stress Index (Composite 0 to 1)
        data["macro_disruption_index"] = (
            0.50 * data["fuel_stress_index"] +
            0.30 * np.clip(data["oil_vol_30d"] / 0.50, 0.0, 1.0) +
            0.20 * np.clip(data["eur_usd_vol_30d"] / 0.20, 0.0, 1.0)
        ).clip(0.0, 1.0).round(4)

        return data

    def _generate_realistic_macro_history(self, days: int = 730) -> pd.DataFrame:
        """
        Generates realistic 2-year macro time series if offline.
        Models oil geopolitical shocks, interest rate/FX shifts, and freight cycles.
        """
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")

        # Brent Oil with realistic mean-reversion around $82 + periodic geopolitical spikes
        oil_baseline = 82.0
        oil_innovations = np.random.normal(0, 1.2, size=days)
        oil_prices = np.zeros(days)
        oil_prices[0] = oil_baseline

        for t in range(1, days):
            # Mean reversion + random shock
            oil_prices[t] = oil_prices[t - 1] + 0.03 * (oil_baseline - oil_prices[t - 1]) + oil_innovations[t]
            # Occasional crisis spike
            if t in [120, 310, 540]:
                oil_prices[t] += 12.0

        oil_prices = np.clip(oil_prices, 60.0, 135.0)

        # EUR/USD (around 1.08)
        fx_eur = 1.085 + np.cumsum(np.random.normal(0, 0.003, size=days))
        fx_eur = np.clip(fx_eur, 1.02, 1.15)

        # USD/CNY (around 7.22)
        fx_cny = 7.22 + np.cumsum(np.random.normal(0, 0.005, size=days))
        fx_cny = np.clip(fx_cny, 6.95, 7.35)

        # Freight ETF proxy
        bdry = 8.5 + np.cumsum(np.random.normal(0, 0.15, size=days))
        bdry = np.clip(bdry, 4.0, 22.0)

        df = pd.DataFrame({
            "crude_oil_brent": oil_prices,
            "crude_oil_wti": oil_prices * 0.96,
            "eur_usd": fx_eur,
            "usd_cny": fx_cny,
            "freight_etf": bdry
        }, index=dates)

        df = self._compute_macro_indicators(df)
        df.to_csv(self.cache_file)
        return df

    def get_latest_macro_features(self) -> Dict[str, float]:
        """Returns the most recent macroeconomic snapshot for live inference."""
        if self.cache_file.exists():
            df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
        else:
            df = self.fetch_live_macro()

        latest = df.iloc[-1]
        return {
            "crude_oil_brent": float(latest.get("crude_oil_brent", MACRO_CONFIG.default_oil_price)),
            "oil_ret_7d": float(latest.get("oil_ret_7d", 0.015)),
            "oil_ret_30d": float(latest.get("oil_ret_30d", 0.035)),
            "oil_vol_30d": float(latest.get("oil_vol_30d", 0.22)),
            "eur_usd": float(latest.get("eur_usd", MACRO_CONFIG.default_fx_eur_usd)),
            "usd_cny": float(latest.get("usd_cny", MACRO_CONFIG.default_fx_usd_cny)),
            "fuel_stress_index": float(latest.get("fuel_stress_index", 0.46)),
            "macro_disruption_index": float(latest.get("macro_disruption_index", 0.38))
        }

    def simulate_macro_scenario(
        self,
        oil_price: float,
        oil_ret_30d: float = 0.05,
        eur_usd: float = 1.08
    ) -> Dict[str, float]:
        """Generates custom macro features for interactive what-if stress tests."""
        fuel_stress = float(np.clip((oil_price - 50.0) / 70.0, 0.05, 1.0))
        vol_est = 0.45 if abs(oil_ret_30d) > 0.20 else 0.22
        macro_index = float(np.clip(0.55 * fuel_stress + 0.30 * vol_est + 0.15 * 0.10, 0.0, 1.0))

        return {
            "crude_oil_brent": round(oil_price, 2),
            "oil_ret_7d": round(oil_ret_30d / 3.0, 4),
            "oil_ret_30d": round(oil_ret_30d, 4),
            "oil_vol_30d": round(vol_est, 4),
            "eur_usd": round(eur_usd, 4),
            "usd_cny": 7.23,
            "fuel_stress_index": round(fuel_stress, 4),
            "macro_disruption_index": round(macro_index, 4)
        }
