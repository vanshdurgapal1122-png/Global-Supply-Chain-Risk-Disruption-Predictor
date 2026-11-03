"""
Unit tests for data ingestion modules.
Uses Python's standard unittest framework.
"""

import unittest
from src.data_ingestion.shipping_data import ShippingDataGenerator
from src.data_ingestion.macro_fetcher import MacroDataFetcher
from src.data_ingestion.news_fetcher import NewsDataFetcher


class TestDataIngestion(unittest.TestCase):

    def test_shipping_data_generator(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ShippingDataGenerator(raw_data_dir=Path(tmpdir))
            df = generator.generate_dataset(n_samples=50, random_seed=123)

            self.assertEqual(len(df), 50)
            self.assertIn("route_delay_days", df.columns)
            self.assertIn("port_bottleneck_risk", df.columns)
            self.assertIn("disruption_severity_index", df.columns)
            self.assertGreaterEqual(df["route_delay_days"].min(), 0.0)
            self.assertLessEqual(df["disruption_severity_index"].max(), 100.0)

    def test_macro_data_fetcher(self):
        fetcher = MacroDataFetcher()
        features = fetcher.get_latest_macro_features()

        self.assertIn("crude_oil_brent", features)
        self.assertIn("fuel_stress_index", features)
        self.assertGreater(features["fuel_stress_index"], 0.0)

    def test_news_data_fetcher(self):
        fetcher = NewsDataFetcher()
        news = fetcher.get_all_news()

        self.assertGreaterEqual(len(news), 5)
        first = news[0]
        self.assertIn("title", first)
        self.assertIn("disruption_score", first)


if __name__ == "__main__":
    unittest.main()
