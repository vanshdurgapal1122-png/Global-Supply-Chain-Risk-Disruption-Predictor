"""Data ingestion package for shipping, macro indicators, and news feeds."""
from src.data_ingestion.macro_fetcher import MacroDataFetcher
from src.data_ingestion.news_fetcher import NewsDataFetcher
from src.data_ingestion.shipping_data import ShippingDataGenerator

__all__ = ["MacroDataFetcher", "NewsDataFetcher", "ShippingDataGenerator"]
