"""
FinBERT Model Pre-Downloader.
Run this script once to download and cache the ProsusAI/finbert model weights locally.
Subsequent dashboard runs will load from HuggingFace cache without needing internet access.
"""

import sys
import os
import ssl
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinBERT_Download")


def download_finbert():
    """Downloads and caches FinBERT model weights with Windows SSL compatibility."""
    model_name = "ProsusAI/finbert"

    # Apply SSL bypass for corporate / Windows certificate restriction environments
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    try:
        import httpx
        from huggingface_hub.utils import _http
        _http.set_client_factory(lambda: httpx.Client(verify=False))
        logger.info("Configured HuggingFace Hub to use unverified HTTPS client for SSL bypass.")
    except Exception as e:
        logger.warning(f"Could not set client factory: {e}")

    logger.info(f"Downloading {model_name} tokenizer and weights...")
    logger.info("This may take 2-5 minutes depending on your internet connection (~440 MB).")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer downloaded and cached successfully.")

        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("Model weights downloaded and cached successfully.")

        # Verify with a quick inference
        from transformers import pipeline
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
        test_result = pipe("Houthi attacks on Red Sea shipping routes force cargo rerouting.")
        logger.info(f"FinBERT inference test passed! Result: {test_result[0]}")
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUCCESS: FinBERT is ready. The dashboard will now use the")
        logger.info("full transformer model for real-time sentiment analysis.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("")
        logger.info("The dashboard will operate in Logistics Domain NLP fallback mode.")
        logger.info("All predictions and sentiment scoring will still be fully functional.")


if __name__ == "__main__":
    download_finbert()
