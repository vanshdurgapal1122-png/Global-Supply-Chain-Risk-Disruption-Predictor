"""
Natural Language Processing (NLP) Engine for Supply Chain Disruption Sentiment.
Loads HuggingFace 'ProsusAI/finbert' to extract negative sentiment probabilities as
disruption scores, with robust keyword-based rule fallback for low-resource environments.
"""

import os
import re
import ssl
import logging
from typing import Dict, List, Any, Optional

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NLPEngine")

# Suppress noisy external logging
for lib in ["httpx", "huggingface_hub.utils._http", "huggingface_hub.repocard"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# High-risk logistics disruption keywords and recovery terms for fallback
DISRUPTION_KEYWORDS = {
    "strike": 0.85, "backlog": 0.75, "blockage": 0.90, "blocked": 0.90,
    "delay": 0.70, "delayed": 0.70, "congestion": 0.75, "bottleneck": 0.80,
    "closure": 0.85, "closed": 0.85, "attack": 0.90, "missile": 0.95,
    "grounding": 0.85, "grounded": 0.85, "crisis": 0.85, "halted": 0.85,
    "rerouted": 0.80, "drought": 0.75, "storm": 0.70, "typhoon": 0.85,
    "accident": 0.75, "disruption": 0.80, "disrupted": 0.80, "vessel backlog": 0.85
}

RECOVERY_KEYWORDS = {
    "smoothly": 0.80, "ahead of schedule": 0.90, "cleared": 0.80, "reopened": 0.85,
    "resumed": 0.80, "recovery": 0.75, "efficient": 0.75, "on time": 0.80,
    "resolved": 0.80, "agreement": 0.75, "normal": 0.70, "record high volume": 0.65
}


def fallback_rule_based_score(headline: str) -> Dict[str, Any]:
    """
    Fallback heuristic rule-based disruption scoring when PyTorch/Transformers are unavailable.
    """
    text_lower = headline.lower()
    
    # Calculate match weights
    disruption_hits = [weight for kw, weight in DISRUPTION_KEYWORDS.items() if kw in text_lower]
    recovery_hits = [weight for kw, weight in RECOVERY_KEYWORDS.items() if kw in text_lower]

    disruption_intensity = max(disruption_hits) if disruption_hits else 0.0
    recovery_intensity = max(recovery_hits) if recovery_hits else 0.0

    if disruption_hits and not recovery_hits:
        disruption_score = min(1.0, 0.40 + disruption_intensity * 0.60)
        sentiment = "negative"
    elif recovery_hits and not disruption_hits:
        disruption_score = max(0.0, 0.20 - recovery_intensity * 0.20)
        sentiment = "positive"
    elif disruption_hits and recovery_hits:
        diff = disruption_intensity - recovery_intensity
        disruption_score = float(np.clip(0.50 + diff * 0.40, 0.05, 0.95))
        sentiment = "negative" if diff > 0 else "neutral"
    else:
        disruption_score = 0.25
        sentiment = "neutral"

    return {
        "disruption_score": round(float(disruption_score), 4),
        "sentiment": sentiment,
        "method": "Rule_Based_Fallback"
    }


class NLPEngine:
    """
    NLP Disruption Engine utilizing HuggingFace ProsusAI/finbert.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", use_transformer: bool = True):
        self.model_name = model_name
        self.use_transformer = use_transformer
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.model_loaded = False

        if self.use_transformer:
            self.load_model()

    def load_model(self):
        """Loads ProsusAI/finbert tokenizer and model with SSL and Windows safeguards."""
        try:
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            ssl._create_default_https_context = ssl._create_unverified_context
            
            try:
                import httpx
                from huggingface_hub.utils import _http
                _http.set_client_factory(lambda: httpx.Client(verify=False))
            except Exception:
                pass

            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading pre-trained HuggingFace model '{self.model_name}' on {'CUDA' if device == 0 else 'CPU'}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                top_k=None
            )
            self.model_loaded = True
            logger.info(f"Successfully loaded '{self.model_name}'.")
        except Exception as ex:
            logger.warning(
                f"Failed to load transformer model '{self.model_name}' ({ex}). "
                f"Falling back to rule-based keyword matching."
            )
            self.model_loaded = False

    def compute_disruption_score(self, headline: str) -> float:
        """
        Passes news text through FinBERT to extract probability of negative sentiment.
        Returns normalized disruption_score between 0.0 (low risk) and 1.0 (extreme disruption).
        """
        if not headline or not headline.strip():
            return 0.0

        if self.model_loaded and self.pipeline is not None:
            try:
                # Truncate to maximum standard BERT length
                truncated = headline.strip()[:512]
                preds = self.pipeline(truncated)[0]
                
                # FinBERT outputs classes: 'positive', 'negative', 'neutral'
                label_scores = {p["label"].lower(): float(p["score"]) for p in preds}
                neg_prob = label_scores.get("negative", 0.0)
                pos_prob = label_scores.get("positive", 0.0)
                neutral_prob = label_scores.get("neutral", 0.0)

                # In logistics context, negative tone indicates disruption shock.
                # Blend negative probability with an offset for neutral/positive dampening.
                disruption_score = neg_prob * 1.0 + neutral_prob * 0.15 - pos_prob * 0.10
                disruption_score = float(np.clip(disruption_score, 0.0, 1.0))
                return round(disruption_score, 4)
            except Exception as e:
                logger.error(f"Inference error with FinBERT: {e}. Switching to rule-based fallback.")

        # Fallback path
        fb = fallback_rule_based_score(headline)
        return fb["disruption_score"]

    def analyze_headline(self, headline: str) -> Dict[str, Any]:
        """Detailed analysis returning sentiment label, probabilities, and disruption score."""
        score = self.compute_disruption_score(headline)
        method = "FinBERT_Transformer" if self.model_loaded else "Rule_Based_Fallback"
        return {
            "headline": headline,
            "disruption_score": score,
            "method": method
        }


# Singleton instance for high efficiency across modules
_nlp_engine_instance: Optional[NLPEngine] = None


def get_nlp_engine() -> NLPEngine:
    global _nlp_engine_instance
    if _nlp_engine_instance is None:
        _nlp_engine_instance = NLPEngine()
    return _nlp_engine_instance


def compute_disruption_score(headline: str) -> float:
    """Convenience function to compute disruption score using the global NLPEngine."""
    engine = get_nlp_engine()
    return engine.compute_disruption_score(headline)


if __name__ == "__main__":
    print("=" * 75)
    print("   GLOBAL SUPPLY CHAIN PREDICTOR: NLP ENGINE VERIFICATION (STEP 3)")
    print("=" * 75)

    test_headlines = [
        "Dockworkers strike causes 5-day vessel backlog at Rotterdam",
        "Port of Singapore operates smoothly ahead of schedule",
        "Red Sea missile attacks force container ships to reroute around Cape of Good Hope",
        "Panama Canal announces normal transit slots as seasonal rains replenish Gatun Lake",
        "Severe typhoon approaches Ningbo-Zhoushan, terminal operations suspended",
        "Global container freight rates stabilize as port congestion clears across European hubs"
    ]

    engine = get_nlp_engine()
    print(f"\nModel Loaded State: {'FinBERT Transformer' if engine.model_loaded else 'Rule-Based Fallback'}\n")

    print(f"{'DISRUPTION SCORE':<18} | {'METHOD':<22} | {'HEADLINE'}")
    print("-" * 75)
    for hl in test_headlines:
        res = engine.analyze_headline(hl)
        score = res["disruption_score"]
        method = res["method"]
        print(f"{score:<18.4f} | {method:<22} | {hl}")

    print("=" * 75)
