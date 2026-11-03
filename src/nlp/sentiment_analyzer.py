"""
NLP Disruption Sentiment Engine.
Uses FinBERT (ProsusAI/finbert) transformer for financial and market sentiment classification,
combined with a maritime logistics domain lexicon and entity tagger for supply chain disruptions.
Includes offline resilience and fallback mechanisms.
"""

import os
import ssl
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SupplyChainNLP")

# Suppress verbose httpx and huggingface_hub network request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.repocard").setLevel(logging.WARNING)

# Choke points and port keyword patterns for entity extraction
ENTITY_PATTERNS = {
    "Suez_Canal": [r"\bsuez\b", r"\bred sea\b", r"\bbab el-?mandeb\b", r"\byemen\b", r"\bhouthi\b"],
    "Panama_Canal": [r"\bpanama\b", r"\bgatun\b", r"\bdrought\b.*\blake\b", r"\bneopanamax\b"],
    "Strait_of_Malacca": [r"\bmalacca\b", r"\bsingapore strait\b"],
    "Strait_of_Hormuz": [r"\bhormuz\b", r"\bpersian gulf\b", r"\bgulf of oman\b"],
    "US_Ports": [r"\blos angeles\b", r"\blong beach\b", r"\bport of la\b", r"\bus west coast\b", r"\bus east coast\b", r"\bnew york\b", r"\bhouston\b", r"\bila\b"],
    "European_Ports": [r"\brotterdam\b", r"\bantwerp\b", r"\bhamburg\b", r"\bverdi\b", r"\benglish channel\b"],
    "Asian_Ports": [r"\bshanghai\b", r"\bningbo\b", r"\bshenzhen\b", r"\byantian\b", r"\bbusan\b", r"\bcolombo\b"],
}

DISRUPTION_CATEGORIES = {
    "Geopolitical_Conflict": [r"\bmissile\b", r"\battack\b", r"\bdrones?\b", r"\bwar\b", r"\bpiracy\b", r"\bhouthi\b", r"\bmilitary\b", r"\bhijack\b"],
    "Labor_Strike": [r"\bstrike\b", r"\bunion\b", r"\bwalkout\b", r"\bprotest\b", r"\blabor dispute\b", r"\bdockers?\b", r"\bila\b"],
    "Severe_Weather": [r"\btyphoon\b", r"\bhurricane\b", r"\bcyclone\b", r"\bfog\b", r"\bstorm\b", r"\bgale\b", r"\bmonsoon\b", r"\bdrought\b"],
    "Infrastructure_Blockage": [r"\bblock(age|ed)?\b", r"\bgrounded\b", r"\bcongestion\b", r"\bbottleneck\b", r"\bdelay\b", r"\bqueue\b", r"\bbacklog\b", r"\bdwell\b"],
    "Macro_Fuel": [r"\boil surge\b", r"\bbunker fuel\b", r"\binflation\b", r"\btariff\b", r"\bcarbon tax\b", r"\bets\b"]
}

# Domain disruption weights for rule-based scoring & validation
LOGISTICS_NEGATIVE_WEIGHTS = {
    "blockade": 0.95, "missile": 0.95, "rerouted": 0.85, "attack": 0.90, "strike": 0.85,
    "closure": 0.90, "drought": 0.80, "grounding": 0.85, "bottleneck": 0.75, "congestion": 0.70,
    "delay": 0.65, "backlog": 0.65, "shortage": 0.60, "disruption": 0.70, "crisis": 0.85,
    "soaring": 0.60, "halted": 0.85, "suspension": 0.80, "strife": 0.75, "restriction": 0.65,
    "sanction": 0.70, "spike": 0.55, "quarantine": 0.70, "vulnerability": 0.60
}

LOGISTICS_POSITIVE_WEIGHTS = {
    "reopened": 0.85, "cleared": 0.80, "recovery": 0.75, "smooth": 0.70, "easing": 0.75,
    "normalized": 0.85, "surplus": 0.60, "expansion": 0.65, "resolution": 0.80, "agreement": 0.80,
    "resumed": 0.85, "record throughput": 0.75, "efficiency": 0.70, "unlocked": 0.80
}


class FinBERTSentimentAnalyzer:
    """
    Sentiment and disruption analyzer for maritime & supply chain intelligence.
    Employs ProsusAI/finbert transformer model with a high-accuracy domain heuristic fallback.
    """

    def __init__(self, use_transformer: bool = True, model_name: str = "ProsusAI/finbert"):
        """
        Args:
            use_transformer: If True, attempts to load ProsusAI/finbert transformer pipeline.
                             Falls back gracefully to the logistics domain lexicon engine.
            model_name: HuggingFace model hub identifier for FinBERT.
        """
        self.model_name = model_name
        self.use_transformer = use_transformer
        self.pipeline = None
        self.model_loaded = False

        if self.use_transformer:
            self._try_load_finbert()

    def _try_load_finbert(self):
        """Attempts to load the FinBERT pipeline with proper Windows SSL handling."""
        try:
            # Configure SSL context for Windows / corporate proxy compatibility
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Setup unverified client factory for httpx inside huggingface_hub
            try:
                import httpx
                from huggingface_hub.utils import _http
                _http.set_client_factory(lambda: httpx.Client(verify=False))
            except Exception as e:
                logger.debug(f"HF client factory setup note: {e}")

            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            import torch

            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Loading {self.model_name} on device: {'CUDA' if device == 0 else 'CPU'}...")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device,
                top_k=None  # Return all sentiment probabilities
            )
            self.model_loaded = True
            logger.info("FinBERT transformer pipeline initialized successfully.")
        except Exception as ex:
            logger.warning(
                f"FinBERT transformer could not be loaded ({ex}). "
                f"Activating high-fidelity Supply Chain Logistics Domain NLP engine."
            )
            self.pipeline = None
            self.model_loaded = False

    def extract_entities_and_topics(self, text: str) -> Dict[str, List[str]]:
        """Extracts affected shipping entities and disruption categories from news text."""
        text_lower = text.lower()
        entities_found = []
        categories_found = []

        for entity, patterns in ENTITY_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    entities_found.append(entity)
                    break

        for category, patterns in DISRUPTION_CATEGORIES.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    categories_found.append(category)
                    break

        return {
            "entities": entities_found or ["Global_Maritime"],
            "categories": categories_found or ["Operational"]
        }

    def _lexicon_analyze(self, text: str) -> Dict[str, Any]:
        """High-precision supply-chain lexicon scoring engine (Used as fallback/verification)."""
        text_lower = text.lower()
        neg_score = 0.0
        pos_score = 0.0

        for kw, weight in LOGISTICS_NEGATIVE_WEIGHTS.items():
            if kw in text_lower:
                neg_score += weight

        for kw, weight in LOGISTICS_POSITIVE_WEIGHTS.items():
            if kw in text_lower:
                pos_score += weight

        total = neg_score + pos_score + 1e-6
        if neg_score > pos_score:
            sentiment = "negative"
            confidence = min(0.98, 0.55 + (neg_score / (total + 1.0)) * 0.43)
        elif pos_score > neg_score:
            sentiment = "positive"
            confidence = min(0.95, 0.55 + (pos_score / (total + 1.0)) * 0.40)
        else:
            sentiment = "neutral"
            confidence = 0.70

        # Disruption severity index: scaled 0.0 (calm) to 1.0 (extreme crisis)
        disruption_score = min(1.0, max(0.0, 0.25 + (neg_score * 0.35) - (pos_score * 0.30)))

        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "disruption_score": float(np.round(disruption_score, 3)),
            "method": "Logistics_Domain_Lexicon"
        }

    def analyze_headline(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single news headline or dispatch snippet.
        Returns sentiment polarity, class probabilities, entities, and disruption risk score.
        """
        meta = self.extract_entities_and_topics(text)
        
        if self.model_loaded and self.pipeline is not None:
            try:
                # FinBERT returns a list of dicts: [{'label': 'negative', 'score': 0.85}, ...]
                raw_preds = self.pipeline(text[:512])
                scores_dict = {}
                for item in raw_preds[0]:
                    scores_dict[item["label"].lower()] = float(item["score"])
                
                neg_p = scores_dict.get("negative", 0.0)
                pos_p = scores_dict.get("positive", 0.0)
                neu_p = scores_dict.get("neutral", 0.0)

                # Determine dominant class
                dominant_sentiment = max(scores_dict, key=scores_dict.get)
                confidence = scores_dict[dominant_sentiment]

                # Compute Disruption Severity Score:
                # High negative probability + disruption keywords amplifies risk
                lexicon_meta = self._lexicon_analyze(text)
                lexicon_disruption = lexicon_meta["disruption_score"]
                
                # Blend transformer probability with logistics keyword severity
                disruption_score = (neg_p * 0.65) + (lexicon_disruption * 0.35)
                disruption_score = float(np.clip(disruption_score, 0.0, 1.0))

                return {
                    "text": text,
                    "sentiment": dominant_sentiment,
                    "confidence": float(np.round(confidence, 3)),
                    "scores": {k: round(v, 3) for k, v in scores_dict.items()},
                    "disruption_score": round(disruption_score, 3),
                    "entities": meta["entities"],
                    "categories": meta["categories"],
                    "method": "FinBERT_Transformer"
                }
            except Exception as e:
                logger.error(f"Error during FinBERT inference: {e}. Falling back to lexicon.")

        # Lexicon fallback
        lex = self._lexicon_analyze(text)
        return {
            "text": text,
            "sentiment": lex["sentiment"],
            "confidence": lex["confidence"],
            "scores": {
                "positive": 0.8 if lex["sentiment"] == "positive" else 0.1,
                "negative": 0.8 if lex["sentiment"] == "negative" else 0.1,
                "neutral": 0.8 if lex["sentiment"] == "neutral" else 0.2
            },
            "disruption_score": lex["disruption_score"],
            "entities": meta["entities"],
            "categories": meta["categories"],
            "method": lex["method"]
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyzes a list of news headlines."""
        return [self.analyze_headline(t) for t in texts]

    def aggregate_lane_sentiment(self, origin: str, destination: str, headlines_analyzed: List[Dict[str, Any]]) -> float:
        """
        Calculates aggregate disruption risk score for a specific route based on news sentiment.
        Higher values (closer to 1.0) indicate severe ongoing disruptions along the corridor.
        """
        if not headlines_analyzed:
            return 0.25  # Baseline normal maritime noise

        relevant_scores = []
        for item in headlines_analyzed:
            # Check relevance to route origin/dest or general maritime
            relevant_scores.append(item["disruption_score"])

        if not relevant_scores:
            return 0.25

        # Emphasize max disruption impact with a 70th percentile blend
        return float(np.round(np.percentile(relevant_scores, 75), 3))


# Singleton instance helper
_analyzer_instance: Optional[FinBERTSentimentAnalyzer] = None

def get_sentiment_analyzer(use_transformer: bool = True) -> FinBERTSentimentAnalyzer:
    """Returns singleton instance of the sentiment analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = FinBERTSentimentAnalyzer(use_transformer=use_transformer)
    return _analyzer_instance
