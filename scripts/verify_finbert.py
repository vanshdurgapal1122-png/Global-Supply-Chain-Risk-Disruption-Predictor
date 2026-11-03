"""Quick verification of full FinBERT transformer inference pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlp.sentiment_analyzer import get_sentiment_analyzer

analyzer = get_sentiment_analyzer(use_transformer=True)
headlines = [
    "Houthi drone attacks on Red Sea shipping routes force massive rerouting around Cape of Good Hope.",
    "Panama Canal fully reopens with record-high transit capacity after rains restore lake levels.",
    "ILA port workers launch nationwide strike across US East Coast container terminals."
]
results = analyzer.analyze_batch(headlines)
for r in results:
    sentiment = r["sentiment"].upper()
    disruption = r["disruption_score"]
    confidence = r["confidence"]
    method = r["method"]
    entities = r["entities"]
    categories = r["categories"]
    print(f"Sentiment: {sentiment:10s} | Disruption: {disruption:.3f} | Confidence: {confidence:.3f} | Method: {method}")
    print(f"  Entities: {entities}  |  Categories: {categories}")
    print()
