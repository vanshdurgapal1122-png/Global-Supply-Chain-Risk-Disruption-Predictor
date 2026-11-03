"""
Unit tests for NLP sentiment and disruption analysis module.
Uses Python's standard unittest framework.
"""

import unittest
from src.nlp.sentiment_analyzer import FinBERTSentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = FinBERTSentimentAnalyzer(use_transformer=False)

    def test_sentiment_analyzer_negative(self):
        headline = "Houthi missile attacks strike container vessel in Bab el-Mandeb, forcing rerouting."
        res = self.analyzer.analyze_headline(headline)

        self.assertEqual(res["sentiment"], "negative")
        self.assertGreater(res["disruption_score"], 0.40)
        self.assertIn("Suez_Canal", res["entities"])
        self.assertIn("Geopolitical_Conflict", res["categories"])

    def test_sentiment_analyzer_positive(self):
        headline = "Panama Canal increases daily transit slots following heavy seasonal rainfall and reservoir recovery."
        res = self.analyzer.analyze_headline(headline)

        self.assertEqual(res["sentiment"], "positive")
        self.assertLess(res["disruption_score"], 0.35)
        self.assertIn("Panama_Canal", res["entities"])

    def test_batch_analysis(self):
        headlines = [
            "Port workers threaten nationwide strike over wages.",
            "Cargo traffic resumed smoothly through European ports."
        ]
        results = self.analyzer.analyze_batch(headlines)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["sentiment"], "negative")
        self.assertEqual(results[1]["sentiment"], "positive")


if __name__ == "__main__":
    unittest.main()
