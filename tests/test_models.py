"""
Unit tests for ML predictor and feature pipeline.
Uses Python's standard unittest framework.
"""

import unittest
from src.models.predictor import SupplyChainPredictor


class TestPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = SupplyChainPredictor()

    def test_predictor_loaded(self):
        self.assertTrue(self.predictor.is_loaded)
        self.assertIsNotNone(self.predictor.model_delay)
        self.assertIsNotNone(self.predictor.model_bottleneck)
        self.assertIsNotNone(self.predictor.model_dsi)

    def test_single_voyage_prediction(self):
        pred = self.predictor.predict_voyage(
            origin_port="CNSHA",
            dest_port="NLRTM",
            vessel_class="New_Panamax",
            rerouted_via_cape=True,
            brent_oil_price=90.0,
            news_disruption_sentiment=0.80
        )

        self.assertIn("predicted_delay_days", pred)
        self.assertGreater(pred["predicted_delay_days"], 5.0)  # Rerouted via Cape causes significant delay
        self.assertLessEqual(pred["delay_ci_95"][0], pred["predicted_delay_days"])
        self.assertLessEqual(pred["predicted_delay_days"], pred["delay_ci_95"][1])
        self.assertIn(pred["bottleneck_risk"], ["Low", "Medium", "High", "Critical"])
        self.assertGreaterEqual(pred["disruption_severity_index"], 0.0)
        self.assertLessEqual(pred["disruption_severity_index"], 100.0)

    def test_stress_test_scenarios(self):
        scenarios = self.predictor.run_stress_test_scenarios()

        self.assertEqual(len(scenarios), 5)
        # The compound crisis scenario should have higher delay than baseline
        baseline = scenarios[0]
        compound = scenarios[-1]
        self.assertGreater(compound["predicted_delay_days"], baseline["predicted_delay_days"])
        self.assertGreater(compound["dsi"], baseline["dsi"])


if __name__ == "__main__":
    unittest.main()
