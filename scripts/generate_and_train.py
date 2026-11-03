"""
End-to-End Orchestrator Script.
Runs data generation, feature engineering, model training, and metric evaluation in a single command.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure utf-8 stdout encoding for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.data_ingestion.shipping_data import ShippingDataGenerator
from src.data_ingestion.macro_fetcher import MacroDataFetcher
from src.data_ingestion.news_fetcher import NewsDataFetcher
from src.models.train import ModelTrainer
from src.models.predictor import SupplyChainPredictor


def main():
    print("=" * 70)
    print("[*] GLOBAL SUPPLY CHAIN RISK & DISRUPTION PREDICTOR: PIPELINE RUN")
    print("=" * 70)

    # 1. Macro Ingestion
    print("\n[Step 1/5] Ingesting Macroeconomic Indicators via YFinance...")
    macro_fetcher = MacroDataFetcher()
    macro_df = macro_fetcher.fetch_live_macro(period="1y")
    print(f"[+] Macro data synced: {len(macro_df)} records (Latest Brent: ${macro_df['crude_oil_brent'].iloc[-1]:.2f})")

    # 2. News Ingestion & Sentiment
    print("\n[Step 2/5] Initializing Supply Chain News Intelligence & FinBERT NLP...")
    news_fetcher = NewsDataFetcher()
    news = news_fetcher.get_all_news()
    print(f"[+] News dispatches indexed: {len(news)} articles analyzed")

    # 3. Shipping Data Synthesis
    print("\n[Step 3/5] Synthesizing Historical Shipping Logs across Trade Lanes...")
    data_gen = ShippingDataGenerator()
    df = data_gen.generate_dataset(n_samples=5500)
    print(f"[+] Historical shipping logs generated: {len(df)} voyage transit records")

    # 4. Model Training
    print("\n[Step 4/5] Fitting Feature Transformers & Training XGBoost Models...")
    trainer = ModelTrainer()
    metrics = trainer.train_all_models(df)
    print("[+] Model 1: Route Delay Regressor MAE: {:.3f} days (R^2: {:.4f})".format(
        metrics["delay_regressor"]["mae_days"],
        metrics["delay_regressor"]["r2_score"]
    ))
    print("[+] Model 2: Port Bottleneck Classifier Accuracy: {:.2f}% (Weighted F1: {:.4f})".format(
        metrics["bottleneck_classifier"]["accuracy"] * 100,
        metrics["bottleneck_classifier"]["weighted_f1"]
    ))
    print("[+] Model 3: Disruption Severity Regressor R^2: {:.4f}".format(
        metrics["dsi_regressor"]["r2_score"]
    ))

    # 5. Verification Inference
    print("\n[Step 5/5] Executing Validation Sample Inference...")
    predictor = SupplyChainPredictor()
    sample_pred = predictor.predict_voyage(
        origin_port="CNSHA",
        dest_port="NLRTM",
        vessel_class="New_Panamax",
        rerouted_via_cape=True,
        brent_oil_price=92.0,
        news_disruption_sentiment=0.85
    )
    print(f"[+] Test Prediction: Shanghai -> Rotterdam (Cape Reroute)")
    print(f"  * Predicted Route Delay: {sample_pred['predicted_delay_days']} days (95% CI: {sample_pred['delay_ci_95']})")
    print(f"  * Bottleneck Risk: {sample_pred['bottleneck_risk']}")
    print(f"  * Disruption Severity (DSI): {sample_pred['disruption_severity_index']}/100 [{sample_pred['risk_tier']}]")

    print("\n" + "=" * 70)
    print("[SUCCESS] PIPELINE EXECUTION COMPLETE. Launch dashboard using: python -m streamlit run dashboard/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
