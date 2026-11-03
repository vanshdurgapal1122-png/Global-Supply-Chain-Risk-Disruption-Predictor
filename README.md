# 🌐 Global Supply Chain Risk & Disruption Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![Transformers](https://img.shields.io/badge/HuggingFace-FinBERT-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end Machine Learning system that predicts **maritime shipping delays**, **port bottleneck risks**, and **composite disruption severity** by fusing:
1. **Historical Shipping Transit Logs** (AIS routes, vessel TEU sizes, choke points, seasonal weather, port dwell).
2. **Real-Time Macroeconomic Indicators** (Brent Crude spot, bunker fuel stress, EUR/USD & USD/CNY exchange rate volatility via `yfinance`).
3. **NLP Disruption Sentiment** (Analyzing breaking maritime news headlines and port advisories via `ProsusAI/finbert`).

---

## 🏛️ System Architecture

```
                                  [ MULTI-MODAL DATA SOURCES ]
                                                 │
          ┌──────────────────────────────────────┼──────────────────────────────────────┐
          ▼                                      ▼                                      ▼
[ Historical Shipping Logs ]           [ YFinance Macro Feed ]                [ Real-Time News Stream ]
• 12 Major Global Lanes                • Brent Crude (BZ=F)                   • Maritime Dispatches
• Vessel Classes (1.5k-24k TEU)        • FX Rates (EUR/USD, USD/CNY)          • Port Advisories / Strikes
• Choke Points (Suez, Panama)          • Bunker Fuel Stress Index             • Geopolitical Threats
          │                                      │                                      │
          │                                      │                                      ▼
          │                                      │                            [ FinBERT Transformer ]
          │                                      │                            • Sentiment Polarity
          │                                      │                            • Disruption Risk Score
          │                                      │                            • Entity & Choke Tagger
          ▼                                      ▼                                      ▼
    ═════════════════════════════════════════════════════════════════════════════════════════════
                                MULTI-MODAL FEATURE FUSION PIPELINE
          • Interaction Features: Fuel Distance Pressure, Weather-Canal Interaction, Choke Sentiment
          • Standard Scaling for numerical tensors & One-Hot Encoding for categorical hubs
    ═════════════════════════════════════════════════════════════════════════════════════════════
                                                 │
                        ┌────────────────────────┼────────────────────────┐
                        ▼                        ▼                        ▼
               [ Model 1: XGBoost ]     [ Model 2: XGBoost ]     [ Model 3: XGBoost ]
              Route Delay Regressor    Port Bottleneck Risk     Disruption Severity
                 (Days with 95% CI)       (Low/Med/High/Crit)      Index (DSI: 0-100)
                        │                        │                        │
                        └────────────────────────┼────────────────────────┘
                                                 ▼
                               [ EXECUTIVE STREAMLIT DASHBOARD ]
                     • Interactive Geospatial Folium Maritime Risk Map
                     • Real-Time Voyage Simulation & Confidence Intervals
                     • Macroeconomic Correlation Heatmaps & News Feed Pulse
                     • 5-Way Crisis Scenario Stress Testing Sandbox
```

---

## 📐 Mathematical Formulation

### 1. Route Delay Modeling (Target 1: Regression)
The voyage delay $\Delta_{\text{days}}$ reflects actual transit time minus scheduled baseline:
$$\hat{y}_{\text{delay}} = f_{\text{XGBoost}}(\mathbf{x}_{\text{route}}, \mathbf{x}_{\text{macro}}, \mathbf{x}_{\text{sentiment}}, \mathbf{x}_{\text{interactions}})$$
The system reports a **95% Confidence Interval**:
$$CI_{95\%} = \left[ \max(0, \hat{y} - 1.96 \cdot \text{RMSE}), \; \hat{y} + 1.96 \cdot \text{RMSE} \right]$$

### 2. Port Bottleneck Classification (Target 2: Multi-Class)
Classifies destination port dwell and berthing queues into 4 operational tiers:
- **Low**: Yard dwell $< 3.2$ days, berthing wait $< 24$ hours.
- **Medium**: Yard dwell $3.2 - 5.0$ days, berthing wait $24 - 48$ hours.
- **High**: Yard dwell $5.0 - 8.0$ days, berthing wait $48 - 96$ hours.
- **Critical**: Yard dwell $> 8.0$ days, berthing wait $> 96$ hours.

### 3. Composite Disruption Severity Index (DSI)
A unified risk index $DSI \in [0, 100]$ synthesized across physical, operational, economic, and sentiment stressors:
$$DSI = \left( 0.40 \cdot \tilde{y}_{\text{delay}} + 0.30 \cdot \frac{\text{Class}_{\text{bot}}}{3} + 0.15 \cdot \text{Stress}_{\text{fuel}} + 0.15 \cdot S_{\text{sentiment}} \right) \times 100$$
- $\tilde{y}_{\text{delay}} = \min(1.0, \frac{\hat{y}_{\text{delay}}}{14.0})$: Normalized route delay factor.
- $\text{Stress}_{\text{fuel}} = \text{clip}\left(\frac{P_{\text{Brent}} - 50}{70}, 0.05, 1.0\right)$: Bunker fuel price pressure.
- $S_{\text{sentiment}} \in [0, 1]$: FinBERT negative disruption sentiment along corridor.

---

## 📊 Model Performance Benchmark

| Model | Target | Primary Metric | Validation Score | Benchmark Target | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1: XGBRegressor** | Route Delay (Days) | **MAE** / **$R^2$** | **0.682 days** / **0.925** | $< 1.2$ days / $> 0.85$ | ✅ Exceeded |
| **Model 2: XGBClassifier** | Port Bottleneck Risk | **Accuracy** / **Weighted F1** | **68.55%** / **0.657** | $> 65\%$ / $> 0.60$ | ✅ Exceeded |
| **Model 3: XGBRegressor** | Disruption Severity (DSI) | **MAE** / **$R^2$** | **4.41 pts** / **0.892** | $< 6.0$ pts / $> 0.80$ | ✅ Exceeded |

---

## 📂 Project Directory Structure

```
d:/final year project/
├── config/
│   ├── __init__.py
│   ├── config.py                 # Hyperparameters, dataclasses, global constants
│   └── ports_routes.py           # Top 16 global ports, choke points & 12 major trade lanes
├── data/
│   ├── raw/
│   │   ├── historical_shipping_logs.csv  # Synthesized multi-year voyage dataset (5,500 records)
│   │   └── macro_indicators.csv          # Cached YFinance oil & FX time series
│   ├── processed/
│   └── news_feed.json            # Curated supply chain dispatches analyzed via FinBERT
├── src/
│   ├── __init__.py
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── shipping_data.py      # Statistical maritime voyage generator & loader
│   │   ├── macro_fetcher.py      # YFinance oil & FX fetcher with offline fallback
│   │   └── news_fetcher.py       # Supply chain news fetcher & aggregator
│   ├── nlp/
│   │   ├── __init__.py
│   │   └── sentiment_analyzer.py # FinBERT transformer pipeline & logistics domain NLP
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   └── pipeline.py           # Multi-modal fusion, interactions, and column transformations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py              # XGBoost training pipeline for 3 targets
│   │   └── predictor.py          # Real-time inference engine & stress testing simulator
│   └── visualization/
│       ├── __init__.py
│       ├── map_visualizer.py     # Interactive Folium global maritime map
│       └── charts.py             # Plotly executive gauges, time series & confusion heatmaps
├── saved_models/
│   ├── model_delay_regressor.joblib
│   ├── model_bottleneck_classifier.joblib
│   ├── model_dsi_regressor.joblib
│   ├── preprocessor.joblib
│   ├── feature_names.joblib
│   └── metrics.json
├── dashboard/
│   └── app.py                    # Streamlit multi-tab executive web application
├── tests/
│   ├── __init__.py
│   ├── test_nlp.py               # FinBERT and sentiment unit tests
│   ├── test_data.py              # Data ingestion unit tests
│   └── test_models.py            # Prediction and stress testing unit tests
├── scripts/
│   ├── generate_and_train.py     # End-to-end data generation and training script
│   └── run_dashboard.bat         # Windows one-click launcher
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quickstart & Execution Guide

### 1. Environment Verification
All dependencies (`streamlit`, `xgboost`, `scikit-learn`, `transformers`, `torch`, `yfinance`, `folium`, `plotly`) are already installed.

### 2. Run All Automated Unit Tests
To verify data ingestion, FinBERT NLP, and ML inference:
```bash
python -m unittest discover tests
```
*Output: `Ran 9 tests in 0.22s - OK`*

### 3. Train or Retrain Models (One-Click Orchestrator)
To regenerate shipping transit data, sync live macro indicators, and train all 3 XGBoost models:
```bash
python scripts/generate_and_train.py
```

### 4. Launch the Interactive Dashboard
Launch the Streamlit executive web application:
```bash
python -m streamlit run dashboard/app.py
```
Or double-click `scripts/run_dashboard.bat`.
Open **`http://localhost:8501`** in any web browser.

---

## 🖥️ Dashboard Features

1. **🌐 Global Maritime Risk Map**: Interactive world map rendering container ports with live bottleneck status markers, choke points (Suez Canal, Panama Canal, Strait of Malacca, Bab el-Mandeb), and major maritime trade lanes.
2. **🚢 Route Delay & Risk Predictor**: Select Origin, Destination, Vessel Class (TEU), Seasonal Month, Choke Point Reroute status, Oil Price, and FinBERT Sentiment to receive instant ML delay predictions, 95% confidence intervals, and primary risk driver explanations.
3. **⚡ What-If Scenario Stress Testing**: Run comparative stress tests across 5 major global disruption scenarios (Red Sea Blockade, Panama Drought, Port Strikes, Oil Shocks).
4. **📰 FinBERT News Sentiment Pulse**: Test breaking news dispatches in real-time using FinBERT NLP, inspect sentiment confidence, and explore the curated maritime intelligence database.
5. **📈 Macroeconomic Intelligence**: Dual-axis energy and foreign exchange tracking via YFinance, rolling volatility metrics, and cross-domain correlation matrices.
6. **🎯 Model Diagnostics & Architecture**: Inspect XGBoost feature importances (Gain), port bottleneck confusion matrices, and the end-to-end data flow architecture.

---

## 🛡️ License
This project is licensed under the MIT License.
#   G l o b a l - S u p p l y - C h a i n - R i s k - D i s r u p t i o n - P r e d i c t o r  
 #   G l o b a l - S u p p l y - C h a i n - R i s k - D i s r u p t i o n - P r e d i c t o r  
 