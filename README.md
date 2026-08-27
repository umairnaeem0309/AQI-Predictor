# AQI Predictor

Production-grade Air Quality Index forecasting system predicting AQI for 24h, 48h, and 72h horizons for Pakistani cities.

## Architecture

```
User → Streamlit Dashboard → FastAPI API → XGBoost Model → Predictions
                                ↓
                    Open-Meteo Live Data → Feature Engineering
                                ↓
                    Historical Dataset (4 years, 3 cities)
```

## Features

- **3-day AQI forecasting** for Karachi, Lahore, Islamabad
- **Real-time predictions** using Open-Meteo live data
- **Historical dataset** of 107,000+ hourly observations (Aug 2022 – Aug 2026)
- **US EPA PM NowCast AQI** calculation (EPA-454/B-24-002, May 2024)
- **Interactive dashboard** with analytics and model explainability
- **REST API** with health checks, predictions, and historical data endpoints

## Model Performance

| Model | MAE | RMSE | R² | Training Time |
|-------|-----|------|----|---------------|
| **XGBoost** | **21.32** | 30.89 | 0.6065 | 18.2s |
| RandomForest | 21.47 | 30.74 | 0.6103 | 477.7s |
| Ridge | 21.98 | 31.99 | 0.5779 | 1.9s |
| LSTM | 26.17 | 38.86 | 0.3771 | 224.3s |

**XGBoost selected** — best MAE, fastest non-linear training, real-time ready.

## Quick Start

### Local Development

```bash
# Create environment
conda create -n aqi-predictor python=3.11
conda activate aqi-predictor
pip install -r requirements.txt

# Generate dataset
python scripts/build_dataset.py

# Train models
python scripts/train_all_models.py

# Register model
python scripts/register_best_model.py

# Start API
uvicorn app.backend.main:app --port 8000

# Start Dashboard (separate terminal)
streamlit run app/frontend/streamlit_app.py --server.port 8501
```

### Docker

```bash
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/prediction` | POST | Get 3-day AQI forecast |
| `/model-info` | GET | Model metadata and metrics |
| `/data/historical` | GET | Historical AQI data |
| `/data/statistics` | GET | City statistics |
| `/explain/feature-importance` | GET | Model feature importance |

## Project Structure

```
AQI-Predictor/
├── app/                    # FastAPI backend + Streamlit frontend
│   ├── backend/            # FastAPI app and config
│   ├── routes/             # API endpoints
│   ├── services/           # Business logic
│   ├── schemas/            # Request/response models
│   └── frontend/           # Streamlit dashboard
├── src/                    # Core source code
│   ├── data/               # Data providers and collection
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and registry
│   ├── feature_store/      # Hopsworks + DuckDB fallback
│   └── utils/              # Utilities
├── scripts/                # Production scripts
├── tests/                  # Test suite
├── docs/                   # Documentation
├── models/                 # Trained model artifacts
└── notebooks/              # Jupyter notebooks
```

## Data Sources

- **Weather**: Open-Meteo Archive API (hourly, 2017+)
- **Air Quality**: Open-Meteo CAMS Global (hourly, Aug 2022+)
- **Real-time**: Open-Meteo current weather + air quality

## Tech Stack

- **Backend**: FastAPI, Python 3.11
- **Frontend**: Streamlit
- **ML**: XGBoost, Scikit-learn, TensorFlow/Keras
- **Tracking**: MLflow (local)
- **Feature Store**: Hopsworks (primary), DuckDB/Parquet (fallback)
- **CI/CD**: GitHub Actions

## License

Internal project — not for distribution.
