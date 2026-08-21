# Handoff Documentation

## AQI Predictor — Project Handoff

**Version:** 1.0  
**Handoff Date:** 21 August 2026  
**Status:** Ready for Handoff

---

## Project Summary

The AQI Predictor is a production-grade MLOps system that forecasts Air Quality Index (AQI) at 24h, 48h, and 72h horizons for Pakistani cities (Karachi, Lahore, Islamabad).

### Key Achievements

- ✅ Complete ML pipeline from data collection to prediction
- ✅ Production-ready API with FastAPI
- ✅ Interactive dashboard with Streamlit
- ✅ Docker deployment with safety checks
- ✅ Monitoring and alerting system
- ✅ CI/CD pipeline with GitHub Actions

---

## Repository Structure

```
AQI-Predictor/
├── app/
│   ├── backend/          # FastAPI application
│   ├── frontend/         # Streamlit dashboard
│   ├── routes/           # API endpoints
│   ├── schemas/          # Pydantic models
│   └── services/         # Business logic
├── src/
│   ├── data/             # Data collection
│   ├── features/         # Feature engineering
│   ├── models/           # ML training
│   ├── feature_store/    # Feature storage
│   ├── monitoring/       # Monitoring system
│   └── utils/            # Utilities
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── ci/               # CI validation tests
│   └── deployment/       # Deployment tests
├── docs/                 # Documentation
├── docker/               # Docker configurations
├── scripts/              # Utility scripts
└── .github/workflows/    # CI/CD workflows
```

---

## Environment Setup

### Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd AQI-Predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your values

# 5. Start in mock mode
MOCK_MODE=true streamlit run app/frontend/streamlit_app.py
```

### Docker Deployment

```bash
# Build and run
docker compose -f docker/docker-compose.prod.yml up -d

# Access services
# Dashboard: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MOCK_MODE` | Yes | `true` for development, `false` for production |
| `API_KEY` | Yes | API authentication key |
| `HOPSWORKS_HOST` | Yes | Hopsworks feature store host |
| `OPENWEATHER_API_KEY` | Yes | OpenWeather API key |
| `AQICN_API_KEY` | Yes | AQICN API key |
| `API_BASE_URL` | Yes | FastAPI backend URL (for frontend) |

### Configuration Files

| File | Purpose |
|---|---|
| `.env` | Environment variables (not committed) |
| `.env.example` | Environment template |
| `config.yaml` | Application configuration |
| `.streamlit/config.toml` | Streamlit configuration |

---

## Key Commands

### Development

```bash
# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
```

### Deployment

```bash
# Pre-deployment checks
python scripts/pre_deploy_checks.py

# Deploy
python scripts/deploy.py --environment production

# Check health
curl http://localhost:8000/health
```

---

## Critical Safety Rules

### Production Deployment

1. **MOCK_MODE must be false** - Enforced by pre-deployment checks
2. **Only approved models** - Synthetic models are rejected
3. **Real API data only** - Synthetic data never used for training

### Model Requirements

| Requirement | Value |
|---|---|
| Status | `production` |
| Approval | `approved` |
| Dataset Type | `real_api_data` |
| Feature Version | Must match current |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service availability |
| GET | `/health` | Health check |
| POST | `/prediction` | Get AQI prediction |
| GET | `/model-info` | Model information |
| GET | `/docs` | API documentation |

### Example Request

```bash
curl -X POST http://localhost:8000/prediction \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"city": "Karachi"}'
```

---

## Known Limitations

### Current Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| No real API keys | Uses mock data | Configure keys for production |
| Historical analytics unavailable | Placeholder shown | Backend endpoint needed |
| SHAP explanations unavailable | Placeholder shown | Backend integration needed |

### Future Enhancements

See `docs/ROADMAP.md` for planned features.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| Backend won't start | Check `MOCK_MODE=false` |
| Frontend can't connect | Verify backend is healthy |
| Model not loaded | Check MLflow registry |
| Feature store error | Verify HOPSWORKS_HOST |

### Debug Commands

```bash
# Check logs
docker compose -f docker/docker-compose.prod.yml logs

# Check environment
docker compose -f docker/docker-compose.prod.yml exec backend env

# Restart services
docker compose -f docker/docker-compose.prod.yml restart
```

---

## Documentation Reference

| Document | Purpose |
|---|---|
| `docs/ARCHITECTURE.md` | System architecture |
| `docs/DEPLOYMENT.md` | Deployment guide |
| `docs/PRODUCTION_READINESS.md` | Production checklist |
| `docs/API.md` | API documentation |
| `docs/TESTING_SUMMARY.md` | Test coverage |
| `docs/DEMO_WORKFLOW.md` | Demo guide |
| `docs/ROADMAP.md` | Future roadmap |

---

## Contact

**Developer:** Umair Naeem  
**Email:** umairnaeem0309@gmail.com  
**Project Repository:** AQI-Predictor

---

## Sign-Off

| Role | Name | Date | Signature |
|---|---|---|---|
| Developer | | | |
| Reviewer | | | |
| Operations | | | |
