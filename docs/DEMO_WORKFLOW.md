# Demo Workflow

## AQI Predictor — Reproducible Demo Guide

**Version:** 1.0  
**Last Updated:** 21 August 2026

---

## Overview

This document provides a step-by-step guide for demonstrating the AQI Predictor system.

---

## Prerequisites

### System Requirements

- Python 3.11
- Docker and Docker Compose
- Modern web browser
- Internet connection (for API calls in production)

### API Keys (Optional for Demo)

- `OPENWEATHER_API_KEY` - Required for real data
- `AQICN_API_KEY` - Required for real data
- `HOPSWORKS_HOST` - Required for feature store

---

## Demo Modes

### 1. Mock Mode Demo (No API Keys Required)

Perfect for demonstrations without real API keys.

```bash
# Start with mock mode
docker compose -f docker/docker-compose.prod.yml up -d

# Or run locally
MOCK_MODE=true streamlit run app/frontend/streamlit_app.py
```

**What Mock Mode Provides:**
- Simulated AQI predictions for all cities
- Mock health and model info responses
- No external API dependencies

### 2. Production Mode Demo (API Keys Required)

Full functionality with real data.

```bash
# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Deploy
docker compose -f docker/docker-compose.prod.yml up -d
```

---

## Demo Script

### Opening (2 minutes)

1. **Introduction**
   - "AQI Predictor is a production-grade ML system for forecasting air quality in Pakistani cities"
   - "It predicts AQI for 24h, 48h, and 72h horizons"

2. **Architecture Overview**
   - Show `docs/ARCHITECTURE.md` diagram
   - Highlight key components: Data Collection → Feature Engineering → ML Training → API → Dashboard

### Main Dashboard Demo (5 minutes)

1. **Navigate to Dashboard**
   - Open browser to `http://localhost:8501`
   - Show the main dashboard

2. **City Selection**
   - Select Karachi from dropdown
   - Show AQI forecast cards (24h, 48h, 72h)
   - Show forecast chart

3. **Switch Cities**
   - Switch to Lahore
   - Show how predictions change
   - Switch to Islamabad
   - Show cleaner AQI values

4. **Refresh Functionality**
   - Click refresh button
   - Show data updates

### System Status Demo (3 minutes)

1. **Navigate to System Page**
   - Click "System" in sidebar

2. **Show Health Status**
   - Backend status: Healthy
   - Model loaded: True
   - Feature store: Connected

3. **Show Model Information**
   - Model name and version
   - Dataset type: real_api_data
   - Training metrics

### API Demo (3 minutes)

1. **Show API Documentation**
   - Open `http://localhost:8000/docs`
   - Show Swagger UI

2. **Make API Call**
   - Use Swagger UI to make prediction request
   - Show request/response

3. **Show Health Endpoint**
   - Call `/health` endpoint
   - Show response

### Analytics & Explainability (2 minutes)

1. **Analytics Page**
   - Navigate to Analytics
   - Show placeholder for historical data
   - Explain future capabilities

2. **Explainability Page**
   - Navigate to Explainability
   - Show SHAP explanation placeholder
   - Explain future capabilities

### Closing (2 minutes)

1. **Summary**
   - "The system is production-ready with Docker deployment"
   - "All safety checks enforced: no synthetic models, production-only"

2. **Future Roadmap**
   - Real API data collection
   - Historical analytics
   - SHAP explainability

---

## Key Talking Points

### Production Safety

- "MOCK_MODE is enforced to false in production"
- "Only approved, real-data models are deployed"
- "Pre-deployment safety checks prevent misconfiguration"

### Architecture

- "Microservices architecture with FastAPI backend and Streamlit frontend"
- "Feature store with Hopsworks primary and local fallback"
- "MLflow for experiment tracking and model registry"

### Monitoring

- "Evidently AI for drift detection"
- "Alerting with cooldown and aggregation"
- "Prediction logging with security checks"

---

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker compose -f docker/docker-compose.prod.yml logs

# Verify environment
docker compose -f docker/docker-compose.prod.yml exec backend env
```

### Frontend Can't Connect

```bash
# Verify backend is running
curl http://localhost:8000/health

# Check API_BASE_URL in frontend env
```

### Mock Mode Not Working

```bash
# Verify MOCK_MODE is set
echo $MOCK_MODE

# Set and restart
MOCK_MODE=true docker compose -f docker/docker-compose.prod.yml up -d
```

---

## Demo Checklist

- [ ] Services running (backend + frontend)
- [ ] Dashboard accessible at http://localhost:8501
- [ ] API docs accessible at http://localhost:8000/docs
- [ ] All three cities working
- [ ] Health checks passing
- [ ] No errors in logs

---

## Post-Demo

### Cleanup

```bash
# Stop services
docker compose -f docker/docker-compose.prod.yml down

# Remove volumes (optional)
docker compose -f docker/docker-compose.prod.yml down -v
```
