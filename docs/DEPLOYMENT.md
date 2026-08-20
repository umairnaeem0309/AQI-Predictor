# Deployment Guide

## AQI Predictor Deployment

**Version:** 1.0.0  
**Last Updated:** 19 August 2026

---

## Overview

This guide covers deploying the AQI Predictor to production using Docker Compose.

---

## Prerequisites

- Docker Engine 20.10+
- Docker Compose v2+
- Python 3.11+
- Valid API keys

---

## Environment Variables

### Required Variables

| Variable | Description | Example |
|---|---|---|
| `MOCK_MODE` | Must be `false` in production | `false` |
| `API_KEY` | API authentication key | `your-api-key` |
| `HOPSWORKS_HOST` | Hopsworks feature store host | `your-host.hopsworks.ai` |
| `OPENWEATHER_API_KEY` | OpenWeather API key | `your-openweather-key` |
| `AQICN_API_KEY` | AQICN API key | `your-aqicn-key` |

### Optional Variables

| Variable | Default | Description |
|---|---|---|
| `BACKEND_PORT` | `8000` | Backend API port |
| `FRONTEND_PORT` | `8501` | Dashboard port |
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_REQUESTS` | `100` | Requests per window |
| `LOG_LEVEL` | `info` | Logging level |

---

## Deployment Steps

### 1. Clone Repository

```bash
git clone <repository-url>
cd AQI-Predictor
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your values
nano .env
```

**.env file:**
```bash
MOCK_MODE=false
API_KEY=your-secure-api-key
HOPSWORKS_HOST=your-host.hopsworks.ai
OPENWEATHER_API_KEY=your-key
AQICN_API_KEY=your-key
```

### 3. Run Pre-Deployment Checks

```bash
python scripts/pre_deploy_checks.py
```

Expected output:
```
✅ MOCK_MODE is false
✅ API_KEY is set
✅ HOPSWORKS_HOST is set
✅ API keys are set
✅ All checks passed. Safe to deploy.
```

### 4. Build and Deploy

```bash
# Build images
docker compose -f docker/docker-compose.prod.yml build

# Deploy services
docker compose -f docker/docker-compose.prod.yml up -d
```

### 5. Verify Deployment

```bash
# Check service status
docker compose -f docker/docker-compose.prod.yml ps

# Check backend health
curl http://localhost:8000/health

# Check frontend
curl http://localhost:8501/_stcore/health
```

---

## Service Endpoints

| Service | URL | Health Check |
|---|---|---|
| Backend API | `http://localhost:8000` | `/health` |
| Dashboard | `http://localhost:8501` | `/_stcore/health` |
| API Docs | `http://localhost:8000/docs` | N/A |

---

## Production Safety

### Pre-Deployment Checklist

- [ ] `MOCK_MODE=false`
- [ ] `API_KEY` is set and secure
- [ ] `HOPSWORKS_HOST` is set (or local fallback disabled)
- [ ] API keys are valid
- [ ] Model is approved for production
- [ ] No synthetic data in production

### Runtime Safety

| Check | Enforcement |
|---|---|
| MOCK_MODE | Refuse to start if true |
| Model status | Must be `production` |
| Approval status | Must be `approved` |
| Dataset type | Must be `real_api_data` |

---

## Rollback

### Manual Rollback

```bash
# Stop current deployment
docker compose -f docker/docker-compose.prod.yml down

# Start previous version (if image available)
docker compose -f docker/docker-compose.prod.yml up -d
```

### Automatic Rollback

The deployment script automatically rolls back if:
- Health check fails after deployment
- Pre-deployment checks fail

---

## Monitoring

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:8501/_stcore/health
```

### Logs

```bash
# Backend logs
docker compose -f docker/docker-compose.prod.yml logs backend

# Frontend logs
docker compose -f docker/docker-compose.prod.yml logs frontend

# Follow logs
docker compose -f docker/docker-compose.prod.yml logs -f
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| Backend won't start | Check MOCK_MODE is false |
| Frontend can't connect | Verify backend is healthy |
| Health check failing | Check API keys and model status |

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=debug docker compose -f docker/docker-compose.prod.yml up

# Check container logs
docker logs aqi-predictor-backend
docker logs aqi-predictor-frontend
```

---

## Security Notes

1. **Never commit .env file** - Contains secrets
2. **Use strong API keys** - Minimum 32 characters
3. **Enable rate limiting** - Prevent abuse
4. **Use HTTPS in production** - Terminate TLS at load balancer
5. **Monitor access logs** - Detect anomalies

---

## CI/CD Integration

### GitHub Actions Deployment

```yaml
# Trigger deployment
- name: Deploy to production
  run: |
    python scripts/pre_deploy_checks.py
    docker compose -f docker/docker-compose.prod.yml up -d
```

### Deployment Script

```bash
# Full deployment with safety checks
python scripts/deploy.py --environment production
```
