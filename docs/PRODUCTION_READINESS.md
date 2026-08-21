# Production Readiness Checklist

## AQI Predictor — Production Readiness

**Last Updated:** 20 August 2026  
**Version:** 1.0

---

## Pre-Deployment Checklist

### Environment Configuration

- [ ] `MOCK_MODE=false` (CRITICAL: Must be false in production)
- [ ] `API_KEY` is set with strong, unique value
- [ ] `HOPSWORKS_HOST` is set (or local fallback disabled)
- [ ] `OPENWEATHER_API_KEY` is valid
- [ ] `AQICN_API_KEY` is valid
- [ ] `API_BASE_URL` points to backend (for frontend)
- [ ] Rate limiting is enabled (`RATE_LIMIT_ENABLED=true`)

### Model Readiness

- [ ] Production model exists in MLflow registry
- [ ] Model status is `production`
- [ ] Model approval status is `approved`
- [ ] Model dataset_type is `real_api_data` (NOT synthetic)
- [ ] Model feature_version matches current features
- [ ] Model metrics meet minimum thresholds

### Infrastructure

- [ ] Docker images built successfully
- [ ] Health endpoints respond correctly
- [ ] Feature store is accessible
- [ ] MLflow registry is accessible
- [ ] Logging is configured

### Security

- [ ] No secrets in code or logs
- [ ] API key authentication enabled
- [ ] Rate limiting configured
- [ ] CORS configured appropriately
- [ ] HTTPS termination configured (if applicable)

---

## Deployment Checklist

### Pre-Deployment

```bash
# Run pre-deployment checks
python scripts/pre_deploy_checks.py

# Expected output: All checks passed
```

### Build Images

```bash
# Build backend
docker compose -f docker/docker-compose.prod.yml build backend

# Build frontend
docker compose -f docker/docker-compose.prod.yml build frontend
```

### Deploy Services

```bash
# Deploy all services
docker compose -f docker/docker-compose.prod.yml up -d
```

### Post-Deployment Verification

```bash
# Check service status
docker compose -f docker/docker-compose.prod.yml ps

# Verify backend health
curl http://localhost:8000/health

# Verify frontend health
curl http://localhost:8501/_stcore/health

# Check logs for errors
docker compose -f docker/docker-compose.prod.yml logs --tail=50
```

---

## Health Check Verification

| Endpoint | Expected Response | Status |
|---|---|---|
| `GET /` | `{"status": "available"}` | 200 |
| `GET /health` | `{"status": "healthy", "model_loaded": true}` | 200 |
| `POST /prediction` | Prediction response | 200 (with valid request) |

---

## Rollback Procedure

### Automatic Rollback Triggers

- Health check fails after deployment
- Pre-deployment checks fail
- Error rate exceeds threshold

### Manual Rollback

```bash
# Stop current deployment
docker compose -f docker/docker-compose.prod.yml down

# Verify services stopped
docker compose -f docker/docker-compose.prod.yml ps

# Restart with previous images (if available)
docker compose -f docker/docker-compose.prod.yml up -d
```

---

## Monitoring Checklist

- [ ] Health checks are running
- [ ] Logs are being collected
- [ ] Error rate is monitored
- [ ] Prediction latency is tracked
- [ ] Model performance is monitored

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| Backend won't start | Check `MOCK_MODE=false` |
| Frontend can't connect | Verify backend is healthy |
| Model not loaded | Check MLflow registry and model status |
| Feature store error | Verify HOPSWORKS_HOST or enable local fallback |

### Debug Commands

```bash
# Check container logs
docker logs aqi-predictor-backend
docker logs aqi-predictor-frontend

# Check environment
docker compose -f docker/docker-compose.prod.yml exec backend env

# Restart specific service
docker compose -f docker/docker-compose.prod.yml restart backend
```

---

## Sign-Off

| Role | Name | Date | Approved |
|---|---|---|---|
| Developer | | | [ ] |
| Reviewer | | | [ ] |
| Operations | | | [ ] |
