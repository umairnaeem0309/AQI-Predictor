# API Documentation

## AQI Predictor API


## Authentication

All prediction and model endpoints require API key authentication.


---

## Endpoints

### GET `/`

Service availability check.

**Response:**
```json
{
  "status": "available",
  "service": "AQI Predictor API"
}
```

---

### GET `/health`

Health check with model and feature store status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_store_connected": true,
  "last_prediction": "2026-08-17T10:00:00Z",
  "version": "1.0.0"
}
```

---

### POST `/prediction`

Generate 3-day AQI prediction.

**Headers:**
- `X-API-Key`: Required

**Request Body:**
```json
{
  "city": "Karachi",
  "include_explanation": false
}
```

**Valid Cities:**
- Karachi
- Lahore
- Islamabad

**Response:**
```json
{
  "city": "Karachi",
  "timestamp": "2026-08-17T10:00:00Z",
  "aqi_24h": 142,
  "aqi_48h": 138,
  "aqi_72h": 145,
  "category_24h": "Unhealthy for Sensitive Groups",
  "category_48h": "Unhealthy for Sensitive Groups",
  "category_72h": "Unhealthy for Sensitive Groups",
  "model_version": "1.0.0",
  "confidence": null
}
```

**Note:** Confidence intervals are computed from residual statistics.

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 | Invalid city name |
| 401 | Invalid or missing API key |
| 429 | Rate limit exceeded |
| 503 | Model not loaded or service unavailable |
| 500 | Internal server error |

---

### GET `/model-info`

Get production model metadata.

**Headers:**
- `X-API-Key`: Required

**Response:**
```json
{
  "model_name": "xgboost_v1",
  "model_version": "1.0.0",
  "status": "production",
  "approval_status": "approved",
  "training_date": "2026-08-15",
  "dataset_type": "real_api_data",
  "feature_version": "1.0.0",
  "metrics": {
    "mae": 15.2,
    "rmse": 20.1,
    "r2": 0.85
  }
}
```

---

## Security

### API Key Authentication

- All prediction endpoints require `X-API-Key` header
- API key validated against `API_KEY` environment variable
- Invalid or missing key returns 401

### Rate Limiting

- Configurable rate limiting (default: 100 requests/minute)
- Exceeding limit returns 429
- Rate limit per client IP

### Data Security

- No sensitive data logged in prediction logs
- Features hashed for privacy
- No API keys or PII in logs

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message",
  "type": "ErrorType"
}
```

### Common Error Types

| Type | Description |
|------|-------------|
| `InvalidCityError` | Invalid city name |
| `ModelNotLoadedError` | Model not loaded |
| `SyntheticModelRejectedError` | Synthetic model blocked |
| `ModelApprovalError` | Model not approved |
| `FeatureStoreError` | Feature store unavailable |
| `PredictionError` | Prediction failed |

---

## AQI Categories

US EPA AQI categories used for forecasts:

| AQI Range | Category | Color |
|-----------|----------|-------|
| 0-50 | Good | #00E400 |
| 51-100 | Moderate | #FFFF00 |
| 101-150 | Unhealthy for Sensitive Groups | #FF7E00 |
| 151-200 | Unhealthy | #FF0000 |
| 201-300 | Very Unhealthy | #8F3F97 |
| 301-500 | Hazardous | #7E0023 |

---

## Rate Limits

Default rate limits:
- **Requests per window:** 100
- **Window duration:** 60 seconds
- **Per client IP**

Configure via environment variables:
- `RATE_LIMIT_ENABLED=true`
- `RATE_LIMIT_REQUESTS=100`
- `RATE_LIMIT_WINDOW_SECONDS=60`

---

## Examples

### Get Prediction

```bash
curl -X POST http://localhost:8000/prediction \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"city": "Karachi"}'
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Info

```bash
curl http://localhost:8000/model-info \
  -H "X-API-Key: your-api-key"
```
