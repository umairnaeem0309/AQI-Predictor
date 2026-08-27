# Model Experiment Plan

## AQI Predictor — Model Training Strategy

**Status:** Ready for execution
**Dataset:** 107,064 hourly observations (3 cities, 4 years)
**Targets:** AQI at 24h, 48h, 72h horizons

---

## 1. Candidate Models

| Model | Type | Why |
|-------|------|-----|
| **Ridge Regression** | Linear baseline | Simple, fast, interpretable. Sets the performance floor. |
| **Random Forest** | Ensemble tree | Handles non-linear relationships, robust to outliers, feature importance built-in. |
| **XGBoost** | Gradient boosting | State-of-the-art tabular performance, handles missing values natively. |
| **LSTM** | Recurrent neural net | Captures temporal patterns directly, but higher complexity and training cost. |

**Important:** Complex models do not have to beat Ridge. Ridge is a comparison baseline. Selection considers performance AND complexity.

---

## 2. Evaluation Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| **MAE** | mean(\|y - ŷ\|) | Primary metric — average prediction error in AQI units |
| **RMSE** | sqrt(mean((y - ŷ)²)) | Penalizes large errors more heavily |
| **R²** | 1 - SS_res/SS_tot | Proportion of variance explained (0–1) |

**Per-horizon evaluation:** MAE, RMSE, R² computed separately for 24h, 48h, 72h.

**Overall evaluation:** Aggregated across all horizons.

---

## 3. Training Configuration

### 3.1 Data Split

| Split | Years | Rows | Purpose |
|-------|-------|------|---------|
| Train | 2022–2024 | 63,648 | Model fitting |
| Validation | 2025 | 26,280 | Hyperparameter tuning |
| Test | 2026 | 17,136 | Final evaluation |

**Split method:** Chronological (no random shuffling).

### 3.2 Preprocessing

- Missing values: fill with 0 (features) or drop rows (targets)
- Scaling: StandardScaler for linear models (Ridge)
- Tree models: no scaling required
- LSTM: sequence input (24-timestep windows)

### 3.3 Hyperparameters

| Model | Parameters |
|-------|-----------|
| Ridge | alpha=1.0 |
| RandomForest | n_estimators=100, max_depth=20, random_state=42 |
| XGBoost | n_estimators=200, max_depth=6, learning_rate=0.1 |
| LSTM | epochs=50, batch_size=32, sequence_length=24 |

---

## 4. Selection Criteria

### 4.1 Performance Criteria

- MAE improvement over Ridge baseline: ≥ 5% improvement justified as meaningful
- R² on test set: > 0.5 indicates useful predictive power
- Consistent performance across horizons: degradation from 24h→72h should be gradual

### 4.2 Complexity Criteria

- Training time: < 5 minutes acceptable
- Inference time: < 100ms per prediction
- Model size: < 100MB for deployment
- Interpretability: feature importance available

### 4.3 Production Criteria

- Can be serialized/deserialized reliably
- Supports multi-output prediction (24h, 48h, 72h)
- Handles missing input features gracefully
- Reproducible results with fixed random seed

---

## 5. Experiment Execution Order

1. **Ridge** — Establish baseline performance
2. **Random Forest** — Test non-linear improvement
3. **XGBoost** — Test gradient boosting improvement
4. **LSTM** — Test temporal pattern capture (if time permits)

Each model:
1. Train on training set
2. Evaluate on validation set
3. Record metrics
4. Compare against Ridge baseline

---

## 6. MLflow Tracking

All experiments logged to local MLflow:

| Parameter | Logged |
|-----------|--------|
| Model type | ✅ |
| Hyperparameters | ✅ |
| Training time | ✅ |
| MAE/RMSE/R² per horizon | ✅ |
| Feature importance | ✅ (tree models) |
| Model artifact | ✅ |

---

## 7. Expected Outcomes

| Model | Expected MAE (24h) | Expected Complexity | Notes |
|-------|-------------------|--------------------|----|
| Ridge | ~20–30 | Very Low | Linear baseline |
| RandomForest | ~15–25 | Medium | Non-linear patterns |
| XGBoost | ~12–20 | Medium-High | Best tabular performance |
| LSTM | ~15–25 | High | Temporal patterns, but may not beat XGBoost on tabular features |

**Note:** These are estimates based on AQI prediction literature. Actual results will determine selection.

---

## 8. Post-Training Steps

After model experiments complete:

1. Select final model based on criteria above
2. Log to MLflow model registry
3. Create model metadata (feature schema, metrics, parameters)
4. Register model version
5. Prepare for API deployment
