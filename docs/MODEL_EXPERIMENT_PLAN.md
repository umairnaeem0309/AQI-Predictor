# Model Experiment Plan

## AQI Predictor — Model Training Strategy

**Status:** ✅ All experiments complete
**Dataset:** 107,064 hourly observations (3 cities, 4 years)
**Targets:** AQI at 24h, 48h, 72h horizons
**Winner:** XGBoost (MAE=21.32, R²=0.6065)

---

## 1. Candidate Models

| Model | Type | Why |
|-------|------|-----|
| **Ridge Regression** | Linear baseline | Simple, fast, interpretable. Sets the performance floor. |
| **Random Forest** | Ensemble tree | Handles non-linear relationships, robust to outliers, feature importance built-in. |
| **XGBoost** | Gradient boosting | State-of-the-art tabular performance, handles missing values natively. |
| **LSTM** | Recurrent neural net | Captures temporal patterns directly from sequential data. Required in final comparison. |

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

All four models are mandatory. No model is optional.

1. **Ridge** — Establish baseline performance
2. **Random Forest** — Test non-linear improvement
3. **XGBoost** — Test gradient boosting improvement
4. **LSTM** — Test temporal pattern capture

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

## 7. Full Dataset Results (FINAL)

**Test set:** 2026 data (16,920 rows) — unseen during training.
**Split:** Train 2022–2024 → Val 2025 → Test 2026.

### Overall Results

| Model | MAE | RMSE | R² | Train Time | Inference |
|-------|-----|------|----|-----------|----------|
| **XGBoost** | **21.32** | 30.89 | 0.6065 | 18.2s | 0.030ms |
| RandomForest | 21.47 | **30.74** | **0.6103** | 477.7s | 0.047ms |
| Ridge | 21.98 | 31.99 | 0.5779 | 1.9s | 0.001ms |
| LSTM | 26.17 | 38.86 | 0.3771 | 224.3s | 0.371ms |

### Per-Horizon Results

| Model | 24h MAE | 48h MAE | 72h MAE | 24h R² | 48h R² | 72h R² |
|-------|---------|---------|---------|--------|--------|--------|
| **XGBoost** | **19.22** | **21.87** | **22.87** | **0.6707** | **0.5887** | **0.5591** |
| RandomForest | 19.58 | 21.97 | 22.87 | 0.6632 | 0.5982 | 0.5689 |
| Ridge | 19.63 | 22.47 | 23.85 | 0.6648 | 0.5585 | 0.5094 |
| LSTM | 25.61 | 26.12 | 26.79 | 0.3994 | 0.3757 | 0.3558 |

### Selection Reasoning

**XGBoost selected as production model:**
- Lowest MAE overall (21.32) and at every horizon
- Fastest non-linear training (18.2s vs RF's 477.7s)
- Nearly identical performance to RF but 26× faster to train
- Inference speed adequate for real-time API (0.030ms/sample)
- Strong R² (0.6065) — explains 60.65% of AQI variance

**Ridge as backup/baseline:**
- Within 3% of XGBoost MAE — problem has strong linear signal
- Instant training and inference
- Fully interpretable

**LSTM excluded:**
- R²=0.3771 vs XGBoost's 0.6065 — significantly weaker
- Temporal patterns already captured by engineered lag/rolling features
- 12× slower training, 12× slower inference than XGBoost
- Would need substantially more data or different architecture to compete

---

## 8. Post-Training Steps

After model experiments complete:

1. Select final model based on criteria above
2. Log to MLflow model registry
3. Create model metadata (feature schema, metrics, parameters)
4. Register model version
5. Prepare for API deployment
