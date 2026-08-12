# Model Report

## AQI Predictor — Model Comparison Report

**Version:** 1.0  
**Date:** 11 August 2026  
**Status:** Phase 7 — ML Experiment Pipeline  

---

## 1. Experiment Overview

| Item | Value |
|---|---|
| **Objective** | Compare forecasting models for 3-day AQI prediction |
| **Target Variables** | target_aqi_24h, target_aqi_48h, target_aqi_72h |
| **Evaluation Metrics** | MAE, RMSE, R² |
| **Validation Approach** | Chronological train/val/test split |
| **Reproducibility** | Fixed random seed, dataset version, feature version |

---

## 2. Data Status

| Property | Value |
|---|---|
| **Dataset Type** | Synthetic test data |
| **Approved for Training** | ❌ No |
| **Approved for Evaluation** | ❌ No |
| **Is Reportable** | ❌ No |

**⚠️ IMPORTANT:** All results in this report are from synthetic test data and are for pipeline validation only. Final model results require real API data.

---

## 3. Candidate Models

| # | Model | Type | Reasoning |
|---|---|---|---|
| 1 | Ridge Regression | Linear baseline | Simple, interpretable comparison point |
| 2 | Random Forest | Tree ensemble | Handles nonlinear relationships |
| 3 | XGBoost | Gradient boosting | Strong for tabular data |
| 4 | LSTM | Deep learning | Captures temporal dependencies |

---

## 4. Results Summary

### 4.1 Performance Metrics

*To be populated after real data experiments.*

| Model | MAE (24h) | MAE (48h) | MAE (72h) | RMSE (24h) | RMSE (48h) | RMSE (72h) | R² (avg) |
|---|---|---|---|---|---|---|---|
| Ridge | — | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — | — |
| XGBoost | — | — | — | — | — | — | — |
| LSTM | — | — | — | — | — | — | — |

### 4.2 Engineering Metrics

| Model | Training Time | Inference Speed | Model Size | Complexity |
|---|---|---|---|---|
| Ridge | — | — | — | Low |
| Random Forest | — | — | — | Medium |
| XGBoost | — | — | — | Medium |
| LSTM | — | — | — | High |

---

## 5. Baseline Comparison

The Ridge regression baseline establishes a comparison point.

Complex models do NOT have to beat Ridge — selection considers both
performance and complexity trade-offs.

| Model | vs Ridge MAE | vs Ridge RMSE | Complexity Increase |
|---|---|---|---|
| Random Forest | — | — | Medium |
| XGBoost | — | — | Medium |
| LSTM | — | — | High |

---

## 6. Feature Importance

*To be populated after training with real data.*

Top features by importance (XGBoost example):
1. —
2. —
3. —

---

## 7. Production Model Selection

**Phase 8 will make the production model decision using:**
- Experimental evidence (MAE, RMSE, R²)
- Engineering metrics (speed, complexity, maintainability)
- Deployment considerations

**This phase (Phase 7) only establishes baselines and comparison data.**

---

## 8. Reproducibility Information

| Item | Value |
|---|---|
| Random Seed | 42 |
| Feature Version | 1.0.0 |
| Schema Version | 1.0 |
| Dataset Version | — |
| Training Timestamp | — |

---

## 9. Next Steps

- [ ] Collect real API data
- [ ] Run experiments with real data
- [ ] Populate this report with real results
- [ ] Phase 8: Make production model decision
