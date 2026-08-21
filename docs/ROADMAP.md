# Limitations and Future Roadmap

## AQI Predictor — Current Limitations and Future Plans

**Version:** 1.0  
**Last Updated:** 21 August 2026

---

## Current Limitations

### Data Collection

| Limitation | Description | Impact | Mitigation |
|---|---|---|---|
| No real API keys | System runs in mock mode without API keys | Simulated data only | Configure API keys |
| Historical data unavailable | No historical API data collected | Cannot train on real data | Accumulate over time |
| Rate limits | Free tier API limits | May miss data points | Implement caching |

### Feature Store

| Limitation | Description | Impact | Mitigation |
|---|---|---|---|
| Hopsworks dependency | Requires Hopsworks account | Setup complexity | Local fallback available |
| No online features | Offline features only | No real-time serving | Future enhancement |

### ML Pipeline

| Limitation | Description | Impact | Mitigation |
|---|---|---|---|
| Synthetic training data | Model trained on synthetic data | Results not production-ready | Wait for real data |
| No LSTM implementation | Only Ridge and XGBoost implemented | Missing deep learning | Future phase |
| No hyperparameter tuning | Manual configuration | Suboptimal models | Add Optuna integration |

### API

| Limitation | Description | Impact | Mitigation |
|---|---|---|---|
| No historical predictions | Only current prediction available | Cannot analyze trends | Add history endpoint |
| No batch predictions | Single prediction only | Limited throughput | Add batch endpoint |
| No rate limiting per user | Global rate limiting only | No user isolation | Future enhancement |

### Dashboard

| Limitation | Description | Impact | Mitigation |
|---|---|---|---|
| No historical analytics | Placeholder shown | Limited insights | Add backend endpoint |
| No SHAP explanations | Placeholder shown | No model interpretability | Add SHAP integration |
| No real-time updates | Manual refresh only | Stale data | Add WebSocket support |

### Monitoring

| Limitation | Description | Impact | Mitigation |
|---|---|---|---|
| No email alerts | Console/log alerts only | Missed notifications | Add email integration |
| No Slack integration | Placeholder only | No team alerts | Add Slack webhook |
| No drift auto-retraining | Manual retraining required | Delayed response | Future automation |

---

## Future Roadmap

### Phase 17: Data Enhancement (Q4 2026)

**Objective:** Real data collection and historical backfill

| Task | Priority | Effort |
|---|---|---|
| Configure API keys | High | 1 day |
| Collect 30 days real data | High | 1 week |
| Historical API backfill | Medium | 1 week |
| Data quality validation | Medium | 2 days |

### Phase 18: Model Enhancement (Q1 2027)

**Objective:** Advanced models and hyperparameter tuning

| Task | Priority | Effort |
|---|---|---|
| Implement LSTM model | High | 2 weeks |
| Add Optuna tuning | Medium | 1 week |
| Ensemble methods | Medium | 1 week |
| Cross-validation | High | 3 days |

### Phase 19: API Enhancement (Q1 2027)

**Objective:** Advanced API features

| Task | Priority | Effort |
|---|---|---|
| Batch predictions | Medium | 1 week |
| Historical predictions | Medium | 1 week |
| WebSocket support | Low | 1 week |
| API versioning | Medium | 3 days |

### Phase 20: Dashboard Enhancement (Q2 2027)

**Objective:** Advanced dashboard features

| Task | Priority | Effort |
|---|---|---|
| Historical analytics | High | 2 weeks |
| SHAP explanations | High | 2 weeks |
| Real-time updates | Medium | 1 week |
| Mobile responsive | Low | 1 week |

### Phase 21: Monitoring Enhancement (Q2 2027)

**Objective:** Production monitoring

| Task | Priority | Effort |
|---|---|---|
| Email alerts | High | 3 days |
| Slack integration | High | 3 days |
| Auto-retraining | Medium | 2 weeks |
| Drift detection alerts | Medium | 1 week |

### Phase 22: Production Hardening (Q3 2027)

**Objective:** Enterprise features

| Task | Priority | Effort |
|---|---|---|
| Kubernetes deployment | Medium | 2 weeks |
| Load balancing | Medium | 1 week |
| SSL/TLS termination | High | 3 days |
| Audit logging | Medium | 1 week |

---

## Technical Debt

### Current Technical Debt

| Item | Priority | Effort | Description |
|---|---|---|---|
| Add type hints | Low | 1 week | Complete type annotation |
| Update dependencies | Medium | 2 days | Bump to latest stable versions |
| Add docstrings | Low | 1 week | Complete API documentation |
| Optimize queries | Medium | 3 days | Database query optimization |

### Code Quality Improvements

| Item | Priority | Effort |
|---|---|---|
| Add more unit tests | Medium | 1 week |
| Integration test coverage | Medium | 1 week |
| Performance testing | Low | 3 days |
| Security audit | High | 1 week |

---

## Research Opportunities

### Model Improvements

- **Transformer models**: Explore temporal transformers for time series
- **Graph neural networks**: Model city relationships
- **Transfer learning**: Pre-train on global AQI data
- **Uncertainty quantification**: Add prediction confidence intervals

### Data Sources

- **Satellite data**: Incorporate satellite imagery
- **Traffic data**: Include traffic patterns
- **Industrial data**: Factor in industrial activity
- **Weather forecasts**: Use weather prediction services

### Features

- **Multi-modal inputs**: Combine tabular and image data
- **Spatial features**: City-to-city relationships
- **Temporal patterns**: Seasonal decomposition
- **External factors**: Policy changes, events

---

## Success Metrics

### Current Metrics

| Metric | Target | Current |
|---|---|---|
| API response time | < 100ms | ~50ms |
| Prediction accuracy (MAE) | < 20 | ~15 (synthetic) |
| Test coverage | > 80% | ~85% |
| Documentation coverage | 100% | 100% |

### Future Metrics

| Metric | Target | Timeline |
|---|---|---|
| Real data MAE | < 20 | Q4 2026 |
| LSTM implementation | Complete | Q1 2027 |
| SHAP explanations | Complete | Q2 2027 |
| Auto-retraining | Complete | Q2 2027 |

---

## Resources

### Documentation

- `docs/ARCHITECTURE.md` - System design
- `docs/DEPLOYMENT.md` - Deployment guide
- `docs/API.md` - API reference
- `docs/TESTING_SUMMARY.md` - Test coverage

### External Resources

- [Hopsworks Documentation](https://docs.hopsworks.ai/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Contributing

### Development Workflow

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request
6. Review and merge

### Code Standards

- Follow PEP 8
- Use black for formatting
- Use isort for imports
- Add type hints
- Write docstrings

---

## Version History

| Version | Date | Changes |
|---|---|---|
| 1.0 | 21 Aug 2026 | Initial release with 15 phases |
