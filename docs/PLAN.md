# Plan Document

## AQI Predictor — Project Roadmap and Timeline

**Version:** 1.0  
**Date:** 31 July 2026  
**Status:** Phase 0 — Foundation  

---

## 1. Project Timeline

**Start Date:** 31 July 2026  
**Target Completion:** 23 August 2026  
**Total Duration:** ~24 days  

---

## 2. Phase Schedule

| Phase | Name | Target Dates | Duration |
|---|---|---|---|
| 0 | Requirement Analysis and Foundation | 31 Jul – 1 Aug | 2 days |
| 1 | Repository and Environment Setup | 1 – 2 Aug | 2 days |
| 2 | Data Collection Architecture | 2 – 4 Aug | 3 days |
| 3 | Real API Integration | 4 – 5 Aug | 2 days |
| 4 | Feature Engineering Pipeline | 5 – 7 Aug | 3 days |
| 5 | Historical Data Backfill | 7 – 8 Aug | 2 days |
| 6 | Feature Store Implementation | 8 – 10 Aug | 3 days |
| 7 | ML Experiment Pipeline | 10 – 12 Aug | 3 days |
| 8 | Model Selection Decision | 12 – 13 Aug | 1 day |
| 9 | MLflow Model Registry | 13 – 14 Aug | 2 days |
| 10 | GitHub Actions Automation | 14 – 15 Aug | 2 days |
| 11 | Monitoring Implementation | 15 – 16 Aug | 2 days |
| 12 | FastAPI Backend | 16 – 18 Aug | 3 days |
| 13 | Streamlit Dashboard | 18 – 20 Aug | 3 days |
| 14 | Deployment | 20 – 21 Aug | 2 days |
| 15 | Final Documentation | 21 – 22 Aug | 2 days |
| 16 | Demo Preparation | 22 – 23 Aug | 2 days |

---

## 3. Critical Path

The critical path runs through:

```
Phase 1 (Setup)
    → Phase 2 (Data Architecture)
    → Phase 3 (API Integration)
    → Phase 4 (Feature Engineering)
    → Phase 5 (Historical Backfill)
    → Phase 6 (Feature Store)
    → Phase 7 (ML Experiments)
    → Phase 8 (Model Selection)
    → Phase 9 (MLflow Registry)
    → Phase 12 (FastAPI)
    → Phase 13 (Streamlit)
    → Phase 14 (Deployment)
```

Phases 10 (GitHub Actions), 11 (Monitoring), 15 (Docs), and 16 (Demo) can overlap with adjacent phases.

---

## 4. Dependency Map

```
Phase 0 ─── Foundation (no code dependencies)
    │
Phase 1 ─── Environment setup
    │
Phase 2 ─── Data collection (mock)
    │
Phase 3 ─── Real API integration (needs Phase 2)
    │
Phase 4 ─── Feature engineering (needs Phase 3 data)
    │
Phase 5 ─── Historical backfill (needs Phase 4 features)
    │
Phase 6 ─── Feature store (needs Phase 5 data)
    │
Phase 7 ─── ML experiments (needs Phase 6 features)
    │
Phase 8 ─── Model selection (needs Phase 7 results)
    │
Phase 9 ─── MLflow registry (needs Phase 8 decision)
    │
Phase 10 ── Automation (can run after Phase 1)
Phase 11 ── Monitoring (needs Phase 9 models)
    │
Phase 12 ── FastAPI backend (needs Phase 9 models)
    │
Phase 13 ── Streamlit dashboard (needs Phase 12 API)
    │
Phase 14 ── Deployment (needs Phase 12 + 13)
    │
Phase 15 ── Final documentation (needs all phases)
    │
Phase 16 ── Demo preparation (needs Phase 14 + 15)
```

---

## 5. Key Milestones

| Milestone | Phase | Date | Deliverable |
|---|---|---|---|
| Documentation Foundation | 0 | 1 Aug | 10 docs files created |
| Environment Ready | 1 | 2 Aug | Python 3.11 venv, Docker, deps |
| First Data Collection | 3 | 5 Aug | Real API data flowing |
| Feature Pipeline Working | 4 | 7 Aug | Features generated from raw data |
| Training Dataset Ready | 5 | 8 Aug | Historical data prepared |
| Feature Store Operational | 6 | 10 Aug | Hopsworks + local fallback |
| Models Trained | 7 | 12 Aug | All 4 models with metrics |
| Production Model Selected | 8 | 13 Aug | Evidence-based decision |
| MLflow Tracking Active | 9 | 14 Aug | Experiments and models registered |
| CI/CD Running | 10 | 15 Aug | GitHub Actions workflows |
| Monitoring Active | 11 | 16 Aug | Evidently AI reports |
| API Serving Predictions | 12 | 18 Aug | FastAPI endpoints live |
| Dashboard Live | 13 | 20 Aug | Streamlit fully functional |
| System Deployed | 14 | 21 Aug | Production deployment |
| Documentation Complete | 15 | 22 Aug | All 15 docs files finalized |
| Demo Ready | 16 | 23 Aug | Full system verified |

---

## 6. Risk Mitigation Timeline

| Risk | Mitigation Phase | Strategy |
|---|---|---|
| Hopsworks free-tier limits | Phase 6 | Reduce write frequency; use local fallback |
| API rate limits | Phase 3 | Caching; mock data during development |
| LSTM training time | Phase 7 | Allocate extra time; can fall back to 3 classical models |
| 72h prediction accuracy | Phase 8 | Document realistic expectations per mentor guidance |
| Windows/Hopsworks compatibility | Phase 6 | Environment variable toggle; Codespaces as option |

---

## 7. Resource Requirements

| Resource | Details |
|---|---|
| Python 3.11 | Required for Hopsworks compatibility |
| OpenWeather API Key | Free tier sufficient for development |
| AQICN API Key | Free tier sufficient for fallback |
| Hopsworks Account | Free tier; account setup in Phase 6 |
| MLflow | Local tracking server |
| GitHub Repository | For version control and CI/CD |
| Docker | For containerization |
| Streamlit Cloud | For frontend deployment |
