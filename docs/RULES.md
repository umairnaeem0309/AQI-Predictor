# Rules Document

## AQI Predictor — Development Rules and Constraints

**Version:** 1.0  
**Date:** 31 July 2026  
**Status:** Phase 0 — Foundation  

---

## 1. Authority

This document summarizes the development rules from MASTER_AGENT_INSTRUCTIONS.md. In case of conflict, MASTER_AGENT_INSTRUCTIONS.md takes precedence.

---

## 2. Code Quality Rules

- Clean directory structure with meaningful naming
- Modular design with single-responsibility modules
- Type hints on function signatures where useful
- Comments explaining complex logic only
- No giant files; keep files focused and small
- No duplicate code; extract shared logic into utilities
- No hidden logic or unexplained constants

---

## 3. No Assumption Policy

**Forbidden behaviors:**
- Replacing approved technologies
- Simplifying architecture without approval
- Making undocumented decisions
- Adding technologies without justification
- Assuming missing requirements

**Required behavior:**
When something is unclear → STOP → Document uncertainty → Present options → Wait for approval.

---

## 4. Decision Categories

### Category 1 — Pre-Approved (Follow exactly)
- Technology stack (Python 3.11, FastAPI, Streamlit, MLflow, Hopsworks, etc.)
- Architecture design
- Application structure
- Required documentation
- Development workflow

### Category 2 — Experiment-Based (Must justify with evidence)
- Best ML model
- Best hyperparameters
- Feature importance rankings
- Data preprocessing choices
- Model performance improvements

---

## 5. Mock Data Rules

- Mock/synthetic data allowed **only** for: unit tests, CI/CD, development without API access, pipeline validation
- Mock data **must not** train final models, generate final metrics, appear as production data, or replace real API collection
- Mock data stored in `data/mock/`
- 90 days of synthetic data for development and testing

---

## 6. Data Storage Rules

| Data Type | Location | Purpose |
|---|---|---|
| Raw API responses | `data/raw/` | Original API responses |
| Cleaned datasets | `data/processed/` | Transformed data |
| Mock/synthetic | `data/mock/` | Testing only |

---

## 7. Secret Management Rules

**Never:**
- Hardcode API keys
- Commit credentials
- Place secrets inside notebooks or source files

**Always:**
- Use environment variables
- Verify `.gitignore` contains `.env`, `*.key`, `*.secret`

---

## 8. Logging Requirements

Every important system component must produce useful logs:
- API calls and responses
- Pipeline execution progress
- Feature generation statistics
- Model training progress and results
- Prediction requests and responses
- All errors and retries

Logs must answer: What happened? When? Why did it fail?

---

## 9. Testing Requirements

| Category | Tests | Location |
|---|---|---|
| Unit | Individual functions, feature calculations, validators | `tests/unit/` |
| Integration | Pipeline connections, feature store interaction, API interaction | `tests/integration/` |
| End-to-End | Complete flow: User → API → Model → Prediction → Response | `tests/end_to_end/` |

Testing is mandatory. A phase is not complete until tests pass.

---

## 10. Documentation Requirements

All 15 documentation files must exist and be maintained throughout development:

```
docs/
├── PRD.md
├── ARCHITECTURE.md
├── DESIGN.md
├── RULES.md
├── PHASES.md
├── PLAN.md
├── CURRENT_STATE.md
├── MEMORY.md
├── DECISIONS.md
├── PROJECT_JOURNAL.md
├── API_DOCUMENTATION.md
├── DATA_DICTIONARY.md
├── MODEL_REPORT.md
├── DEPLOYMENT_GUIDE.md
└── TROUBLESHOOTING.md
```

Every phase must update relevant documentation.

---

## 11. Git Rules

- Use GitHub Flow (main + feature branches)
- Feature branch naming: `feature/data-pipeline`, `feature/model-training`, etc.
- Commit messages: conventional format (`feat:`, `fix:`, `docs:`, etc.)
- Commits must be meaningful and related to actual work
- Commit dates: 31 July 2026 to 23 August 2026 (realistic progression)
- No AI-generated author references in commits, docs, or source files
- Use `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` for realistic dates

---

## 12. Dependency Management Rules

- All dependencies use controlled version management in `requirements.txt`
- No blind installation of latest package versions
- Before changing dependency versions: check compatibility → test → document reason
- Dependency upgrades treated as engineering decisions

---

## 13. Performance Rules

- Prefer simple, maintainable solutions
- Reusable components over one-off implementations
- Clear architecture over premature optimization
- Do not introduce additional infrastructure without approval (no Kubernetes, Kafka, Redis, etc.)

---

## 14. Security Rules

- Never commit secrets or expose API keys
- Always use environment variables for credentials
- Always validate inputs
- Handle errors safely without exposing internal details
