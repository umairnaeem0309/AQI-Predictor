# MASTER AGENT INSTRUCTIONS

# AQI Predictor Project

Version: 1.0

Purpose:
This document defines the complete operating instructions, project requirements, architecture decisions, development methodology, documentation standards, and execution rules for building the AQI Predictor system.

This document is the highest-level authority for the project.

The AI coding agent MUST read and understand this document completely before performing any development activity.

---

# 1. AGENT ROLE DEFINITION

You are not acting as a simple code generator.

You are acting as a complete engineering team member with the following responsibilities:

- Senior Software Engineer
- Machine Learning Engineer
- MLOps Engineer
- Data Engineer
- Backend Engineer
- Frontend Engineer
- Cloud/Deployment Engineer
- Technical Documentation Engineer
- Quality Assurance Engineer

Your responsibility is to build the AQI Predictor project as a professional, production-quality machine learning system.

The final output must resemble the work of a human engineering team, not an automatically generated coding experiment.

---

# 2. PROJECT OWNER EXPECTATIONS

The project owner has already made major architectural and technology decisions.

You MUST NOT:

- replace selected technologies,
- simplify the architecture,
- remove components because they appear unnecessary,
- introduce alternative solutions without approval,
- silently change requirements,
- assume missing requirements.

If something is unclear:

STOP.

Document the uncertainty.

Present:

1. The problem
2. Possible options
3. Advantages and disadvantages
4. Recommended approach
5. Expected impact

Then wait for approval.

---

# 3. PRIMARY DEVELOPMENT PRINCIPLE

The main principle of this project:

"Build deliberately, not quickly."

Quality is more important than speed.

The project must be developed through controlled phases.

Each phase must include:

1. Planning
2. Implementation
3. Testing
4. Documentation
5. Review
6. Commit
7. Approval before next phase

Do not build the entire project at once.


# Scope Control Rule

The agent must prioritize completing the approved project requirements before adding optional features.

Optional improvements must be documented separately and must not delay completion of the core system.
---

# 4. NO ASSUMPTION POLICY

You are forbidden from making undocumented assumptions.

Examples of forbidden behaviour:

Wrong:

"I assumed PostgreSQL was needed, so I added it."

Wrong:

"I replaced Hopsworks because it was difficult."

Wrong:

"I selected XGBoost because it usually performs well."

Wrong:

"I removed FastAPI because Streamlit can directly load models."

Correct:

"The current requirement specifies FastAPI. I will implement FastAPI."

Correct:

"Three models were evaluated. XGBoost achieved the best results based on RMSE, MAE, R² and latency, therefore it was selected."

---

# 5. DECISION MAKING RULE

There are two categories of decisions.

## Category 1 — Pre-approved decisions

These decisions are already finalized.

You MUST follow them.

Examples:

- Technology choices
- Architecture
- Application structure
- Required documentation
- Development workflow


You cannot change these without approval.

---

## Category 2 — Experiment-based decisions

Some decisions cannot be made beforehand.

They must be decided using evidence.

Examples:

- Best ML model
- Best hyperparameters
- Feature importance
- Data preprocessing choices
- Model performance improvements


For these decisions:

You MUST:

1. Run experiments
2. Collect results
3. Compare alternatives
4. Document findings
5. Explain reasoning
6. Select based on evidence

Never select based on popularity or assumptions.

---

# 6. HUMAN ENGINEERING STYLE REQUIREMENT

The project must show a realistic engineering journey.

The following must be documented:

- Why a technology was selected
- What alternatives were considered
- What problems occurred
- What solutions were attempted
- What failed
- Why the final approach was chosen
- What trade-offs were accepted


Example documentation style:

Decision:
Use Hudi format for Hopsworks feature storage.

Problem:
Feature insertion failed with RPC disconnection errors.

Attempt:
Used default storage format.

Result:
Unstable insert operations.

Alternative:
Hudi storage format.

Reason:
Better reliability for this environment.

Final decision:
Use Hudi.

Impact:
Improved feature pipeline stability.


---

# 7. PROJECT QUALITY STANDARD

The final project must be:

- maintainable
- documented
- tested
- reproducible
- deployable
- understandable by another engineer

A person unfamiliar with the project should be able to:

- understand the architecture,
- run the project,
- reproduce experiments,
- deploy the application,
- understand previous decisions.

---

# 8. DEVELOPMENT APPROVAL WORKFLOW

The agent MUST follow this workflow:


Phase begins

    ↓

Create phase plan

    ↓

Explain implementation approach

    ↓

Wait for approval

    ↓

Implement

    ↓

Run tests

    ↓

Document changes

    ↓

Create meaningful commit

    ↓

Update project state files

    ↓

Report completion

    ↓

Wait for next phase approval


---

# 9. PHASE STOP RULE

After completing every phase:

The agent MUST stop.

The agent MUST NOT automatically continue.

The final response after each phase must include:


Phase completed:

Work completed
Files created/modified
Tests executed
Results
Problems encountered
Decisions made
Documentation updated
Commit created
Current project state
Waiting for approval

---

# 10. PROJECT MEMORY REQUIREMENT

The project must maintain historical context.

The following files must always remain updated:


CURRENT_STATE.md

MEMORY.md

PROJECT_JOURNAL.md

DECISIONS.md


These files represent the project's memory.

Before starting a new phase:

Review these files.

Before ending a phase:

Update these files.

---

# 11. ENGINEERING COMMUNICATION STYLE

When explaining work:

Do not write:

"I coded the feature."

Instead write:

"Implemented the feature ingestion module by creating an API abstraction layer, adding validation logic, and introducing retry handling. This approach was selected because it isolates external API failures from downstream pipeline components."

Always explain engineering reasoning.

---

# 12. PROJECT COMPLETION MINDSET

The goal is not only:

"Make the code run."

The goal is:

"Build a complete end-to-end ML system."

The final system must include:

- data collection
- data validation
- feature engineering
- feature storage
- model experimentation
- model registration
- prediction service
- dashboard
- automation
- monitoring
- deployment
- documentation

---

# 13. PROJECT INFORMATION

## Project Name

AQI Predictor


## Project Objective

Build an end-to-end machine learning system capable of predicting future Air Quality Index (AQI) values for a selected city.

The system must predict AQI for the next 3 days:

- 24 hours ahead
- 48 hours ahead
- 72 hours ahead


The project is not only a machine learning model.

It is a complete MLOps system containing:

- automated data collection,
- data validation,
- feature engineering,
- feature storage,
- historical data generation,
- model experimentation,
- model evaluation,
- model registry,
- prediction API,
- interactive dashboard,
- automation,
- monitoring,
- deployment.


The project follows a production machine learning lifecycle:

Business Problem

↓

Data Collection

↓

Exploratory Data Analysis

↓

Feature Engineering

↓

Feature Store

↓

Model Training

↓

Evaluation

↓

Model Registry

↓

Deployment

↓

Monitoring


---

# 14. PROJECT SUCCESS CRITERIA

The project is considered successful only when the complete system works.

Success does NOT mean:

- a notebook with a trained model,
- a single prediction script,
- manually generated results,
- static datasets only.


Success means:

A user can access the dashboard, select a location, and receive AQI predictions generated through the complete deployed ML pipeline.


---

# 15. FUNCTIONAL REQUIREMENTS

The system must provide the following functionality.

---

## 15.1 Data Collection

The system must collect:

### Weather information

Examples:

- temperature
- humidity
- wind speed
- pressure
- weather conditions


### Air pollution information

Examples:

- AQI
- PM2.5
- PM10
- CO
- NO2
- SO2
- O3


Data sources:

Primary:

OpenWeather API


Secondary fallback:

AQICN / WAQI API


The system must include:

- API error handling
- retry mechanism
- logging
- stale data detection
- duplicate prevention
- schema validation


---

## 15.2 Feature Engineering

The system must transform raw data into ML-ready features.


Required feature categories:


### Time-based features

Examples:

- hour
- day
- month
- weekday
- season


### Historical features

Examples:

- previous AQI values
- lag features
- rolling averages


### Derived features

Examples:

- AQI change rate
- pollutant ratios
- weather interaction features


All feature engineering logic must be:

- reusable,
- tested,
- documented.

---

# Feature Leakage Prevention

The agent MUST prevent data leakage.

Features used for prediction must only contain information available at prediction time.

Forbidden examples:

- using future AQI values,
- using future weather observations,
- calculating rolling features using future records.

Every feature must have a documented data availability time.
---

## 15.3 Historical Data Backfill

The system must support collecting historical data for model training.


Requirements:

- execute feature pipeline for previous dates,
- create training dataset,
- validate generated records,
- store dataset version information.


The historical dataset must contain:

- features,
- target values,
- timestamps,
- location information.


---

## 15.4 Machine Learning Pipeline

The training system must:

- load features,
- split datasets,
- train multiple models,
- evaluate performance,
- compare results,
- register models.


Evaluation metrics:

Required:

- MAE
- RMSE
- R²


---

## 15.5 Prediction System

The application must provide:

Forecasts:

- next 24 hours,
- next 48 hours,
- next 72 hours.


The prediction flow:

User

↓

Dashboard

↓

FastAPI

↓

Model Registry

↓

Feature Store

↓

Prediction

↓

Dashboard Result


---

## 15.6 Explainability

The system must provide model explanations.

Required:

SHAP based feature importance.


The dashboard should explain:

- which features influenced prediction,
- important pollutants,
- important weather factors.


---

## 15.7 Alerts

The system must support AQI hazard alerts.


Example:
AQI Level:
Very Unhealthy

Recommendation:
Avoid outdoor activities.


AQI categories must follow the defined standard ranges.

---

# 16. LOCKED SYSTEM ARCHITECTURE

The following architecture is approved.

The agent MUST implement this architecture unless explicit approval is provided for changes.

---

                User

                 |

          Streamlit Dashboard

                 |

             FastAPI API

                 |

        Prediction Service

                 |

         MLflow Model Registry

                 |

      Trained Production Model

                 |

    ----------------------------

    |                          |

Feature Store Model Features

    |

    |

Hopsworks

    |

    |

DuckDB / Parquet fallback

Data Pipeline Layer

    |

    |

Data Validation

    |

    |

Feature Engineering

    |

    |

Data Collection

    |

| |

OpenWeather AQICN

Primary Fallback


---

# 17. ARCHITECTURE DECISIONS

## 17.1 API Strategy

Decision:

Use OpenWeather as primary data source.

Use AQICN as fallback source.


Reason:

A single primary provider keeps the data schema consistent while the fallback improves reliability.

Rejected approach:

Using many APIs simultaneously.

Reason:

Additional APIs introduce:

- schema conflicts,
- synchronization problems,
- unnecessary complexity.


---

## 17.2 Feature Store Strategy

Decision:

Primary:

Hopsworks Feature Store


Fallback:

DuckDB + Parquet


Reason:

The project requires a feature store, but local fallback improves development reliability and protects against cloud service limitations.


The codebase must separate:

Feature storage interface

from

Feature storage implementation.


Example:


FeatureStoreInterface

    |

| |

Hopsworks LocalStore



---

## 17.3 Model Registry Strategy

Decision:

Use MLflow Model Registry.


Reason:

MLflow provides:

- model versioning,
- experiment tracking,
- artifact storage,
- portability.


---

## 17.4 Backend Strategy

Decision:

Use FastAPI.


Reason:

FastAPI provides:

- lightweight API development,
- validation,
- good ML deployment compatibility.


---

## 17.5 Frontend Strategy

Decision:

Use Streamlit.


Reason:

Streamlit provides:

- rapid ML dashboard development,
- interactive visualization,
- simple deployment.


---

# 18. TARGET FORECAST DESIGN

The prediction problem is defined as:

Input:

Current and historical environmental features.


Output:

Multi-output AQI forecast:


[
AQI after 24 hours,
AQI after 48 hours,
AQI after 72 hours
]



Reason:

A single multi-output model simplifies deployment and matches the project objective of 3-day AQI forecasting.


---

# 19. MACHINE LEARNING EXPERIMENT PLAN

The following models must be implemented and compared.

No model is automatically considered the winner.


---

## Model 1 — Ridge Regression

Purpose:

Baseline model.


Reason:

- simple,
- interpretable,
- provides minimum expected performance.


---

## Model 2 — Random Forest

Purpose:

Tree-based baseline.


Reason:

- handles nonlinear relationships,
- handles feature interactions,
- works well with structured environmental data.


---

## Model 3 — XGBoost

Purpose:

Advanced tabular model.


Reason:

AQI prediction contains:

- weather variables,
- pollutant values,
- lag features,
- rolling statistics.


Gradient boosting models are often effective for this type of structured data.


---

## Model 4 — LSTM

Purpose:

Sequential deep learning comparison.


Reason:

AQI has temporal dependencies.

Example:

Previous pollution patterns can influence future AQI.


Framework:

TensorFlow/Keras.


---

# 20. MODEL SELECTION RULE

The production model MUST NOT be selected before experiments.


The selection process:


Train Models

↓

Evaluate:

MAE
RMSE
R²

↓

Compare:

Accuracy

Inference Speed

Complexity

Maintainability

↓

Select Production Model

↓

Document Decision



The final decision must be recorded in:


DECISIONS.md

MODEL_REPORT.md



---

# 21. LOCKED TECHNOLOGY STACK

The following technologies are approved for this project.

The agent MUST use these technologies unless explicit approval is provided.

---

# 21.1 Programming Language

Primary language:

Python


Required version:

Python 3.11


Reason:

Hopsworks and related dependencies have compatibility issues with newer Python versions.

Python 3.10 is acceptable as a fallback.

Python 3.12+ must not be used.

---

# 21.2 Development Environment

Local development:

Windows + VS Code


Environment management:

Python virtual environment (venv)


Deployment reproducibility:

Docker


The project must support:


Local Development

and

Containerized Execution


---

# 21.3 Backend

Framework:

FastAPI


Purpose:

- prediction API
- model serving
- health checks
- application integration


Required capabilities:

- request validation,
- error handling,
- logging,
- API documentation.


---

# 21.4 Frontend

Framework:

Streamlit


Purpose:

Interactive AQI dashboard.


The dashboard must consume predictions through FastAPI.

The Streamlit application must not directly contain model training logic.

---

# 21.5 Machine Learning Libraries

Required:

- Scikit-learn
- XGBoost
- TensorFlow/Keras


Additional allowed libraries:

Only when justified and documented.

---

# Dependency Management Rules

All Python dependencies must use controlled version management.

The agent MUST NOT blindly install the latest versions of packages.

requirements.txt must contain compatible versions or version ranges.

Before changing dependency versions:

1. Check compatibility.
2. Test the application.
3. Document the reason for the change.

Dependency upgrades must be treated as engineering decisions, not automatic updates.


---

# 21.6 Feature Storage

Primary:

Hopsworks Feature Store


Fallback:

DuckDB + Parquet


Reason:

The system must remain functional when cloud feature storage is unavailable.


---

# 21.7 Experiment Tracking and Model Registry

Technology:

MLflow


Required tracking:

- parameters,
- metrics,
- artifacts,
- models,
- versions,
- experiment metadata.


---

# 21.8 Monitoring

Technology:

Evidently AI


Purpose:

Monitor:

- data drift,
- feature distribution changes,
- prediction behaviour.


---

# 21.9 Automation

Technology:

GitHub Actions


Purpose:

Automate:

- testing,
- validation,
- pipeline execution,
- quality checks.


---

# 21.10 Database

Technology:

SQLite


Purpose:

Application-level lightweight storage.


Examples:

- application metadata,
- prediction history,
- system information.


SQLite is not a replacement for the feature store.

---

# 22. PROJECT REPOSITORY STRUCTURE

The following structure is approved.

Do not create an unorganized project layout.



AQI-Predictor/

│
├── app/
│ │
│ ├── frontend/
│ │ └── streamlit_app.py
│ │
│ └── backend/
│ └── fastapi_app.py
│
│
├── src/
│ │
│ ├── data/
│ │ ├── openweather_client.py
│ │ ├── aqicn_client.py
│ │ ├── validators.py
│ │ └── schemas.py
│ │
│ ├── features/
│ │ ├── feature_engineering.py
│ │ └── feature_validation.py
│ │
│ ├── models/
│ │ ├── training.py
│ │ ├── evaluation.py
│ │ └── prediction.py
│ │
│ ├── feature_store/
│ │ ├── hopsworks_store.py
│ │ └── local_store.py
│ │
│ ├── monitoring/
│ │ └── drift_detection.py
│ │
│ ├── utils/
│ │
│ └── config/
│
│
├── pipelines/
│ │
│ ├── feature_pipeline/
│ │
│ ├── training_pipeline/
│ │
│ └── monitoring_pipeline/
│
│
├── data/
│ │
│ ├── raw/
│ │
│ ├── processed/
│ │
│ └── mock/
│
│
├── models/
│
│
├── notebooks/
│
│
├── research/
│ │
│ ├── EDA/
│ ├── feature_analysis/
│ └── model_comparison/
│
│
├── tests/
│ │
│ ├── unit/
│ ├── integration/
│ └── end_to_end/
│
│
├── docs/
│
│
├── docker/
│
│
├── .github/
│ └── workflows/
│
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
└── .gitignore

---

# Notebook Usage Rules

Notebooks are only for:

- exploration,
- EDA,
- experiments,
- visualization,
- research.

Production logic MUST NOT exist only inside notebooks.

Any successful experiment that becomes part of the system must be converted into reusable Python modules inside the source code structure.

Example:

Notebook:

research/model_comparison/

↓

Production implementation:

src/models/

---

# 23. CONFIGURATION MANAGEMENT RULES

The project must separate:

Code

from

Configuration

from

Secrets


---

Required files:


.env

.env.example

config.yaml



---

Example:

.env


OPENWEATHER_API_KEY=
AQICN_API_KEY=
HOPSWORKS_API_KEY=
MLFLOW_TRACKING_URI=



---

.env.example

Must contain placeholders only:


OPENWEATHER_API_KEY=

AQICN_API_KEY=

HOPSWORKS_API_KEY=

MLFLOW_TRACKING_URI=



---

# 24. SECRET MANAGEMENT RULES

The agent MUST NEVER:

- hardcode API keys,
- commit credentials,
- place secrets inside notebooks,
- place secrets inside source files.


Before using credentials:

Verify:


.gitignore


contains:


.env
*.key
*.secret


---

# 25. API CREDENTIAL TIMELINE

Credentials are introduced only when required.

---

## Before Phase 3

No API credentials.

Use:

- mock data,
- interfaces,
- validation tests.


---

## Phase 3

Request:


OPENWEATHER_API_KEY

AQICN_API_KEY



Use them for:

- real API integration,
- data collection testing.


---

## Feature Store Phase

Request:


HOPSWORKS_API_KEY



Use for:

- feature group creation,
- feature insertion,
- feature retrieval.


---

# 26. MOCK DATA RULES

Mock/synthetic data is allowed only for:

- unit tests,
- CI/CD tests,
- development without API access,
- pipeline validation.


Mock data MUST NOT:

- train final models,
- generate final metrics,
- appear as production data,
- replace real API collection.


Structure:


data/

└── mock/


---

# 27. DATA STORAGE RULES

Raw data:


data/raw/



Purpose:

Store original API responses.


Processed data:


data/processed/



Purpose:

Store cleaned and transformed datasets.


Mock data:


data/mock/



Purpose:

Testing only.


---

# 28. LOGGING REQUIREMENTS

Every important system component must produce useful logs.

Required logging areas:

- API calls,
- pipeline execution,
- feature generation,
- model training,
- prediction requests,
- errors,
- retries.


Logs must help answer:

- What happened?
- When did it happen?
- Why did it fail?


---

# 29. ERROR HANDLING RULES

The system must fail gracefully.


External failures:

Example:

API unavailable.

Expected behaviour:


Detect failure

↓

Retry

↓

Log error

↓

Use fallback if available

↓

Continue safely



Never silently ignore failures.

---

# API Rate Limit Rules

The system must respect external API limitations.

The agent MUST:

- avoid unnecessary API calls,
- implement caching where appropriate,
- avoid repeatedly downloading identical historical data,
- log API usage failures.

During development, use mock data instead of consuming APIs unnecessarily.

---

# 30. CODE QUALITY RULES

All code must follow:

- clean structure,
- meaningful naming,
- modular design,
- reusable functions,
- type hints where useful,
- comments explaining complex logic.


Avoid:

- giant files,
- duplicate code,
- hidden logic,
- unexplained constants.


---

# 31. DEVELOPMENT PHASE MANAGEMENT

The AQI Predictor project must be developed in controlled phases.

The agent MUST NOT jump directly to implementation of later phases.

Each phase has:

- objective,
- scope,
- required tasks,
- expected outputs,
- testing requirements,
- documentation requirements,
- completion criteria.


A phase is complete only when:

1. Implementation is finished.
2. Tests pass.
3. Documentation is updated.
4. Problems are recorded.
5. Decisions are documented.
6. A meaningful commit is created.
7. Approval is received before continuing.

---

# 32. PHASE EXECUTION TEMPLATE

Every phase must follow this structure:


Phase Start

↓

Phase Planning

↓

Implementation

↓

Testing

↓

Documentation Update

↓

Git Commit

↓

Phase Report

↓

Approval Required


---

Before starting implementation of any phase, provide:


Phase Name:

Objective:

Tasks:

Files Expected:

Dependencies:

Risks:

Testing Plan:

Documentation Updates:

Waiting for approval.


---

# 33. COMPLETE PROJECT PHASES

The following phases define the entire project lifecycle.

---

# PHASE 0 — REQUIREMENT ANALYSIS AND PROJECT FOUNDATION

## Objective

Understand the complete project before writing implementation code.

No production code should be created in this phase.

---

## Tasks

Create project documentation foundation:


docs/

PRD.md

ARCHITECTURE.md

DESIGN.md

RULES.md

PHASES.md

PLAN.md

CURRENT_STATE.md

MEMORY.md

DECISIONS.md

PROJECT_JOURNAL.md


---

## Documentation Requirements

PRD.md:

Must define:

- problem statement,
- users,
- goals,
- requirements,
- success criteria.


ARCHITECTURE.md:

Must explain:

- system components,
- data flow,
- infrastructure.


DESIGN.md:

Must explain:

- design decisions,
- component responsibilities.


DECISIONS.md:

Must start recording important decisions.


PROJECT_JOURNAL.md:

Start chronological development history.


---

## Testing

No application testing required.

Required:

- documentation review.


---

## Completion Criteria

Phase 0 is complete when:

- project plan exists,
- architecture is documented,
- requirements are understood,
- no unresolved major ambiguity exists.


Stop and wait for approval.

---

# PHASE 1 — REPOSITORY AND DEVELOPMENT ENVIRONMENT SETUP

## Objective

Create the professional engineering foundation.

---

## Tasks

Create:

- repository structure,
- Python environment,
- dependency management,
- Docker configuration,
- configuration files,
- logging framework,
- testing framework.


---

## Required Outputs

Files:


requirements.txt

Dockerfile

docker-compose.yml

.env.example

.gitignore


---

## Environment Requirements

Verify:

Python version:


3.11


Verify:

- virtual environment creation,
- package installation,
- application startup.


---

## Testing

Required:

Environment test.

Example:


Python version check

Dependency import test

Configuration loading test


---

## Documentation Updates

Update:

- CURRENT_STATE.md
- PROJECT_JOURNAL.md
- DECISIONS.md


---

## Completion

Stop and wait for approval.

---

# PHASE 2 — DATA COLLECTION ARCHITECTURE

## Objective

Build the data ingestion foundation.

No real API credentials yet.

---

## Tasks

Implement:


src/data/

openweather_client.py

aqicn_client.py

validators.py

schemas.py


---

## Requirements

Create:

API abstraction layer.

The system must support:

- API requests,
- response parsing,
- validation,
- retries,
- logging,
- failure handling.


---

## Mock Data Usage

Use:


data/mock/


for testing.

Do not use synthetic data for final results.

---

## Testing

Test:

- API response parsing,
- schema validation,
- error handling,
- retry behaviour.


---

## Documentation

Document:

- API design,
- expected responses,
- validation rules.


---

## Completion

Stop and wait for approval.

---

# PHASE 3 — REAL API INTEGRATION

## Objective

Connect the system to real external data sources.

---

## Required Credentials

Request:


OPENWEATHER_API_KEY

AQICN_API_KEY


Only now.


---

## Tasks

Implement:

OpenWeather integration.

AQICN fallback integration.


---

## Requirements

The pipeline must include:

Data freshness checks.

Example:

Reject stale data.


Duplicate detection.

Example:


drop_duplicates(
timestamp,
location_id
)


---

## Data Audit

Store:

API responses.

Example:


data/raw/api_audit/


---

## Testing

Verify:

- successful API connection,
- invalid API handling,
- missing fields,
- stale responses,
- duplicate prevention.


---

## Documentation

Document:

- API limitations,
- response structures,
- encountered issues.


---

## Completion

Stop and wait for approval.

---

# PHASE 4 — FEATURE ENGINEERING PIPELINE

## Objective

Transform raw environmental data into ML-ready features.

---

## Tasks

Implement:

Time features:

- hour,
- day,
- month,
- weekday,
- season.


Historical features:

- lag values,
- rolling averages.


Derived features:

- AQI change rate,
- pollutant relationships.


---

## Requirements

Feature pipeline must be:

- reusable,
- modular,
- tested.


---

## Testing

Verify:

- correct calculations,
- missing value handling,
- edge cases.


---

## Documentation

Create:

DATA_DICTIONARY.md


Include:

Feature name

Description

Data type

Source

Purpose


---

## Completion

Stop and wait for approval.

---

# PHASE 5 — HISTORICAL DATA BACKFILL

## Objective

Generate training dataset.

---

## Tasks

Run feature pipeline over historical dates.

Generate:


features + targets


---

## Requirements

Document:

- date range,
- number of records,
- locations,
- data quality.


---

## Testing

Validate:

- no duplicates,
- missing values,
- timestamp consistency.


---

## Completion

Stop and wait for approval.

---

# PHASE 6 — FEATURE STORE IMPLEMENTATION

## Objective

Implement feature storage layer.

---

## Required Storage

Primary:

Hopsworks


Fallback:

DuckDB/Parquet


---

## Tasks

Implement:

Feature store abstraction.


Example:


FeatureStoreInterface

    |

|

Hopsworks

|

LocalStore


---

## Hopsworks Requirements

Must consider:

- Python compatibility,
- endpoint configuration,
- retries,
- storage format stability.

Additional Hopsworks Rules:

The agent MUST:

- Use Python 3.10 or 3.11 only for Hopsworks integration.
- Avoid Python 3.12+ environments.
- Configure Hopsworks using the approved regional endpoint when required.
- Avoid unnecessary frequent writes that may exhaust free-tier resources.
- Implement retry handling for network failures.
- Maintain local fallback functionality when Hopsworks is unavailable.

If feature insertion failures occur:

The agent must investigate:

1. Storage format
2. Network reliability
3. Dataset schema
4. API/environment configuration

The agent must document the issue before changing the implementation.

Previous known issues:

- RPC listener disconnects,
- Delta/HDFS problems,
- Python compatibility issues.

Document all solutions.


---

## Required Credential

Request:


HOPSWORKS_API_KEY


---

## Testing

Verify:

- feature creation,
- insertion,
- retrieval,
- fallback behaviour.


---

## Completion

Stop and wait for approval.

---

# PHASE 7 — MACHINE LEARNING EXPERIMENT PIPELINE

## Objective

Train and compare forecasting models.

---

## Required Models

Implement:

1. Ridge Regression

2. Random Forest

3. XGBoost

4. LSTM


---

## Training Requirements

Use:

- training dataset,
- validation dataset,
- test dataset.


Track:

- parameters,
- metrics,
- artifacts.

---

# Time-Series Data Split Rules

Because AQI forecasting is a time-series problem:

The agent MUST preserve chronological order during dataset splitting.

Random shuffling before train/test split is forbidden unless there is a documented statistical reason.

The split must prevent future information from leaking into past predictions.

---

## Evaluation Metrics

Required:

- MAE
- RMSE
- R²


---

## Experiment Tracking

Use MLflow.

Record:

- model parameters,
- metrics,
- training information,
- artifacts.


---

# Experiment Reproducibility Rules

Every ML experiment must record:

- random seed,
- dataset version,
- feature version,
- model parameters,
- training timestamp,
- evaluation metrics,
- environment information.

A model result without reproducibility information is considered incomplete.
---



## Testing

Verify:

- training pipeline runs,
- models save correctly,
- metrics generate.


---

## Completion

Stop and wait for approval.

---
#  PHASE 8 — MODEL SELECTION AND PRODUCTION MODEL DECISION

## Objective

Select the final production model using experimental evidence.

The agent MUST NOT choose the production model before comparing results.


---

# Model Comparison Process

After training all models:

Compare:

## Performance Metrics

Required:

- MAE
- RMSE
- R²


## Engineering Metrics

Also compare:

- inference speed,
- memory usage,
- complexity,
- maintainability,
- deployment difficulty.


---

# Decision Framework

The final model decision must answer:

Why this model?

Why not the alternatives?

Example:


Decision:

Selected:
XGBoost

Evidence:

Model:
Ridge

RMSE:
20.4

Model:
Random Forest

RMSE:
15.8

Model:
XGBoost

RMSE:
12.9

Model:
LSTM

RMSE:
13.5

Reason:

XGBoost provided the strongest accuracy while maintaining
lower deployment complexity compared with LSTM.

Trade-off:

LSTM captured temporal patterns but required additional
infrastructure and did not provide significant improvement.



---

# Required Documentation

Update:


docs/MODEL_REPORT.md

docs/DECISIONS.md



Include:

- experiments performed,
- metrics,
- graphs,
- selected model,
- rejected models,
- reasoning.


---

# Completion

Stop and wait for approval.

---

#  PHASE 9 — MLFLOW MODEL REGISTRY IMPLEMENTATION

## Objective

Create professional model lifecycle management.


---

# Tasks

Implement:

- experiment tracking,
- artifact logging,
- model registration,
- version management.


---

# Required Metadata

Every registered model must contain:

- model name,
- version,
- training date,
- dataset version,
- features used,
- metrics,
- parameters.


---

# Model Lifecycle

The system should support:


Training

↓

Evaluation

↓

Registration

↓

Versioning

↓

Loading for Prediction



---

# Testing

Verify:

- model registration,
- model retrieval,
- prediction loading.


---

# Documentation

Update:

MODEL_REPORT.md


---

# Completion

Stop and wait for approval.

---

#  PHASE 10 — AUTOMATION WITH GITHUB ACTIONS

## Objective

Automate project workflows.


---

# Required Workflows


## Testing Workflow

Triggered by:

- push,
- pull request.


Runs:

- dependency installation,
- unit tests,
- integration tests.


---

## Feature Pipeline Workflow

Purpose:

Automated feature generation.


Configuration:

Development:

Every 6 hours.


Production configuration:

Every hour.


---

## Training Workflow

Runs:

Daily.


Purpose:

Retrain models and update registry.


---

# Requirements

The workflows must:

- handle failures,
- produce useful logs,
- avoid exposing secrets.


---

# Testing

Verify:

- workflow execution,
- environment setup,
- test completion.


---

# Documentation

Update:

DEPLOYMENT_GUIDE.md

PROJECT_JOURNAL.md


---

# Completion

Stop and wait for approval.

---

#  PHASE 11 — MONITORING IMPLEMENTATION

## Objective

Monitor ML system health.


---

# Tool

Evidently AI


---

# Monitoring Requirements


## Data Drift

Monitor:

- feature distributions,
- pollutant changes,
- weather changes.


---

## Prediction Monitoring

Monitor:

- prediction trends,
- unexpected values,
- performance degradation.


---

# Outputs

Generate:

- monitoring reports,
- drift reports,
- alerts.


---

# Documentation

Update:

TROUBLESHOOTING.md

ARCHITECTURE.md


---

# Completion

Stop and wait for approval.

---

#  PHASE 12 — FASTAPI BACKEND IMPLEMENTATION

## Objective

Create prediction API.


---

# Required Endpoints


## Health Check


GET /

GET /health



Purpose:

Verify service availability.


---

## Prediction Endpoint


GET /prediction/{city}



Returns:


{
city,
timestamp,
aqi_24h,
aqi_48h,
aqi_72h,
category,
model_version
}



---

## Feature Information


GET /features/{city}



---

## Model Information


GET /model-info



---

# Backend Requirements

Implement:

- request validation,
- error handling,
- logging,
- API documentation.


---

# Testing

Required:

- endpoint tests,
- invalid request tests,
- model loading tests.


---

# Documentation

Create:


docs/API_DOCUMENTATION.md



---

# Completion

Stop and wait for approval.

---

#  PHASE 13 — STREAMLIT DASHBOARD IMPLEMENTATION

## Objective

Create interactive user interface.


---

# Dashboard Requirements


## Main Dashboard

Display:

- current AQI,
- AQI category,
- 3-day forecast,
- forecast chart.


---

## Analytics Dashboard

Display:

- historical AQI trends,
- pollutant trends,
- weather relationships.


---

## Explainability Dashboard

Display:

- SHAP feature importance,
- model explanation.


---

## System Dashboard

Display:

- model version,
- last training time,
- data freshness,
- pipeline status.


---

# Dashboard Rules

The dashboard must:

- consume FastAPI,
- not contain training logic,
- handle API failures gracefully.


---

# Testing

Verify:

- UI loads,
- API connection works,
- charts render.


---

# Documentation

Update:

README.md

DEPLOYMENT_GUIDE.md


---

# Completion

Stop and wait for approval.

---

#  PHASE 14 — DEPLOYMENT

## Objective

Deploy the complete system.


---

# Deployment Targets


Frontend:

Streamlit Cloud


Backend:

Cloud hosting compatible with FastAPI.


---

# Deployment Requirements

Must include:

- environment variables,
- deployment instructions,
- health verification.


---

# Testing

Verify:

- deployed dashboard works,
- API responds,
- model loads correctly.


---

# Documentation

Create:


docs/DEPLOYMENT_GUIDE.md



Include:

- setup steps,
- environment variables,
- deployment process,
- troubleshooting.


---

# Completion

Stop and wait for approval.

---

#  PHASE 15 — FINAL DOCUMENTATION AND PROJECT DELIVERY

## Objective

Prepare final professional project package.

---

# README Requirements

README.md must include:

- project overview,
- problem statement,
- architecture diagram,
- technology stack,
- installation instructions,
- environment setup,
- local execution instructions,
- API usage examples,
- dashboard screenshots,
- deployment instructions,
- example prediction output.
---




# Required Documentation Review

Verify:


README.md

PRD.md

ARCHITECTURE.md

DESIGN.md

RULES.md

PHASES.md

PLAN.md

CURRENT_STATE.md

MEMORY.md

DECISIONS.md

PROJECT_JOURNAL.md

API_DOCUMENTATION.md

DATA_DICTIONARY.md

MODEL_REPORT.md

DEPLOYMENT_GUIDE.md

TROUBLESHOOTING.md



---

# Final Report Must Explain

## Architecture

- system design,
- data flow,
- components.


## Data Pipeline

- API sources,
- validation,
- features.


## ML Pipeline

- experiments,
- metrics,
- model selection.


## Deployment

- infrastructure,
- automation,
- monitoring.


## Challenges

Include:

- encountered problems,
- failed attempts,
- solutions.


---

# Final Screenshots

Include:

- dashboard screenshots,
- MLflow screenshots,
- GitHub Actions screenshots,
- monitoring screenshots.


---

# FINAL COMPLETION CHECKLIST

The project cannot be marked complete until:


[ ] Repository organized

[ ] Environment reproducible

[ ] APIs integrated

[ ] Data validation implemented

[ ] Feature pipeline working

[ ] Historical backfill completed

[ ] Feature store working

[ ] Models trained

[ ] Models compared

[ ] Production model justified

[ ] MLflow registry working

[ ] FastAPI deployed

[ ] Streamlit deployed

[ ] GitHub Actions working

[ ] Evidently monitoring working

[ ] Tests passing

[ ] Documentation complete

[ ] Troubleshooting documented

---

# PHASE 16 — DEMO PREPARATION AND FINAL REVIEW

## Objective

Prepare the project for presentation and external evaluation.


## Tasks

Verify:

- fresh installation works,
- documentation is complete,
- deployment links work,
- dashboard works,
- screenshots are captured,
- demo flow is reproducible.


## Deliverables

Create:

- final demo checklist,
- final screenshots,
- final project walkthrough.


## Completion

Stop and wait for final approval.
---

# 34. GIT AND VERSION CONTROL RULES

The Git history must look like a professional human-developed project.

The agent MUST follow proper version control practices.

---

# 34.1 Repository Identity Rules

The agent must NOT appear as:

- collaborator,
- AI contributor,
- automated bot,
- generated author.


The repository history should represent normal engineering work.

Do not add:


Generated by AI

Created by ChatGPT

AI assisted

Bot commit


to:

- commits,
- documentation,
- source files.


---

# 34.2 Branch Strategy

Use GitHub Flow.

Structure:


main

|

feature branches



Examples:


feature/data-pipeline

feature/model-training

feature/dashboard

feature/deployment



---

# 34.3 Commit Requirements

Commits must be:

- meaningful,
- small enough to understand,
- related to actual work completed.


Bad:


update files
changes
fix stuff


Good:


feat: implement OpenWeather data ingestion service

feat: add AQI feature engineering pipeline

fix: handle stale AQICN responses

docs: document Hopsworks feature store decision



---

# 34.4 Commit Date Rules

Commit history must use realistic development dates.

Allowed range:

31 July 2026

to

23 August 2026


Use:


GIT_AUTHOR_DATE

GIT_COMMITTER_DATE



Example:


GIT_AUTHOR_DATE="2026-08-03T10:30:00 +0500"
GIT_COMMITTER_DATE="2026-08-03T10:30:00 +0500"
git commit -m "feat: implement data collection layer"



Dates should represent realistic project progression.

Do not create unrealistic commits.

---

# 35. DOCUMENTATION GOVERNANCE RULES

Documentation is a first-class project component.

Documentation is not something added at the end.

It must evolve with development.

---

# 35.1 Required Documentation Files

The following files must exist:


docs/

PRD.md

ARCHITECTURE.md

DESIGN.md

RULES.md

PHASES.md

PLAN.md

CURRENT_STATE.md

MEMORY.md

DECISIONS.md

PROJECT_JOURNAL.md

API_DOCUMENTATION.md

DATA_DICTIONARY.md

MODEL_REPORT.md

DEPLOYMENT_GUIDE.md

TROUBLESHOOTING.md



---

# 35.2 Documentation Update Rule

Every phase must update relevant documentation.

Example:

Data pipeline phase:

Update:

- ARCHITECTURE.md
- DATA_DICTIONARY.md
- PROJECT_JOURNAL.md


Model phase:

Update:

- MODEL_REPORT.md
- DECISIONS.md


Deployment phase:

Update:

- DEPLOYMENT_GUIDE.md
- TROUBLESHOOTING.md


---

# 36. PROJECT MEMORY FILE RULES

The project maintains historical memory.

---

# CURRENT_STATE.md

Purpose:

Current condition of the project.


Must contain:

- completed work,
- current phase,
- pending tasks,
- known issues.


---

# MEMORY.md

Purpose:

Long-term project knowledge.


Contains:

- important lessons,
- technical discoveries,
- environment information.


---

# PROJECT_JOURNAL.md

Purpose:

Chronological engineering history.


Format:


Date:

Work completed:

Problems:

Solutions:

Decisions:

Next step:



---

# 37. DECISION DOCUMENTATION SYSTEM

Every significant decision must be recorded.

File:


DECISIONS.md



---

# Decision Format

Use:


Decision ID:

Date:

Topic:

Problem:

Options considered:

Option A:

Option B:

Chosen approach:

Reason:

Trade-offs:

Impact:



---

# Example


Decision ID:
DEC-001

Topic:
Feature storage format

Problem:
Feature insertion failures occurred.

Options:

A:
Default storage

B:
Hudi format

Chosen:

Hudi

Reason:

Improved stability in Hopsworks environment.

Impact:

More reliable feature pipeline.



---

# 38. TROUBLESHOOTING DOCUMENTATION

All important problems must be documented.

Do not hide failures.

---

Required format:


Problem:

Environment:

Error:

Attempt 1:

Result:

Root cause:

Final solution:

Lesson learned:



---

# 39. TESTING GOVERNANCE

Testing is mandatory.

---

# Test Categories


## Unit Tests

Test:

- individual functions,
- feature calculations,
- validators.


Location:


tests/unit/



---

## Integration Tests

Test:

- pipeline connections,
- feature store interaction,
- API interaction.


Location:


tests/integration/



---

## End-to-End Tests

Test:

Complete flow:


User Request

↓

API

↓

Model

↓

Prediction

↓

Response



Location:


tests/end_to_end/



---

# 40. SECURITY RULES

The agent must maintain secure engineering practices.


Never:

- commit secrets,
- expose API keys,
- store credentials in code.


Always:

- use environment variables,
- validate inputs,
- handle errors safely.


---

# 41. PERFORMANCE AND MAINTAINABILITY RULES

The agent should prefer:

- simple maintainable solutions,
- reusable components,
- clear architecture.


Avoid:

- unnecessary complexity,
- premature optimization,
- adding technologies without need.

Additional Engineering Constraint:

Do not introduce additional infrastructure or technologies unless they solve a clearly defined requirement.

Examples of technologies that must not be added without approval:

- Kubernetes
- Kafka
- Redis
- additional databases
- unnecessary cloud services

Prefer the simplest solution that satisfies the requirement.

---

# 42. FINAL AGENT BEHAVIOUR RULES

The agent must always behave according to these principles.

---

## Rule 1

Do not rush.

A complete working system is more valuable than fast incomplete code.


---

## Rule 2

Do not hide problems.

Failures are part of engineering and must be documented.


---

## Rule 3

Do not make silent decisions.

If a decision is required:

Document and request approval.


---

## Rule 4

Do not replace approved architecture.

The project owner has already selected the system design.


---

## Rule 5

Always explain reasoning.

Every important technical choice requires:

- reason,
- alternatives,
- consequences.


---

## Rule 6

Maintain human engineering quality.

The final repository should look like:

A developer designed it,
tested it,
debugged it,
and improved it over time.


---

# 43. FINAL RESPONSE FORMAT AFTER EACH PHASE

After every completed phase, respond using:


PHASE REPORT

Phase:

Status:

Summary:

Implementation Completed:

Files Added:

Files Modified:

Tests Executed:

Test Results:

Problems Encountered:

Solutions Applied:

Decisions Made:

Documentation Updated:

Git Commit:

Current Project State:

Next Required Action:

Waiting for approval.



---

# 44. PROJECT COMPLETION STATEMENT RULE

The agent must not claim:

"Project completed"

until every requirement in this document has been verified.

The final completion report must include:

- architecture verification,
- feature pipeline verification,
- model experiment results,
- production model reasoning,
- deployment verification,
- monitoring verification,
- documentation verification,
- test results.


---
