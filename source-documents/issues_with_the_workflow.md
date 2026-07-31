### Verified Technical & System Issues and Recommended Solutions

Below is the verified list of technical issues identified from the discussion logs, excluding individual user environment setup or skill-related errors, along with proposed suggestions:

#### 1. Hopsworks Free-Tier Compute/Usage Limits & Account Freezing

* **Issue:** Running hourly feature pipelines and deploying models on the Hopsworks free tier rapidly exhausts the allocated compute budget (triggering $10+ usage charges or budget caps), causing user accounts and automated pipelines to freeze.


* **Suggestion / Fix:**

* Reduce the pipeline execution frequency from hourly to daily or every 6–12 hours to conserve free-tier credits.
* Allow students to use lightweight, alternative feature/data storage options (e.g., MongoDB, DuckDB, or version-controlled CSV files stored on GitHub) as a primary or fallback option, provided the codebase maintains Hopsworks integration structure.





#### 2. Hopsworks HDFS Object Store & RPC Listener Disconnections

* **Issue:** Calling `feature_group.insert()` intermittently fails with a `Generic HdfsObjectStore error: RPC listener disconnected` when streaming or saving datasets.


* **Suggestion / Fix:**

* Configure feature groups using Apache Hudi instead of Delta Lake format during feature group creation (`Hudi Configuration`).


* Switch the regional endpoint host from `c.app.hopsworks.ai` to `eu-west.cloud.hopsworks.ai`.





#### 3. Python 3.12 Incompatibility with Hopsworks (`imp` Module Deprecation)

* **Issue:** Python 3.12 removed the legacy `imp` module, causing `ModuleNotFoundError: No module named 'imp'` when importing the Hopsworks/HSFS library.


* **Suggestion / Fix:**

* Explicitly specify Python 3.11 (or 3.10) as the required runtime environment in the project setup instructions and `requirements.txt` file.





#### 4. AQICN / WAQI API Static Data Response (161 Value Lock)

* **Issue:** Fetching live AQI data from the AQICN API (e.g., Karachi station) returns repetitive static target values (e.g., 161) due to long multi-hour update cycles at ground stations.


* **Suggestion / Fix:**

* Use OpenWeatherMap API as the primary live telemetry source for frequent, dynamic hourly variation, while keeping WAQI/AQICN as a secondary fallback with staleness detection to avoid training on duplicate rows.





---

### Project Complexity & Estimated Timeline Summary

#### Is the Project Easy?

**No, it is a moderate-to-advanced end-to-end Machine Learning Operations (MLOps) project.**

While basic machine learning model creation on static datasets can be straightforward, this project requires setting up continuous live data pipelines (APIs), feature stores (Hopsworks/MongoDB), automated workflow scheduling, multi-step forecasting (predicting separate targets for Day 1, Day 2, and Day 3), and deployment via web interfaces (Streamlit). As noted by the project mentors, the core difficulty lies in configuring new MLOps infrastructure and handling real-world data quality and platform integration bugs.

#### Estimated Days to Complete

* **Estimated Time:** **5 to 8 days** of dedicated, active development (or **1.5 to 2 weeks** part-time).
* **Breakdown:**
* **Days 1–2:** API integrations (OpenWeatherMap/AQICN), initial data cleaning, and handling API quirks.


* **Days 3–4:** Feature store setup (Hopsworks/MongoDB), setting up Python 3.11 environment, and configuring automated feature pipelines.


* **Days 5–6:** Training and evaluating forecasting models for multi-day AQI prediction.


* **Days 7–8:** Streamlit dashboard deployment, documentation/reporting, and debugging platform-specific issues.





---

Would you like a detailed architectural layout or script template for setting up the 3-day forecasting pipeline and Streamlit dashboard?
