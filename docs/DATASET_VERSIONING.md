# Dataset Versioning

## AQI Predictor — Dataset Versioning Approach

**Version:** 1.0  
**Last Updated:** 21 August 2026

---

## Overview

This document describes the dataset versioning approach for the AQI Predictor project.

---

## Version Schema

### Version ID Format

```
{dataset_type}_{date}_{version}
```

### Examples

| Version ID | Description |
|---|---|
| `synthetic_test_data_20260808_v1.0` | Initial synthetic test data |
| `real_api_data_20260828_v1.0` | First real API data collection |
| `real_api_data_20260915_v1.1` | Updated real API data |

### Version Components

| Component | Description | Example |
|---|---|---|
| `dataset_type` | Type of dataset | `real_api_data`, `synthetic_test_data` |
| `date` | Collection date (YYYYMMDD) | `20260828` |
| `version` | Version number | `v1.0`, `v1.1` |

---

## Version Metadata

### Required Fields

```json
{
  "dataset_id": "real_api_data_20260828_v1.0",
  "dataset_type": "real_api_data",
  "source": "openweather_aqicn",
  "date_range_start": "2026-08-28",
  "date_range_end": "2026-09-28",
  "cities": ["Karachi", "Lahore", "Islamabad"],
  "resolution": "hourly",
  "total_observations": 2160,
  "features": 37,
  "quality_score": 0.95,
  "created_at": "2026-09-28T10:00:00Z",
  "approved_for_training": true,
  "approved_for_evaluation": true,
  "version": "real_api_data_20260828_v1.0"
}
```

### Optional Fields

```json
{
  "parent_version": "real_api_data_20260801_v1.0",
  "notes": "Extended collection period for better coverage"
}
```

---

## Version Storage

### Directory Structure

```
data/processed/versions/
├── real_api_data_20260828_v1.0.json
├── real_api_data_20260915_v1.1.json
└── synthetic_test_data_20260808_v1.0.json
```

### Dataset Files

```
data/processed/datasets/
├── real_api_data_20260828_v1.0/
│   ├── observations.csv
│   ├── features.csv
│   ├── targets.csv
│   └── metadata.json
└── real_api_data_20260915_v1.1/
    └── ...
```

---

## Approval Workflow

### Dataset Types

| Type | Training | Evaluation | Description |
|---|---|---|---|
| `synthetic_test_data` | ❌ No | ❌ No | Pipeline testing only |
| `real_api_data` | ✅ Yes | ✅ Yes | Production training |

### Approval Process

1. **Collect Data** → API collection pipeline
2. **Run Quality Gates** → Automated quality checks
3. **Review Quality Report** → Manual review
4. **Approve Version** → Set `approved_for_training=true`
5. **Use for Training** → Load approved version

### Approval Commands

```python
from src.data.dataset_versioning import DatasetVersionManager

manager = DatasetVersionManager()

# Approve for training
manager.approve_version(
    "real_api_data_20260828_v1.0",
    approved_for_training=True,
    approved_for_evaluation=True,
)

# Get latest approved
latest = manager.get_latest_approved()
```

---

## Version Lineage

### Tracking Changes

Each version can reference its parent version:

```json
{
  "dataset_id": "real_api_data_20260915_v1.1",
  "parent_version": "real_api_data_20260828_v1.0",
  "notes": "Added 15 more days of data"
}
```

### Lineage Visualization

```
synthetic_test_data_20260808_v1.0
    │
    └── (testing only, no lineage)
    
real_api_data_20260828_v1.0
    │
    └── real_api_data_20260915_v1.1
            │
            └── real_api_data_20261001_v1.2
```

---

## Quality Scores

### Quality Score Calculation

```python
quality_score = (
    completeness_score * 0.3 +
    freshness_score * 0.2 +
    accuracy_score * 0.3 +
    consistency_score * 0.2
)
```

### Quality Thresholds

| Score | Rating | Action |
|---|---|---|
| ≥ 0.90 | Excellent | Approve for training |
| 0.70 - 0.89 | Good | Approve with notes |
| 0.50 - 0.69 | Fair | Requires improvement |
| < 0.50 | Poor | Reject |

---

## Usage Examples

### Create New Version

```python
from src.data.dataset_versioning import DatasetVersionManager

manager = DatasetVersionManager()

version = manager.create_version(
    dataset_type="real_api_data",
    source="openweather_aqicn",
    date_range_start="2026-08-28",
    date_range_end="2026-09-28",
    cities=["Karachi", "Lahore", "Islamabad"],
    resolution="hourly",
    total_observations=2160,
    features=37,
    quality_score=0.95,
)
```

### List Versions

```python
# List all versions
all_versions = manager.list_versions()

# List only real data versions
real_versions = manager.list_versions(dataset_type="real_api_data")
```

### Get Latest Approved

```python
# Get latest version approved for training
latest = manager.get_latest_approved()

if latest:
    print(f"Using dataset: {latest.dataset_id}")
    print(f"Quality score: {latest.quality_score}")
```

---

## Best Practices

### Version Naming

1. Use consistent format: `{type}_{date}_{version}`
2. Increment version for minor changes
3. Use new date for major changes

### Quality Documentation

1. Always run quality gates before approval
2. Document any quality issues
3. Include quality score in metadata

### Lineage Tracking

1. Reference parent version when updating
2. Document changes between versions
3. Maintain clear audit trail
