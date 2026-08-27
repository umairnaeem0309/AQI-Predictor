#!/usr/bin/env python
"""Merge individual model results into model_results_full.json."""
import json

files = [
    "data/processed/ridge_results.json",
    "data/processed/rf_results.json",
    "data/processed/xgb_results.json",
    "data/processed/lstm_results.json",
]

all_results = []
for f in files:
    with open(f) as fh:
        data = json.load(fh)
        all_results.extend(data)

with open("data/processed/model_results_full.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("Merged results:")
for r in all_results:
    o = r["metrics"]["overall"]
    print(f"  {r['model']:15s}  MAE={o['mae']:6.2f}  RMSE={o['rmse']:6.2f}  R²={o['r2']:.4f}  train={r['train_time']:.1f}s  infer={r['infer_ms']:.3f}ms")

print(f"\nSaved to data/processed/model_results_full.json ({len(all_results)} models)")
