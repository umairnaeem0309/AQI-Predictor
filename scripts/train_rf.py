#!/usr/bin/env python
"""Train RandomForest on full dataset - optimized for speed."""
import sys, time, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TC = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
EX = ["timestamp","location_id","city_name","data_source","aqi_category",
      "aqi_standard","aqi_method","aqi_method_version","aqi_source"]
SEED = 42

print("Loading data...")
train_f = pd.read_csv("data/processed/train_features.csv")
train_t = pd.read_csv("data/processed/train_targets.csv")
test_f = pd.read_csv("data/processed/test_features.csv")
test_t = pd.read_csv("data/processed/test_targets.csv")
print(f"Train: {train_f.shape}, Test: {test_f.shape}")

fc = [c for c in train_f.columns if c not in EX and train_f[c].dtype in ["float64","int64","bool"]]

def prep(f, t):
    m = t[TC].notna().all(axis=1)
    return f.loc[m, fc].fillna(0).values, t.loc[m, TC].values

Xtr, ytr = prep(train_f, train_t)
Xte, yte = prep(test_f, test_t)
print(f"Xtr: {Xtr.shape}, Xte: {Xte.shape}")

print("Training RandomForest (n_estimators=100, max_depth=20)...")
t0 = time.time()
m = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=100, max_depth=20, random_state=SEED, n_jobs=-1
)).fit(Xtr, ytr)
rt = time.time() - t0
print(f"  Trained in {rt:.1f}s")

t0 = time.time()
yp = m.predict(Xte)
it = (time.time() - t0) / len(Xte) * 1000
print(f"  Inference: {it:.4f}ms/sample")

horizons = ["24h", "48h", "72h"]
results = {"per_horizon": [], "overall": {}}
for i, h in enumerate(horizons):
    results["per_horizon"].append({
        "horizon": h,
        "mae": round(mean_absolute_error(yte[:, i], yp[:, i]), 2),
        "rmse": round(np.sqrt(mean_squared_error(yte[:, i], yp[:, i])), 2),
        "r2": round(r2_score(yte[:, i], yp[:, i]), 4),
    })
    print(f"  {h}: MAE={results['per_horizon'][-1]['mae']} RMSE={results['per_horizon'][-1]['rmse']} R²={results['per_horizon'][-1]['r2']}")
results["overall"] = {
    "mae": round(mean_absolute_error(yte.flatten(), yp.flatten()), 2),
    "rmse": round(np.sqrt(mean_squared_error(yte.flatten(), yp.flatten())), 2),
    "r2": round(r2_score(yte.flatten(), yp.flatten()), 4),
}
print(f"  Overall: MAE={results['overall']['mae']} RMSE={results['overall']['rmse']} R²={results['overall']['r2']}")

out = [{"model": "RandomForest", "metrics": results, "train_time": round(rt, 1),
        "infer_ms": round(it, 4), "params": {"n_estimators": 100, "max_depth": 20},
        "subset": "full"}]
with open("data/processed/rf_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved to data/processed/rf_results.json")
