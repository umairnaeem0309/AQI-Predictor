#!/usr/bin/env python
"""Train LSTM on full dataset - lighter config for dev machine."""
import sys, time, json, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

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

from src.models.lstm_model import LSTMModel

SEQ = 24
lstm = LSTMModel(
    sequence_length=SEQ,
    n_features=Xtr.shape[1],
    n_targets=ytr.shape[1],
    lstm_units=[32, 16],       # lighter than [64, 32]
    dropout_rate=0.2,
    learning_rate=0.002,       # slightly higher LR
    random_seed=SEED,
)

print("Training LSTM (15 epochs, batch_size=128)...")
t0 = time.time()
hist = lstm.fit(Xtr, ytr, Xte, yte, epochs=15, batch_size=128, verbose=0)
rt = time.time() - t0
print(f"  Trained in {rt:.1f}s ({hist.get('epochs_trained', '?')} epochs)")

t0 = time.time()
yp_raw = lstm.predict(Xte)
it = (time.time() - t0) / max(len(yp_raw), 1) * 1000
print(f"  Inference: {it:.4f}ms/sample")

yte_a = yte[SEQ - 1:SEQ - 1 + len(yp_raw)]

horizons = ["24h", "48h", "72h"]
results = {"per_horizon": [], "overall": {}}
for i, h in enumerate(horizons):
    results["per_horizon"].append({
        "horizon": h,
        "mae": round(mean_absolute_error(yte_a[:, i], yp_raw[:, i]), 2),
        "rmse": round(np.sqrt(mean_squared_error(yte_a[:, i], yp_raw[:, i])), 2),
        "r2": round(r2_score(yte_a[:, i], yp_raw[:, i]), 4),
    })
    print(f"  {h}: MAE={results['per_horizon'][-1]['mae']} RMSE={results['per_horizon'][-1]['rmse']} R²={results['per_horizon'][-1]['r2']}")
results["overall"] = {
    "mae": round(mean_absolute_error(yte_a.flatten(), yp_raw.flatten()), 2),
    "rmse": round(np.sqrt(mean_squared_error(yte_a.flatten(), yp_raw.flatten())), 2),
    "r2": round(r2_score(yte_a.flatten(), yp_raw.flatten()), 4),
}
print(f"  Overall: MAE={results['overall']['mae']} RMSE={results['overall']['rmse']} R²={results['overall']['r2']}")

out = [{"model": "LSTM", "metrics": results, "train_time": round(rt, 1),
        "infer_ms": round(it, 4), "params": {"sequence_length": SEQ, "lstm_units": [32, 16],
                                                "epochs": 15, "batch_size": 128},
        "epochs_trained": hist.get("epochs_trained", 0), "subset": "full"}]
with open("data/processed/lstm_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved to data/processed/lstm_results.json")
