#!/usr/bin/env python
"""LSTM training only."""
import sys, time, json, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.models.lstm_model import LSTMModel

TC = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
EX = ["timestamp","location_id","city_name","data_source","aqi_category","aqi_standard","aqi_method","aqi_method_version","aqi_source"]

train_f = pd.read_csv("data/processed/train_features.csv")
train_t = pd.read_csv("data/processed/train_targets.csv")
test_f = pd.read_csv("data/processed/test_features.csv")
test_t = pd.read_csv("data/processed/test_targets.csv")

fc = [c for c in train_f.columns if c not in EX and train_f[c].dtype in ["float64","int64","bool"]]
m1 = train_t[TC].notna().all(axis=1)
Xtr_full = train_f.loc[m1, fc].fillna(0).values
ytr_full = train_t.loc[m1, TC].values
m2 = test_t[TC].notna().all(axis=1)
Xte = test_f.loc[m2, fc].fillna(0).values
yte = test_t.loc[m2, TC].values

np.random.seed(42)
idx = np.random.choice(len(Xtr_full), size=int(len(Xtr_full)*0.3), replace=False)
Xtr = Xtr_full[idx]
ytr = ytr_full[idx]
print(f"Using {len(Xtr)} training rows")

def ev(yt, yp):
    r = {"per_horizon": [], "overall": {}}
    for i, h in enumerate(["24h", "48h", "72h"]):
        mae = round(mean_absolute_error(yt[:,i], yp[:,i]), 2)
        rmse = round(np.sqrt(mean_squared_error(yt[:,i], yp[:,i])), 2)
        r2 = round(r2_score(yt[:,i], yp[:,i]), 4)
        r["per_horizon"].append({"horizon": h, "mae": mae, "rmse": rmse, "r2": r2})
    r["overall"] = {
        "mae": round(mean_absolute_error(yt.flatten(), yp.flatten()), 2),
        "rmse": round(np.sqrt(mean_squared_error(yt.flatten(), yp.flatten())), 2),
        "r2": round(r2_score(yt.flatten(), yp.flatten()), 4),
    }
    return r

seq_len = 24
n_feat = Xtr.shape[1]
n_tgt = ytr.shape[1]

lstm = LSTMModel(sequence_length=seq_len, n_features=n_feat, n_targets=n_tgt, lstm_units=[64, 32], random_seed=42)
print(f"LSTM params: {lstm.model.count_params() if lstm.model else 'not built yet'}")

t0 = time.time()
hist = lstm.fit(Xtr, ytr, Xte, yte, epochs=30, batch_size=64, verbose=1)
rt = time.time() - t0
print(f"Training time: {rt:.1f}s, epochs: {hist.get('epochs_trained', 0)}")

t0 = time.time()
yp = lstm.predict(Xte)
it = (time.time() - t0) / max(len(yp), 1) * 1000

n_pred = len(yp)
yte_a = yte[seq_len-1:seq_len-1+n_pred]
met = ev(yte_a, yp)
print(f"\nLSTM: MAE={met['overall']['mae']} R2={met['overall']['r2']} infer={it:.4f}ms")
for h in met["per_horizon"]:
    print(f"  {h['horizon']}: MAE={h['mae']} RMSE={h['rmse']} R2={h['r2']}")

# Load existing results and append LSTM
with open("data/processed/model_results.json") as f:
    results = json.load(f)
results.append({"model": "LSTM", "metrics": met, "train_time": round(rt,1), "infer_ms": round(it,4), "epochs": hist.get("epochs_trained", 0), "params": {"sequence_length": seq_len, "lstm_units": [64,32], "epochs": 30, "batch_size": 64}})
with open("data/processed/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved.")
