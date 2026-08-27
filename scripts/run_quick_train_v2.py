#!/usr/bin/env python
"""Quick model training with subset for speed."""
import sys, time, json
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

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

# Use 30% subset for faster training
np.random.seed(42)
idx = np.random.choice(len(Xtr_full), size=int(len(Xtr_full)*0.3), replace=False)
Xtr = Xtr_full[idx]
ytr = ytr_full[idx]
print(f"Using {len(Xtr)} / {len(Xtr_full)} training rows (30% subset)")

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

results = []

# Ridge
sc = StandardScaler()
Xs = sc.fit_transform(Xtr)
Xtes = sc.transform(Xte)
t0 = time.time()
m = Ridge(alpha=1.0, random_state=42).fit(Xs, ytr)
rt = time.time() - t0
t0 = time.time()
yp = m.predict(Xtes)
it = (time.time() - t0) / len(Xte) * 1000
met = ev(yte, yp)
results.append({"model": "Ridge", "metrics": met, "train_time": round(rt,3), "infer_ms": round(it,4)})
print(f"RIDGE: MAE={met['overall']['mae']} R2={met['overall']['r2']} train={rt:.3f}s infer={it:.4f}ms")
for h in met["per_horizon"]:
    print(f"  {h['horizon']}: MAE={h['mae']} RMSE={h['rmse']} R2={h['r2']}")

# RF
t0 = time.time()
m = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)).fit(Xtr, ytr)
rt = time.time() - t0
t0 = time.time()
yp = m.predict(Xte)
it = (time.time() - t0) / len(Xte) * 1000
met = ev(yte, yp)
results.append({"model": "RandomForest", "metrics": met, "train_time": round(rt,1), "infer_ms": round(it,4)})
print(f"RF: MAE={met['overall']['mae']} R2={met['overall']['r2']} train={rt:.1f}s infer={it:.4f}ms")
for h in met["per_horizon"]:
    print(f"  {h['horizon']}: MAE={h['mae']} RMSE={h['rmse']} R2={h['r2']}")

# XGB
t0 = time.time()
m = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0, n_jobs=-1)).fit(Xtr, ytr)
rt = time.time() - t0
t0 = time.time()
yp = m.predict(Xte)
it = (time.time() - t0) / len(Xte) * 1000
met = ev(yte, yp)
results.append({"model": "XGBoost", "metrics": met, "train_time": round(rt,1), "infer_ms": round(it,4)})
print(f"XGB: MAE={met['overall']['mae']} R2={met['overall']['r2']} train={rt:.1f}s infer={it:.4f}ms")
for h in met["per_horizon"]:
    print(f"  {h['horizon']}: MAE={h['mae']} RMSE={h['rmse']} R2={h['r2']}")

# LSTM
try:
    from src.models.lstm_model import LSTMModel
    print("\nTraining LSTM...")
    seq_len = 24
    n_feat = Xtr.shape[1]
    n_tgt = ytr.shape[1]
    lstm = LSTMModel(sequence_length=seq_len, n_features=n_feat, n_targets=n_tgt, lstm_units=[64, 32], random_seed=42)
    t0 = time.time()
    hist = lstm.fit(Xtr, ytr, Xte, yte, epochs=30, batch_size=64, verbose=0)
    rt = time.time() - t0
    t0 = time.time()
    yp = lstm.predict(Xte)
    it = (time.time() - t0) / max(len(yp), 1) * 1000
    n_pred = len(yp)
    yte_a = yte[seq_len-1:seq_len-1+n_pred]
    met = ev(yte_a, yp)
    results.append({"model": "LSTM", "metrics": met, "train_time": round(rt,1), "infer_ms": round(it,4), "epochs": hist.get("epochs_trained", 0)})
    print(f"LSTM: MAE={met['overall']['mae']} R2={met['overall']['r2']} train={rt:.1f}s infer={it:.4f}ms epochs={hist.get('epochs_trained', 0)}")
    for h in met["per_horizon"]:
        print(f"  {h['horizon']}: MAE={h['mae']} RMSE={h['rmse']} R2={h['r2']}")
except Exception as e:
    print(f"LSTM failed: {e}")

with open("data/processed/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nAll results saved to data/processed/model_results.json")
