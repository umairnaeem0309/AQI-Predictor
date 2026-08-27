#!/usr/bin/env python
"""Train Ridge, RF, XGBoost on full dataset."""
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
SEED = 42

train_f = pd.read_csv("data/processed/train_features.csv")
train_t = pd.read_csv("data/processed/train_targets.csv")
val_f = pd.read_csv("data/processed/val_features.csv")
val_t = pd.read_csv("data/processed/val_targets.csv")
test_f = pd.read_csv("data/processed/test_features.csv")
test_t = pd.read_csv("data/processed/test_targets.csv")

fc = [c for c in train_f.columns if c not in EX and train_f[c].dtype in ["float64","int64","bool"]]
def prep(f, t):
    m = t[TC].notna().all(axis=1)
    return f.loc[m, fc].fillna(0).values, t.loc[m, TC].values

Xtr, ytr = prep(train_f, train_t)
Xv, yv = prep(val_f, val_t)
Xte, yte = prep(test_f, test_t)
print(f"Train: {Xtr.shape}, Val: {Xv.shape}, Test: {Xte.shape}")

def ev(yt, yp):
    r = {"per_horizon": [], "overall": {}}
    for i, h in enumerate(["24h", "48h", "72h"]):
        r["per_horizon"].append({"horizon": h, "mae": round(mean_absolute_error(yt[:,i],yp[:,i]),2),
                                 "rmse": round(np.sqrt(mean_squared_error(yt[:,i],yp[:,i])),2),
                                 "r2": round(r2_score(yt[:,i],yp[:,i]),4)})
    r["overall"] = {"mae": round(mean_absolute_error(yt.flatten(),yp.flatten()),2),
                     "rmse": round(np.sqrt(mean_squared_error(yt.flatten(),yp.flatten())),2),
                     "r2": round(r2_score(yt.flatten(),yp.flatten()),4)}
    return r

sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)
results = []

# Ridge
print("\n--- Ridge ---")
t0 = time.time(); m = Ridge(alpha=1.0, random_state=SEED).fit(Xtr_s, ytr); rt = time.time()-t0
t0 = time.time(); yp = m.predict(Xte_s); it = (time.time()-t0)/len(Xte)*1000
met = ev(yte, yp)
results.append({"model":"Ridge","metrics":met,"train_time":round(rt,3),"infer_ms":round(it,4),"params":{"alpha":1.0},"subset":"full"})
print(f"  MAE={met['overall']['mae']} R2={met['overall']['r2']} train={rt:.3f}s infer={it:.4f}ms")

# RF
print("\n--- RandomForest ---")
t0 = time.time(); m = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,max_depth=20,random_state=SEED,n_jobs=-1)).fit(Xtr,ytr); rt = time.time()-t0
t0 = time.time(); yp = m.predict(Xte); it = (time.time()-t0)/len(Xte)*1000
met = ev(yte, yp)
results.append({"model":"RandomForest","metrics":met,"train_time":round(rt,1),"infer_ms":round(it,4),"params":{"n_estimators":100,"max_depth":20},"subset":"full"})
print(f"  MAE={met['overall']['mae']} R2={met['overall']['r2']} train={rt:.1f}s infer={it:.4f}ms")

# XGB
print("\n--- XGBoost ---")
t0 = time.time(); m = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=SEED,verbosity=0,n_jobs=-1)).fit(Xtr,ytr); rt = time.time()-t0
t0 = time.time(); yp = m.predict(Xte); it = (time.time()-t0)/len(Xte)*1000
met = ev(yte, yp)
results.append({"model":"XGBoost","metrics":met,"train_time":round(rt,1),"infer_ms":round(it,4),"params":{"n_estimators":200,"max_depth":6,"learning_rate":0.1},"subset":"full"})
print(f"  MAE={met['overall']['mae']} R2={met['overall']['r2']} train={rt:.1f}s infer={it:.4f}ms")

with open("data/processed/model_results_full.json","w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to data/processed/model_results_full.json")
