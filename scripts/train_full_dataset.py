#!/usr/bin/env python
"""
Full Dataset Model Training — Runs all 4 models on the complete training split.

Train: 2022–2024 (63,504 rows)
Val:   2025 (26,280 rows)
Test:  2026 (16,920 rows)

Models: Ridge, Random Forest, XGBoost, LSTM
Metrics: MAE, RMSE, R² per horizon + overall
"""
import sys, time, json, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

TC = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
EX = ["timestamp","location_id","city_name","data_source","aqi_category",
      "aqi_standard","aqi_method","aqi_method_version","aqi_source"]
SEED = 42


def load_data():
    """Load train/val/test with feature columns."""
    train_f = pd.read_csv("data/processed/train_features.csv")
    train_t = pd.read_csv("data/processed/train_targets.csv")
    val_f = pd.read_csv("data/processed/val_features.csv")
    val_t = pd.read_csv("data/processed/val_targets.csv")
    test_f = pd.read_csv("data/processed/test_features.csv")
    test_t = pd.read_csv("data/processed/test_targets.csv")

    fc = [c for c in train_f.columns if c not in EX
          and train_f[c].dtype in ["float64", "int64", "bool"]]

    def prep(feat, tgt):
        mask = tgt[TC].notna().all(axis=1)
        X = feat.loc[mask, fc].fillna(0).values
        y = tgt.loc[mask, TC].values
        return X, y

    Xtr, ytr = prep(train_f, train_t)
    Xv, yv = prep(val_f, val_t)
    Xte, yte = prep(test_f, test_t)
    return Xtr, ytr, Xv, yv, Xte, yte, fc


def ev(yt, yp):
    """Compute per-horizon and overall metrics."""
    r = {"per_horizon": [], "overall": {}}
    for i, h in enumerate(["24h", "48h", "72h"]):
        r["per_horizon"].append({
            "horizon": h,
            "mae": round(mean_absolute_error(yt[:, i], yp[:, i]), 2),
            "rmse": round(np.sqrt(mean_squared_error(yt[:, i], yp[:, i])), 2),
            "r2": round(r2_score(yt[:, i], yp[:, i]), 4),
        })
    r["overall"] = {
        "mae": round(mean_absolute_error(yt.flatten(), yp.flatten()), 2),
        "rmse": round(np.sqrt(mean_squared_error(yt.flatten(), yp.flatten())), 2),
        "r2": round(r2_score(yt.flatten(), yp.flatten()), 4),
    }
    return r


def main():
    print("=" * 70)
    print("FULL DATASET MODEL TRAINING")
    print("=" * 70)

    Xtr, ytr, Xv, yv, Xte, yte, fc = load_data()
    print(f"Train: {Xtr.shape}, Val: {Xv.shape}, Test: {Xte.shape}")
    print(f"Features: {len(fc)}")

    # Scale for linear models
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    results = []

    # --- 1. Ridge ---
    print("\n--- Ridge Regression ---")
    t0 = time.time()
    m = Ridge(alpha=1.0, random_state=SEED).fit(Xtr_s, ytr)
    rt = time.time() - t0
    t0 = time.time()
    yp = m.predict(Xte_s)
    it = (time.time() - t0) / len(Xte) * 1000
    met = ev(yte, yp)
    results.append({"model": "Ridge", "metrics": met, "train_time": round(rt, 3),
                     "infer_ms": round(it, 4), "params": {"alpha": 1.0},
                     "subset": "full"})
    print(f"  MAE={met['overall']['mae']} R²={met['overall']['r2']} train={rt:.3f}s infer={it:.4f}ms")
    for h in met["per_horizon"]:
        print(f"    {h['horizon']}: MAE={h['mae']} RMSE={h['rmse']} R²={h['r2']}")

    # --- 2. Random Forest ---
    print("\n--- Random Forest ---")
    t0 = time.time()
    m = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=100, max_depth=20, random_state=SEED, n_jobs=-1
    )).fit(Xtr, ytr)
    rt = time.time() - t0
    t0 = time.time()
    yp = m.predict(Xte)
    it = (time.time() - t0) / len(Xte) * 1000
    met = ev(yte, yp)
    results.append({"model": "RandomForest", "metrics": met, "train_time": round(rt, 1),
                     "infer_ms": round(it, 4),
                     "params": {"n_estimators": 100, "max_depth": 20},
                     "subset": "full"})
    print(f"  MAE={met['overall']['mae']} R²={met['overall']['r2']} train={rt:.1f}s infer={it:.4f}ms")
    for h in met["per_horizon"]:
        print(f"    {h['horizon']}: MAE={h['mae']} RMSE={h['rmse']} R²={h['r2']}")

    # --- 3. XGBoost ---
    print("\n--- XGBoost ---")
    t0 = time.time()
    m = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=SEED, verbosity=0, n_jobs=-1
    )).fit(Xtr, ytr)
    rt = time.time() - t0
    t0 = time.time()
    yp = m.predict(Xte)
    it = (time.time() - t0) / len(Xte) * 1000
    met = ev(yte, yp)
    results.append({"model": "XGBoost", "metrics": met, "train_time": round(rt, 1),
                     "infer_ms": round(it, 4),
                     "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
                     "subset": "full"})
    print(f"  MAE={met['overall']['mae']} R²={met['overall']['r2']} train={rt:.1f}s infer={it:.4f}ms")
    for h in met["per_horizon"]:
        print(f"    {h['horizon']}: MAE={h['mae']} RMSE={h['rmse']} R²={h['r2']}")

    # --- 4. LSTM ---
    print("\n--- LSTM ---")
    try:
        from src.models.lstm_model import LSTMModel

        SEQ = 24
        lstm = LSTMModel(
            sequence_length=SEQ, n_features=Xtr.shape[1], n_targets=ytr.shape[1],
            lstm_units=[64, 32], dropout_rate=0.2, learning_rate=0.001, random_seed=SEED,
        )
        t0 = time.time()
        hist = lstm.fit(Xtr, ytr, Xte, yte, epochs=30, batch_size=64, verbose=0)
        rt = time.time() - t0

        t0 = time.time()
        yp_raw = lstm.predict(Xte)
        it = (time.time() - t0) / max(len(yp_raw), 1) * 1000

        yte_a = yte[SEQ - 1:SEQ - 1 + len(yp_raw)]
        met = ev(yte_a, yp_raw)
        results.append({"model": "LSTM", "metrics": met, "train_time": round(rt, 1),
                         "infer_ms": round(it, 4),
                         "params": {"sequence_length": SEQ, "lstm_units": [64, 32],
                                    "epochs": 30, "batch_size": 64},
                         "epochs_trained": hist.get("epochs_trained", 0),
                         "subset": "full"})
        print(f"  MAE={met['overall']['mae']} R²={met['overall']['r2']} train={rt:.1f}s infer={it:.4f}ms")
        for h in met["per_horizon"]:
            print(f"    {h['horizon']}: MAE={h['mae']} RMSE={h['rmse']} R²={h['r2']}")
    except Exception as e:
        print(f"  LSTM failed: {e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("FINAL RESULTS (FULL DATASET)")
    print("=" * 70)
    for r in results:
        o = r["metrics"]["overall"]
        print(f"  {r['model']:15s}  MAE={o['mae']:6.2f}  RMSE={o['rmse']:6.2f}  R²={o['r2']:.4f}  "
              f"train={r['train_time']:.1f}s  infer={r['infer_ms']:.3f}ms")

    # Save
    with open("data/processed/model_results_full.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to data/processed/model_results_full.json")


if __name__ == "__main__":
    main()
