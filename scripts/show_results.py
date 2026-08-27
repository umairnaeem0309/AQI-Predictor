#!/usr/bin/env python
"""Display full model comparison results."""
import json

with open("data/processed/model_results_full.json") as f:
    results = json.load(f)

print("=" * 70)
print("  FULL DATASET MODEL COMPARISON (Test Set: 2026)")
print("=" * 70)

print()
print("  OVERALL RESULTS:")
header = f"  {'Model':<15} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Train':>10} {'Infer':>10}"
print(header)
print("  " + "-" * 60)
for r in results:
    o = r["metrics"]["overall"]
    print(f"  {r['model']:<15} {o['mae']:>8.2f} {o['rmse']:>8.2f} {o['r2']:>8.4f} {r['train_time']:>8.1f}s {r['infer_ms']:>8.3f}ms")

print()
print("  PER-HORIZON BREAKDOWN:")
header2 = f"  {'Model':<15} {'24h MAE':>8} {'48h MAE':>8} {'72h MAE':>8} {'24h R2':>8} {'48h R2':>8} {'72h R2':>8}"
print(header2)
print("  " + "-" * 70)
for r in results:
    ph = r["metrics"]["per_horizon"]
    print(f"  {r['model']:<15} {ph[0]['mae']:>8.2f} {ph[1]['mae']:>8.2f} {ph[2]['mae']:>8.2f} {ph[0]['r2']:>8.4f} {ph[1]['r2']:>8.4f} {ph[2]['r2']:>8.4f}")

print()
print("  COMPARISON VS RIDGE BASELINE:")
ridge_mae = next(r for r in results if r["model"] == "Ridge")["metrics"]["overall"]["mae"]
for r in results:
    mae = r["metrics"]["overall"]["mae"]
    diff = ((mae - ridge_mae) / ridge_mae) * 100
    better = "BETTER" if diff < 0 else "WORSE"
    print(f"  {r['model']:<15} MAE={mae:.2f}  ({abs(diff):.1f}% {better} than Ridge)")

print()
print("  RANKING BY MAE (lower is better):")
ranked = sorted(results, key=lambda r: r["metrics"]["overall"]["mae"])
for i, r in enumerate(ranked, 1):
    o = r["metrics"]["overall"]
    print(f"  {i}. {r['model']:<15} MAE={o['mae']:.2f}  R2={o['r2']:.4f}  Train={r['train_time']:.1f}s")

print()
print("  WHY XGBOOST WAS SELECTED:")
print("  1. Lowest MAE at EVERY horizon (24h, 48h, 72h)")
print("  2. Fastest non-linear training (18.2s vs RF 477.7s)")
print("  3. Real-time inference (0.030ms/sample)")
print("  4. Strong R2 (0.6065) - explains 60.65% of variance")
print()
print("  WHY NOT THE OTHERS:")
print("  - RandomForest: Nearly identical MAE (21.47 vs 21.32) but 26x slower")
print("  - Ridge: Within 3% of XGBoost - strong linear baseline")
print("  - LSTM: R2=0.3771 vs XGBoost 0.6065 - temporal patterns captured by features")
