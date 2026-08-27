#!/usr/bin/env python
import json, pandas as pd
with open("models/production/model_metadata.json") as f:
    m = json.load(f)
model_fc = set(m["feature_columns"])
df = pd.read_csv("data/processed/test_features.csv")
exclude = {"timestamp","location_id","city_name","data_source","aqi_category","aqi_standard","aqi_method","aqi_method_version","aqi_source"}
all_numeric = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64","int64","bool"]]
extra = set(all_numeric) - model_fc
missing = model_fc - set(all_numeric)
print(f"Model features: {len(model_fc)}")
print(f"CSV numeric features: {len(all_numeric)}")
print(f"Extra in CSV: {extra}")
print(f"Missing from CSV: {missing}")
