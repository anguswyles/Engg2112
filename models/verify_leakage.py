import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from prepare import load_all_stations, build_xgboost_dataset

stations = load_all_stations()
data     = build_xgboost_dataset(stations)

feature_cols = [
    'sm_value', 'sm_lag_24h', 'sm_lag_48h', 'sm_lag_72h',
    'hour', 'month', 'dayofyear',
    'latitude', 'longitude', 'elevation_m', 'depth_m'
]
feature_cols_no_leak = [
    'sm_lag_24h', 'sm_lag_48h', 'sm_lag_72h',
    'hour', 'month', 'dayofyear',
    'latitude', 'longitude', 'elevation_m', 'depth_m'
]

X      = data[feature_cols].values
X_nl   = data[feature_cols_no_leak].values
y      = data['target'].values
split  = int(len(X) * 0.8)

X_train,    X_test    = X[:split],    X[split:]
X_train_nl, X_test_nl = X_nl[:split], X_nl[split:]
y_train,    y_test    = y[:split],    y[split:]

model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                     random_state=0, tree_method='hist', nthread=-1)

print("=== WITH sm_value (potential leak) ===")
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.4f}  R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test  MAE: {mean_absolute_error(y_test,  y_pred_test):.4f}  R²: {r2_score(y_test,  y_pred_test):.4f}")

print("\n=== WITHOUT sm_value (lag features only) ===")
model.fit(X_train_nl, y_train)
y_pred_train_nl = model.predict(X_train_nl)
y_pred_test_nl  = model.predict(X_test_nl)
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train_nl):.4f}  R²: {r2_score(y_train, y_pred_train_nl):.4f}")
print(f"Test  MAE: {mean_absolute_error(y_test,  y_pred_test_nl):.4f}  R²: {r2_score(y_test,  y_pred_test_nl):.4f}")
