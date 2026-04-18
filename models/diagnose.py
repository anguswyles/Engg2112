import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from prepare import load_all_stations, build_xgboost_dataset

stations = load_all_stations()
data     = build_xgboost_dataset(stations)

all_feature_cols = [
    'sm_value', 'sm_lag_24h', 'sm_lag_48h', 'sm_lag_72h',
    'hour', 'month', 'dayofyear',
    'latitude', 'longitude', 'elevation_m', 'depth_m'
]

def evaluate(X_train, X_test, y_train, y_test, label):
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                         random_state=0, tree_method='hist', nthread=-1, verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"  {label:<45} MAE={mean_absolute_error(y_test, y_pred):.4f}  R²={r2_score(y_test, y_pred):.4f}")

split = int(len(data) * 0.8)
y = data['target'].values

print("=== NAIVE BASELINES ===")
y_train, y_test = y[:split], y[split:]
# predict the mean of training set
y_mean_pred = np.full_like(y_test, y_train.mean())
print(f"  {'Predict training mean':<45} MAE={mean_absolute_error(y_test, y_mean_pred):.4f}  R²={r2_score(y_test, y_mean_pred):.4f}")
# predict sm_lag_24h as the forecast (persistence model)
lag24 = data['sm_lag_24h'].values
y_persist = lag24[split:]
print(f"  {'Persistence (use sm at t-24 as forecast)':<45} MAE={mean_absolute_error(y_test, y_persist):.4f}  R²={r2_score(y_test, y_persist):.4f}")

print("\n=== FEATURE ABLATION (temporal split) ===")
configs = {
    'all features':                    all_feature_cols,
    'no sm_value':                     [c for c in all_feature_cols if c != 'sm_value'],
    'lag features only':               ['sm_lag_24h', 'sm_lag_48h', 'sm_lag_72h'],
    'station metadata only':           ['latitude', 'longitude', 'elevation_m', 'depth_m'],
    'temporal features only':          ['hour', 'month', 'dayofyear'],
    'no lag features':                 ['sm_value', 'hour', 'month', 'dayofyear', 'latitude', 'longitude', 'elevation_m', 'depth_m'],
}
for label, cols in configs.items():
    X = data[cols].values
    evaluate(X[:split], X[split:], y[:split], y[split:], label)

print("\n=== SPLIT STRATEGY (all features) ===")
X = data[all_feature_cols].values

# temporal split
evaluate(X[:split], X[split:], y[:split], y[split:], 'temporal split (80/20)')

# station holdout — hold out 20% of stations entirely
all_stations = data['station'].unique()
rng = np.random.default_rng(42)
holdout_stations = set(rng.choice(all_stations, size=max(1, len(all_stations)//5), replace=False))
train_mask = ~data['station'].isin(holdout_stations)
test_mask  =  data['station'].isin(holdout_stations)
evaluate(X[train_mask], X[test_mask], y[train_mask], y[test_mask],
         f'station holdout ({len(holdout_stations)} stations held out)')

# random split
rng2  = np.random.default_rng(0)
idx   = rng2.permutation(len(X))
n_tr  = int(len(X) * 0.8)
evaluate(X[idx[:n_tr]], X[idx[n_tr:]], y[idx[:n_tr]], y[idx[n_tr:]], 'random split (leaky)')

print("\n=== PREDICTION HORIZON COMPARISON ===")
print(f"  {'Horizon':<10} {'Persistence MAE':>16} {'Persistence R²':>15} {'XGBoost MAE':>12} {'XGBoost R²':>11}")
print(f"  {'-'*66}")

for horizon in [24, 48, 72, 120, 168]:
    frames = []
    for key, (df, meta) in stations.items():
        if len(df) < 200:
            continue
        feat = df.copy()
        feat['target'] = feat['sm_value'].shift(-horizon)
        for lag in (24, 48, 72):
            feat[f'sm_lag_{lag}h'] = feat['sm_value'].shift(lag)
        feat['hour']      = feat.index.hour
        feat['month']     = feat.index.month
        feat['dayofyear'] = feat.index.dayofyear
        for k, v in meta.items():
            feat[k] = v
        feat['station'] = key
        frames.append(feat.dropna())

    d      = pd.concat(frames).sort_index()
    X_h    = d[all_feature_cols].values
    y_h    = d['target'].values
    sp     = int(len(d) * 0.8)

    X_tr, X_te = X_h[:sp], X_h[sp:]
    y_tr, y_te = y_h[:sp], y_h[sp:]

    # persistence baseline
    y_persist = d['sm_lag_24h'].values[sp:]
    p_mae = mean_absolute_error(y_te, y_persist)
    p_r2  = r2_score(y_te, y_persist)

    # xgboost
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                         random_state=0, tree_method='hist', nthread=-1, verbosity=0)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    m_mae = mean_absolute_error(y_te, y_pred)
    m_r2  = r2_score(y_te, y_pred)

    print(f"  {f't+{horizon}h':<10} {p_mae:>16.4f} {p_r2:>15.4f} {m_mae:>12.4f} {m_r2:>11.4f}")
