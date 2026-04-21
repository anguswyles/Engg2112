"""
Diagnostic scripts for model validation.
Run all checks:  python models/diagnostics.py

Sections:
  check_leakage()       — compare with/without sm_value feature
  feature_ablation()    — ablation study + split strategy comparison
  horizon_comparison()  — MAE/R² across prediction horizons
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from prepare import load_all_stations, build_xgboost_dataset, get_feature_cols
from config import XGB_PARAMS


def _xgb():
    return XGBRegressor(**XGB_PARAMS)


def check_leakage():
    print('=== LEAKAGE CHECK ===')
    stations     = load_all_stations()
    data         = build_xgboost_dataset(stations)
    feature_cols = get_feature_cols()
    no_leak_cols = [c for c in feature_cols if c != 'sm_value']

    X      = data[feature_cols].values
    X_nl   = data[no_leak_cols].values
    y      = data['target'].values
    split  = int(len(X) * 0.8)

    y_train, y_test = y[:split], y[split:]

    m = _xgb()
    m.fit(X[:split], y_train)
    print('\n--- With sm_value (potential leak) ---')
    print(f"Train MAE: {mean_absolute_error(y_train, m.predict(X[:split])):.4f}  R²: {r2_score(y_train, m.predict(X[:split])):.4f}")
    print(f"Test  MAE: {mean_absolute_error(y_test,  m.predict(X[split:])):.4f}  R²: {r2_score(y_test,  m.predict(X[split:])):.4f}")

    m2 = _xgb()
    m2.fit(X_nl[:split], y_train)
    print('\n--- Without sm_value (lag features only) ---')
    print(f"Train MAE: {mean_absolute_error(y_train, m2.predict(X_nl[:split])):.4f}  R²: {r2_score(y_train, m2.predict(X_nl[:split])):.4f}")
    print(f"Test  MAE: {mean_absolute_error(y_test,  m2.predict(X_nl[split:])):.4f}  R²: {r2_score(y_test,  m2.predict(X_nl[split:])):.4f}")


def feature_ablation():
    stations     = load_all_stations()
    data         = build_xgboost_dataset(stations)
    feature_cols = get_feature_cols()
    y            = data['target'].values
    split        = int(len(data) * 0.8)

    def _eval(X, label):
        m = _xgb()
        m.fit(X[:split], y[:split])
        p = m.predict(X[split:])
        print(f"  {label:<45} MAE={mean_absolute_error(y[split:], p):.4f}  R²={r2_score(y[split:], p):.4f}")

    print('\n=== NAIVE BASELINES ===')
    y_mean = np.full(len(y) - split, y[:split].mean())
    y_pers = data['sm_lag_24h'].values[split:]
    print(f"  {'Predict training mean':<45} MAE={mean_absolute_error(y[split:], y_mean):.4f}  R²={r2_score(y[split:], y_mean):.4f}")
    print(f"  {'Persistence (sm at t-24 as forecast)':<45} MAE={mean_absolute_error(y[split:], y_pers):.4f}  R²={r2_score(y[split:], y_pers):.4f}")

    print('\n=== FEATURE ABLATION (temporal split) ===')
    configs = {
        'all features':           feature_cols,
        'no sm_value':            [c for c in feature_cols if c != 'sm_value'],
        'lag features only':      ['sm_lag_24h', 'sm_lag_48h', 'sm_lag_72h'],
        'station metadata only':  ['latitude', 'longitude', 'elevation_m', 'depth_m'],
        'temporal features only': ['hour', 'month', 'dayofyear'],
        'no lag features':        [c for c in feature_cols if 'lag' not in c],
    }
    for label, cols in configs.items():
        _eval(data[cols].values, label)

    print('\n=== SPLIT STRATEGY (all features) ===')
    X = data[feature_cols].values
    _eval(X, 'temporal split (80/20)')

    all_keys = data['station'].unique()
    rng      = np.random.default_rng(42)
    holdout  = set(rng.choice(all_keys, size=max(1, len(all_keys) // 5), replace=False))
    train_mask = ~data['station'].isin(holdout)
    test_mask  =  data['station'].isin(holdout)
    m2 = _xgb()
    m2.fit(X[train_mask], y[train_mask])
    p2 = m2.predict(X[test_mask])
    print(f"  {'station holdout':<45} MAE={mean_absolute_error(y[test_mask], p2):.4f}  R²={r2_score(y[test_mask], p2):.4f}  ({len(holdout)} held out)")

    rng2 = np.random.default_rng(0)
    idx  = rng2.permutation(len(X))
    n_tr = int(len(X) * 0.8)
    m3 = _xgb()
    m3.fit(X[idx[:n_tr]], y[idx[:n_tr]])
    p3 = m3.predict(X[idx[n_tr:]])
    print(f"  {'random split (leaky)':<45} MAE={mean_absolute_error(y[idx[n_tr:]], p3):.4f}  R²={r2_score(y[idx[n_tr:]], p3):.4f}")


def horizon_comparison():
    print('\n=== PREDICTION HORIZON COMPARISON ===')
    print(f"  {'Horizon':<10} {'Persistence MAE':>16} {'Persistence R²':>15} {'XGBoost MAE':>12} {'XGBoost R²':>11}")
    print(f"  {'-'*66}")

    stations     = load_all_stations()
    feature_cols = get_feature_cols()

    for horizon in [24, 48, 72, 120, 168]:
        data = build_xgboost_dataset(stations, horizon=horizon)
        X    = data[feature_cols].values
        y    = data['target'].values
        sp   = int(len(data) * 0.8)

        y_persist = data['sm_lag_24h'].values[sp:]
        p_mae = mean_absolute_error(y[sp:], y_persist)
        p_r2  = r2_score(y[sp:], y_persist)

        m = _xgb()
        m.fit(X[:sp], y[:sp])
        pred  = m.predict(X[sp:])
        m_mae = mean_absolute_error(y[sp:], pred)
        m_r2  = r2_score(y[sp:], pred)

        print(f"  {f't+{horizon}h':<10} {p_mae:>16.4f} {p_r2:>15.4f} {m_mae:>12.4f} {m_r2:>11.4f}")


if __name__ == '__main__':
    check_leakage()
    feature_ablation()
    horizon_comparison()
