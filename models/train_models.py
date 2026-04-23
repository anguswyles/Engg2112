"""
Train XGBoost, Random Forest, and LSTM models for soil moisture forecasting.
Run all three:  python models/train_models.py
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, recall_score, precision_score, f1_score
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras import layers

from prepare import load_all_stations, build_xgboost_dataset, build_lstm_dataset, get_feature_cols, station_temporal_split
from config import THRESHOLD, XGB_PARAM_GRID, RF_PARAM_GRID


def _print_metrics(name, y_true, y_pred):
    actual = (y_true < THRESHOLD).astype(int)
    pred   = (y_pred < THRESHOLD).astype(int)
    print(f"\n=== {name} ===")
    print(f"MAE:       {mean_absolute_error(y_true, y_pred):.4f} m³/m³")
    print(f"R²:        {r2_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(actual, pred):.3f}")
    print(f"Precision: {precision_score(actual, pred):.3f}")
    print(f"F1:        {f1_score(actual, pred):.3f}")


def train_xgboost(stations=None, data=None, horizon=None, with_weather=False):
    if stations is None:
        stations = load_all_stations(with_weather=with_weather)
    if data is None:
        data = build_xgboost_dataset(stations, horizon=horizon, with_weather=with_weather)
    feature_cols = get_feature_cols(with_weather=with_weather, horizon=horizon)

    X      = data[feature_cols].values
    y      = data['target'].values
    groups = data['station'].values
    train_mask, test_mask = station_temporal_split(data)

    cv_splits = list(GroupKFold(n_splits=5).split(X[train_mask], y[train_mask], groups[train_mask]))
    search = GridSearchCV(
        XGBRegressor(random_state=0, tree_method='hist'),
        XGB_PARAM_GRID,
        cv=cv_splits, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1,
    )
    search.fit(X[train_mask], y[train_mask])

    best   = search.best_estimator_
    y_pred = best.predict(X[test_mask])
    print(f"Best params: {search.best_params_}")
    _print_metrics('XGBoost', y[test_mask], y_pred)
    return best


def train_random_forest(stations=None, data=None, horizon=None, with_weather=False):
    if stations is None:
        stations = load_all_stations(with_weather=with_weather)
    if data is None:
        data = build_xgboost_dataset(stations, horizon=horizon, with_weather=with_weather)
    feature_cols = get_feature_cols(with_weather=with_weather, horizon=horizon)

    X      = data[feature_cols].values
    y      = data['target'].values
    groups = data['station'].values

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(100_000, len(X)), replace=False)
    idx.sort()
    X, y, groups = X[idx], y[idx], groups[idx]
    # Rebuild a lightweight frame so station_temporal_split can group by station
    import pandas as pd
    _df = pd.DataFrame({'station': groups})
    train_mask, test_mask = station_temporal_split(_df)

    cv_splits = list(GroupKFold(n_splits=5).split(X[train_mask], y[train_mask], groups[train_mask]))
    search = GridSearchCV(
        RandomForestRegressor(random_state=0, n_jobs=-1),
        RF_PARAM_GRID,
        cv=cv_splits, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1,
    )
    search.fit(X[train_mask], y[train_mask])

    best   = search.best_estimator_
    y_pred = best.predict(X[test_mask])
    print(f"Best params: {search.best_params_}")
    _print_metrics('Random Forest', y[test_mask], y_pred)
    return best


def train_lstm(stations=None):
    if stations is None:
        stations = load_all_stations()
    X, y = build_lstm_dataset(stations)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(100_000, len(X)), replace=False)
    idx.sort()
    X, y = X[idx], y[idx]

    X_mean, X_std = X.mean(), X.std()
    X_norm = (X - X_mean) / X_std
    split  = int(len(X) * 0.8)

    X_train = X_norm[:split].reshape(-1, X.shape[1], 1)
    X_test  = X_norm[split:].reshape(-1, X.shape[1], 1)
    y_train, y_test = y[:split], y[split:]

    model = keras.Sequential([
        keras.Input(shape=(X_train.shape[1], 1)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    model.fit(
        X_train, y_train,
        epochs=20, batch_size=512, validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                  restore_best_weights=True)],
        verbose=1,
    )

    y_pred = model.predict(X_test).flatten()
    _print_metrics('LSTM', y_test, y_pred)
    return model


if __name__ == '__main__':
    print('Loading data...')
    stations = load_all_stations(with_weather=True)
    data     = build_xgboost_dataset(stations, with_weather=True)

    train_xgboost(stations, data, with_weather=True)
    train_random_forest(stations, data, with_weather=True)
    train_lstm(stations)
