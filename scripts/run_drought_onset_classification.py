"""
Train four models to predict drought onset directly as a classification task.

This script does not replace the regression experiments. It reuses the existing
feature builders and temporal splits, then filters evaluation/training to samples
that are currently above the drought threshold and labels whether they transition
below the threshold by the forecast horizon.

The training target is intentionally strict (drought at the exact horizon). The
evaluation also reports tolerant success metrics where a predicted warning counts
if drought occurs within --success-tolerance-hours after the target time, plus
event-level metrics that collapse repeated hourly rows into warning/onset events.

Usage:
    python scripts/run_drought_onset_classification.py
    python scripts/run_drought_onset_classification.py "test 12 - drought onset classification"
    python scripts/run_drought_onset_classification.py --horizon 120 --no-weather
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBClassifier

from config import THRESHOLD, XGB_PARAMS
from prepare import (
    build_lstm_dataset,
    build_tft_dataset,
    build_xgboost_dataset,
    get_feature_cols,
    load_all_stations,
    station_temporal_split,
    temporal_split_by_station,
)


IMAGES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
COLORS = {
    "Persistence": "black",
    "Linear Trend": "gray",
    "Random Forest": "steelblue",
    "XGBoost": "darkorange",
    "LSTM": "green",
    "TFT": "purple",
}


def next_test_folder_name() -> str:
    dirs = [
        d
        for d in os.listdir(IMAGES_ROOT)
        if os.path.isdir(os.path.join(IMAGES_ROOT, d)) and d.startswith("test ")
    ]
    nums: list[int] = []
    for d in dirs:
        try:
            nums.append(int(d.split(" ", 2)[1]))
        except (IndexError, ValueError):
            pass
    return f"test {max(nums) + 1 if nums else 1}"


def cap_rows(rng: np.random.Generator, *arrays, max_n: int | None = None):
    arrays = tuple(arrays)
    if max_n is None or len(arrays[0]) <= max_n:
        return arrays
    idx = rng.choice(len(arrays[0]), size=max_n, replace=False)
    idx.sort()
    return tuple(a[idx] for a in arrays)


def class_weight_dict(y: np.ndarray) -> dict[int, float]:
    y = np.asarray(y).astype(int)
    neg = max(int((y == 0).sum()), 1)
    pos = max(int((y == 1).sum()), 1)
    total = neg + pos
    return {0: total / (2 * neg), 1: total / (2 * pos)}


def scale_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    neg = max(int((y == 0).sum()), 1)
    pos = max(int((y == 1).sum()), 1)
    return neg / pos


def onset_filter(current_sm: np.ndarray, future_sm: np.ndarray, threshold: float):
    eligible = current_sm >= threshold
    y_onset = (future_sm < threshold).astype(int)
    return eligible, y_onset


def tolerant_onset_labels(
    stations: dict,
    station_keys: np.ndarray,
    target_times: np.ndarray,
    threshold: float,
    tolerance_hours: int,
) -> np.ndarray:
    """True if drought occurs at target time or within the success tolerance after it."""
    end_delta = pd.Timedelta(hours=int(tolerance_hours))
    series_cache = {
        key: pair[0]["sm_value"].ffill().bfill().dropna()
        for key, pair in stations.items()
        if "sm_value" in pair[0].columns
    }
    labels = np.zeros(len(station_keys), dtype=int)
    for idx, (station, target_time) in enumerate(zip(station_keys, target_times)):
        sm = series_cache.get(station)
        if sm is None:
            continue
        start = pd.Timestamp(target_time)
        window = sm.loc[start : start + end_delta]
        labels[idx] = int((window < threshold).any()) if len(window) else 0
    return labels


def training_onset_labels(
    stations: dict,
    station_keys: np.ndarray,
    target_times: np.ndarray,
    exact_labels: np.ndarray,
    threshold: float,
    train_tolerance_hours: int,
) -> np.ndarray:
    """Return labels for model training, optionally broadened to the warning window."""
    if train_tolerance_hours <= 0:
        return exact_labels
    return tolerant_onset_labels(stations, station_keys, target_times, threshold, train_tolerance_hours)


def evaluate_probs(
    model_name: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    label: str,
) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "Model": model_name,
        "Threshold label": label,
        "Probability threshold": round(float(threshold), 4),
        "Test samples": int(len(y_true)),
        "Onsets": int(y_true.sum()),
        "Caught": int(tp),
        "Missed": int(fn),
        "False alarms": int(fp),
        "Recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, pred, zero_division=0), 4),
        "False alarm rate": round(false_alarm_rate, 4),
        "PR AUC": round(average_precision_score(y_true, prob), 4),
        "ROC AUC": round(roc_auc_score(y_true, prob), 4) if len(np.unique(y_true)) == 2 else np.nan,
    }


def _event_intervals(stations: np.ndarray, times: np.ndarray, flags: np.ndarray) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    df = pd.DataFrame(
        {
            "station": stations,
            "time": pd.to_datetime(times),
            "flag": np.asarray(flags).astype(bool),
        }
    )
    df = df[df["flag"]].sort_values(["station", "time"])
    intervals: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for station, group in df.groupby("station", sort=False):
        start = prev = None
        for time in group["time"]:
            if start is None:
                start = prev = time
                continue
            if time - prev > pd.Timedelta(hours=1.5):
                intervals.append((station, start, prev))
                start = time
            prev = time
        if start is not None and prev is not None:
            intervals.append((station, start, prev))
    return intervals


def _overlaps(left: tuple[str, pd.Timestamp, pd.Timestamp], right: tuple[str, pd.Timestamp, pd.Timestamp]) -> bool:
    return left[0] == right[0] and left[1] <= right[2] and right[1] <= left[2]


def evaluate_event_probs(
    model_name: str,
    y_true_tolerant: np.ndarray,
    prob: np.ndarray,
    station_keys: np.ndarray,
    target_times: np.ndarray,
    threshold: float,
    label: str,
) -> dict:
    pred = (prob >= threshold).astype(int)
    actual_events = _event_intervals(station_keys, target_times, y_true_tolerant)
    pred_events = _event_intervals(station_keys, target_times, pred)
    actual_hits = sum(any(_overlaps(actual, predicted) for predicted in pred_events) for actual in actual_events)
    pred_hits = sum(any(_overlaps(predicted, actual) for actual in actual_events) for predicted in pred_events)
    missed = len(actual_events) - actual_hits
    false_alarm_events = len(pred_events) - pred_hits
    event_recall = actual_hits / len(actual_events) if actual_events else 0.0
    event_precision = pred_hits / len(pred_events) if pred_events else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    return {
        "Model": model_name,
        "Threshold label": label,
        "Probability threshold": round(float(threshold), 4),
        "Actual onset events": int(len(actual_events)),
        "Predicted warning events": int(len(pred_events)),
        "Caught events": int(actual_hits),
        "Missed events": int(missed),
        "False alarm events": int(false_alarm_events),
        "Event recall": round(event_recall, 4),
        "Event precision": round(event_precision, 4),
        "Event F1": round(event_f1, 4),
    }


def threshold_sweeps(y_true: np.ndarray, prob: np.ndarray, max_false_alarm_rate: float):
    thresholds = np.linspace(0.01, 0.99, 99)
    records = []
    best_f1 = (0.5, -1.0)
    best_recall_at_fa = (0.5, -1.0, 1.0)
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        far = fp / (fp + tn) if (fp + tn) else 0.0
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        records.append(
            {
                "Probability threshold": round(float(threshold), 4),
                "Recall": round(float(rec), 4),
                "Precision": round(float(prec), 4),
                "F1": round(float(f1), 4),
                "False alarm rate": round(float(far), 4),
            }
        )
        if f1 > best_f1[1]:
            best_f1 = (threshold, f1)
        if far <= max_false_alarm_rate and rec > best_recall_at_fa[1]:
            best_recall_at_fa = (threshold, rec, far)
    return pd.DataFrame(records), float(best_f1[0]), float(best_recall_at_fa[0])


def build_lstm_classifier(seq_len: int, n_features: int, future_steps: int | None, n_future_wx: int | None):
    seq_in = keras.Input(shape=(seq_len, n_features), name="sequence")
    x = layers.LSTM(64, return_sequences=True)(seq_in)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    inputs = [seq_in]
    if future_steps is not None and n_future_wx:
        fut_in = keras.Input(shape=(future_steps, n_future_wx), name="future_weather")
        fw = layers.Flatten()(fut_in)
        fw = layers.Dense(32, activation="relu")(fw)
        x = layers.Concatenate()([x, fw])
        inputs.append(fut_in)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out)


def build_tft_classifier(
    seq_len: int,
    n_features: int,
    n_static: int,
    future_steps: int | None,
    n_future_wx: int | None,
):
    seq_in = keras.Input(shape=(seq_len, n_features), name="sequence")
    sta_in = keras.Input(shape=(n_static,), name="static")
    s = layers.Dense(16, activation="elu")(sta_in)
    s = layers.Dense(16)(s)
    s_exp = layers.RepeatVector(seq_len)(s)
    x = layers.Concatenate(axis=-1)([seq_in, s_exp])
    x = layers.LSTM(64, return_sequences=True)(x)
    a = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    g = layers.Dense(64, activation="sigmoid")(a)
    x = layers.Add()([layers.Multiply()([g, a]), x])
    x = layers.LayerNormalization()(x)
    x = layers.Lambda(lambda t: t[:, -1, :])(x)
    inputs = [seq_in]
    if future_steps is not None and n_future_wx:
        fut_in = keras.Input(shape=(future_steps, n_future_wx), name="future_weather")
        fw = layers.Flatten()(fut_in)
        fw = layers.Dense(16, activation="elu")(fw)
        fw = layers.LayerNormalization()(fw)
        x = layers.Concatenate()([x, fw])
        inputs.append(fut_in)
    inputs.append(sta_in)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out)


def train_tabular_classifiers(args, stations: dict) -> dict[str, dict]:
    data = build_xgboost_dataset(stations, horizon=args.horizon, with_weather=args.weather)
    feature_cols = get_feature_cols(with_weather=args.weather, horizon=args.horizon)
    X = data[feature_cols].values
    future_sm = data["target"].values
    current_sm = data["sm_value"].values
    tr, te = station_temporal_split(data)
    eligible, y_onset = onset_filter(current_sm, future_sm, args.threshold)
    train_mask = tr & eligible
    test_mask = te & eligible
    X_train, y_train = X[train_mask], y_onset[train_mask]
    X_test, y_test = X[test_mask], y_onset[test_mask]
    train_stations = data["station"].values[train_mask]
    train_target_times = pd.to_datetime(data.index.values[train_mask]) + pd.Timedelta(hours=args.horizon)
    test_stations = data["station"].values[test_mask]
    test_target_times = pd.to_datetime(data.index.values[test_mask]) + pd.Timedelta(hours=args.horizon)
    y_train = training_onset_labels(
        stations,
        train_stations,
        train_target_times,
        y_train,
        args.threshold,
        args.train_tolerance_hours,
    )
    y_tolerant = tolerant_onset_labels(
        stations,
        test_stations,
        test_target_times,
        args.threshold,
        args.success_tolerance_hours,
    )
    rng = np.random.default_rng(args.seed)
    X_train, y_train = cap_rows(rng, X_train, y_train, max_n=args.max_tabular_train)

    print(
        f"Tabular onset set: train={len(y_train)} positives={int(y_train.sum())} "
        f"test={len(y_test)} exact_positives={int(y_test.sum())}"
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    print("Training Random Forest classifier...")
    rf.fit(X_train, y_train)
    prob_rf = rf.predict_proba(X_test)[:, 1]

    xgb_params = {
        key: value
        for key, value in XGB_PARAMS.items()
        if key not in {"nthread", "random_state", "verbosity"}
    }
    xgb = XGBClassifier(
        **xgb_params,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight(y_train),
        random_state=args.seed,
        verbosity=0,
    )
    print("Training XGBoost classifier...")
    xgb.fit(X_train, y_train)
    prob_xgb = xgb.predict_proba(X_test)[:, 1]
    del X, X_train, rf, xgb
    gc.collect()
    return {
        "Random Forest": {
            "y_true": y_test.copy(),
            "y_tolerant": y_tolerant,
            "prob": prob_rf,
            "station": test_stations,
            "target_time": test_target_times.to_numpy(),
        },
        "XGBoost": {
            "y_true": y_test.copy(),
            "y_tolerant": y_tolerant,
            "prob": prob_xgb,
            "station": test_stations,
            "target_time": test_target_times.to_numpy(),
        },
    }


def build_persistence_baseline(args, stations: dict) -> dict[str, dict]:
    """No-change baseline: because samples are currently wet, it predicts no onset."""
    data = build_xgboost_dataset(stations, horizon=args.horizon, with_weather=args.weather)
    future_sm = data["target"].values
    current_sm = data["sm_value"].values
    _, te = station_temporal_split(data)
    eligible, y_onset = onset_filter(current_sm, future_sm, args.threshold)
    test_mask = te & eligible
    y_test = y_onset[test_mask]
    test_stations = data["station"].values[test_mask]
    test_target_times = pd.to_datetime(data.index.values[test_mask]) + pd.Timedelta(hours=args.horizon)
    y_tolerant = tolerant_onset_labels(
        stations,
        test_stations,
        test_target_times,
        args.threshold,
        args.success_tolerance_hours,
    )
    prob = np.zeros(len(y_test), dtype=float)
    print(
        f"Persistence onset baseline: test={len(y_test)} positives={int(y_test.sum())} "
        f"tolerant_positives={int(y_tolerant.sum())}"
    )
    return {
        "Persistence": {
            "y_true": y_test.copy(),
            "y_tolerant": y_tolerant,
            "prob": prob,
            "station": test_stations,
            "target_time": test_target_times.to_numpy(),
        }
    }


def _trend_probability(current_sm: np.ndarray, lag_sm: np.ndarray, horizon: int, lag_hours: int, threshold: float):
    slope_per_hour = (current_sm - lag_sm) / float(lag_hours)
    projected = current_sm + slope_per_hour * float(horizon)
    # Convert projected crossing distance into a smooth risk score for threshold sweeps.
    distance = threshold - projected
    return 1.0 / (1.0 + np.exp(-distance / 0.01))


def build_linear_trend_baseline(args, stations: dict) -> dict[str, dict]:
    """Naive physical baseline: extrapolate recent SM trend to the forecast horizon."""
    data = build_xgboost_dataset(stations, horizon=args.horizon, with_weather=args.weather)
    lag_col = f"sm_lag_{args.trend_lag_hours}h"
    if lag_col not in data.columns:
        raise SystemExit(f"Trend baseline needs {lag_col}; available lags are configured in models/config.py")
    future_sm = data["target"].values
    current_sm = data["sm_value"].values
    lag_sm = data[lag_col].values
    _, te = station_temporal_split(data)
    eligible, y_onset = onset_filter(current_sm, future_sm, args.threshold)
    test_mask = te & eligible
    y_test = y_onset[test_mask]
    test_stations = data["station"].values[test_mask]
    test_target_times = pd.to_datetime(data.index.values[test_mask]) + pd.Timedelta(hours=args.horizon)
    y_tolerant = tolerant_onset_labels(
        stations,
        test_stations,
        test_target_times,
        args.threshold,
        args.success_tolerance_hours,
    )
    prob = _trend_probability(
        current_sm[test_mask],
        lag_sm[test_mask],
        args.horizon,
        args.trend_lag_hours,
        args.threshold,
    )
    print(
        f"Linear trend baseline: lag={args.trend_lag_hours}h test={len(y_test)} "
        f"positives={int(y_test.sum())} tolerant_positives={int(y_tolerant.sum())}"
    )
    return {
        "Linear Trend": {
            "y_true": y_test.copy(),
            "y_tolerant": y_tolerant,
            "prob": prob,
            "station": test_stations,
            "target_time": test_target_times.to_numpy(),
        }
    }


def train_lstm_classifier(args, stations: dict) -> dict:
    if args.weather:
        X_seq, X_fut, future_sm, keys, current_times, target_times = build_lstm_dataset(
            stations,
            horizon=args.horizon,
            with_weather=True,
            max_sequences=args.max_sequences,
            max_sequences_seed=args.seed,
            return_times=True,
        )
    else:
        X_seq, future_sm, keys, current_times, target_times = build_lstm_dataset(
            stations,
            horizon=args.horizon,
            with_weather=False,
            max_sequences=args.max_sequences,
            max_sequences_seed=args.seed,
            return_times=True,
        )
        X_fut = None
    current_sm = X_seq[:, -1, 0]
    tr, te = temporal_split_by_station(keys)
    eligible, y_onset = onset_filter(current_sm, future_sm, args.threshold)
    train_mask = tr & eligible
    test_mask = te & eligible

    rng = np.random.default_rng(args.seed)
    train_stations = keys[train_mask]
    train_target_times = target_times[train_mask]
    y_train_all = training_onset_labels(
        stations,
        train_stations,
        train_target_times,
        y_onset[train_mask],
        args.threshold,
        args.train_tolerance_hours,
    )
    if args.weather:
        X_train, F_train, y_train = X_seq[train_mask], X_fut[train_mask], y_train_all
        X_test, F_test, y_test = X_seq[test_mask], X_fut[test_mask], y_onset[test_mask]
        X_train, F_train, y_train = cap_rows(rng, X_train, F_train, y_train, max_n=args.max_sequence_train)
    else:
        X_train, y_train = X_seq[train_mask], y_train_all
        X_test, y_test = X_seq[test_mask], y_onset[test_mask]
        X_train, y_train = cap_rows(rng, X_train, y_train, max_n=args.max_sequence_train)
        F_train = F_test = None
    test_stations = keys[test_mask]
    test_target_times = target_times[test_mask]
    y_tolerant = tolerant_onset_labels(
        stations,
        test_stations,
        test_target_times,
        args.threshold,
        args.success_tolerance_hours,
    )

    print(
        f"LSTM onset set: train={len(y_train)} positives={int(y_train.sum())} "
        f"test={len(y_test)} exact_positives={int(y_test.sum())}"
    )
    x_mean = X_train.mean(axis=(0, 1), keepdims=True)
    x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std

    future_steps = n_future_wx = None
    train_inputs: list[np.ndarray] = [X_train]
    test_inputs: list[np.ndarray] = [X_test]
    if args.weather and F_train is not None and F_test is not None:
        f_mean = F_train.mean(axis=(0, 1), keepdims=True)
        f_std = F_train.std(axis=(0, 1), keepdims=True) + 1e-8
        F_train = (F_train - f_mean) / f_std
        F_test = (F_test - f_mean) / f_std
        future_steps, n_future_wx = F_train.shape[1], F_train.shape[2]
        train_inputs.append(F_train)
        test_inputs.append(F_test)

    model = build_lstm_classifier(X_train.shape[1], X_train.shape[2], future_steps, n_future_wx)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
    model.fit(
        train_inputs,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=args.keras_verbose,
        class_weight=class_weight_dict(y_train),
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )
    prob = model.predict(test_inputs, verbose=0).flatten()
    del model, X_seq
    gc.collect()
    return {
        "y_true": y_test.copy(),
        "y_tolerant": y_tolerant,
        "prob": prob,
        "station": test_stations,
        "target_time": test_target_times,
    }


def train_tft_classifier(args, stations: dict) -> dict:
    if args.weather:
        X_seq, X_fut, X_static, future_sm, keys, current_times, target_times = build_tft_dataset(
            stations,
            horizon=args.horizon,
            with_weather=True,
            max_sequences=args.max_sequences,
            max_sequences_seed=args.seed,
            return_times=True,
        )
    else:
        X_seq, X_static, future_sm, keys, current_times, target_times = build_tft_dataset(
            stations,
            horizon=args.horizon,
            with_weather=False,
            max_sequences=args.max_sequences,
            max_sequences_seed=args.seed,
            return_times=True,
        )
        X_fut = None
    current_sm = X_seq[:, -1, 0]
    tr, te = temporal_split_by_station(keys)
    eligible, y_onset = onset_filter(current_sm, future_sm, args.threshold)
    train_mask = tr & eligible
    test_mask = te & eligible

    rng = np.random.default_rng(args.seed + 1)
    train_stations = keys[train_mask]
    train_target_times = target_times[train_mask]
    y_train_all = training_onset_labels(
        stations,
        train_stations,
        train_target_times,
        y_onset[train_mask],
        args.threshold,
        args.train_tolerance_hours,
    )
    if args.weather:
        X_train, F_train, S_train, y_train = (
            X_seq[train_mask],
            X_fut[train_mask],
            X_static[train_mask],
            y_train_all,
        )
        X_test, F_test, S_test, y_test = (
            X_seq[test_mask],
            X_fut[test_mask],
            X_static[test_mask],
            y_onset[test_mask],
        )
        X_train, F_train, S_train, y_train = cap_rows(
            rng, X_train, F_train, S_train, y_train, max_n=args.max_sequence_train
        )
    else:
        X_train, S_train, y_train = X_seq[train_mask], X_static[train_mask], y_train_all
        X_test, S_test, y_test = X_seq[test_mask], X_static[test_mask], y_onset[test_mask]
        X_train, S_train, y_train = cap_rows(rng, X_train, S_train, y_train, max_n=args.max_sequence_train)
        F_train = F_test = None
    test_stations = keys[test_mask]
    test_target_times = target_times[test_mask]
    y_tolerant = tolerant_onset_labels(
        stations,
        test_stations,
        test_target_times,
        args.threshold,
        args.success_tolerance_hours,
    )

    print(
        f"TFT onset set: train={len(y_train)} positives={int(y_train.sum())} "
        f"test={len(y_test)} exact_positives={int(y_test.sum())}"
    )
    x_mean = X_train.mean(axis=(0, 1), keepdims=True)
    x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    s_mean = S_train.mean(axis=0)
    s_std = S_train.std(axis=0) + 1e-8
    S_train = (S_train - s_mean) / s_std
    S_test = (S_test - s_mean) / s_std

    future_steps = n_future_wx = None
    train_inputs: list[np.ndarray] = [X_train]
    test_inputs: list[np.ndarray] = [X_test]
    if args.weather and F_train is not None and F_test is not None:
        f_mean = F_train.mean(axis=(0, 1), keepdims=True)
        f_std = F_train.std(axis=(0, 1), keepdims=True) + 1e-8
        F_train = (F_train - f_mean) / f_std
        F_test = (F_test - f_mean) / f_std
        future_steps, n_future_wx = F_train.shape[1], F_train.shape[2]
        train_inputs.append(F_train)
        test_inputs.append(F_test)
    train_inputs.append(S_train)
    test_inputs.append(S_test)

    model = build_tft_classifier(X_train.shape[1], X_train.shape[2], S_train.shape[1], future_steps, n_future_wx)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    model.fit(
        train_inputs,
        y_train,
        epochs=args.epochs + 10,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=args.keras_verbose,
        class_weight=class_weight_dict(y_train),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0),
        ],
    )
    prob = model.predict(test_inputs, verbose=0).flatten()
    del model, X_seq
    gc.collect()
    return {
        "y_true": y_test.copy(),
        "y_tolerant": y_tolerant,
        "prob": prob,
        "station": test_stations,
        "target_time": test_target_times,
    }


def save_outputs(out_dir: str, results: dict[str, dict], args) -> None:
    rows = []
    tolerant_rows = []
    event_rows = []
    sweep_frames = []
    for name, result in results.items():
        y_true = result["y_true"]
        y_tolerant = result["y_tolerant"]
        prob = result["prob"]
        threshold_target = y_tolerant if args.threshold_selection == "tolerant" else y_true
        sweep, best_f1_threshold, best_recall_threshold = threshold_sweeps(
            threshold_target, prob, max_false_alarm_rate=args.max_false_alarm_rate
        )
        sweep.insert(0, "Model", name)
        sweep.insert(1, "Threshold selection target", args.threshold_selection)
        sweep_frames.append(sweep)
        thresholds = [
            ("default_0.5", 0.5),
            ("best_f1", best_f1_threshold),
            (f"max_recall_at_fa<={args.max_false_alarm_rate}", best_recall_threshold),
        ]
        for threshold_label, threshold in thresholds:
            rows.append(evaluate_probs(name, y_true, prob, threshold, threshold_label))
            tolerant_rows.append(evaluate_probs(name, y_tolerant, prob, threshold, f"tolerant_{threshold_label}"))
            event_rows.append(
                evaluate_event_probs(
                    name,
                    y_tolerant,
                    prob,
                    result["station"],
                    result["target_time"],
                    threshold,
                    threshold_label,
                )
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(out_dir, "drought_onset_classification_metrics.csv"), index=False)
    tolerant_metrics = pd.DataFrame(tolerant_rows)
    tolerant_metrics.to_csv(os.path.join(out_dir, "drought_onset_tolerant_metrics.csv"), index=False)
    event_metrics = pd.DataFrame(event_rows)
    event_metrics.to_csv(os.path.join(out_dir, "drought_onset_event_metrics.csv"), index=False)
    pd.concat(sweep_frames, ignore_index=True).to_csv(
        os.path.join(out_dir, "drought_onset_threshold_sweep.csv"), index=False
    )

    print("\nExact-horizon metrics:")
    print(metrics.to_string(index=False))
    print(f"\nTolerant sample metrics (success window: +{args.success_tolerance_hours}h):")
    print(tolerant_metrics.to_string(index=False))
    print("\nEvent-level tolerant metrics:")
    print(event_metrics.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    best_rows = metrics[metrics["Threshold label"] == "best_f1"]
    x = np.arange(len(best_rows))
    axes[0].bar(x, best_rows["Recall"], color=[COLORS[m] for m in best_rows["Model"]])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(best_rows["Model"], rotation=12)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Onset recall at best-F1 threshold")
    axes[1].bar(x, best_rows["Precision"], color=[COLORS[m] for m in best_rows["Model"]])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(best_rows["Model"], rotation=12)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Onset precision at best-F1 threshold")
    plt.suptitle(f"Drought onset classification, t+{args.horizon}h")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drought_onset_precision_recall_bar.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, result in results.items():
        y_true = result["y_tolerant"] if args.threshold_selection == "tolerant" else result["y_true"]
        prob = result["prob"]
        prec, rec, _ = precision_recall_curve(y_true, prob)
        axes[0].plot(rec, prec, color=COLORS[name], label=f"{name} AP={average_precision_score(y_true, prob):.3f}")
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, prob)
            axes[1].plot(fpr, tpr, color=COLORS[name], label=f"{name} AUC={roc_auc_score(y_true, prob):.3f}")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-recall")
    axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC")
    axes[1].legend(fontsize=8)
    plt.suptitle(f"Drought onset classifier curves, t+{args.horizon}h ({args.threshold_selection} target)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drought_onset_curves.png"), dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default=None, help="Output folder under images/ (default: next test N)")
    parser.add_argument("--horizon", type=int, default=72, help="Forecast horizon in hours (default: 72)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Drought threshold for soil moisture")
    parser.add_argument("--no-weather", action="store_true", help="Train without weather/future-weather inputs")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baselines (persistence + linear trend)")
    parser.add_argument("--trend-lag-hours", type=int, default=24, help="Lag used for linear trend baseline")
    parser.add_argument(
        "--train-tolerance-hours",
        type=int,
        default=0,
        help="Broaden training positives to drought occurring from horizon to horizon+N hours (default: exact horizon)",
    )
    parser.add_argument(
        "--threshold-selection",
        choices=["exact", "tolerant"],
        default="exact",
        help="Choose whether best-F1 and false-alarm thresholds are selected on exact or tolerant labels",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--keras-verbose", type=int, default=1)
    parser.add_argument("--max-tabular-train", type=int, default=None, help="Optional cap for tabular train rows")
    parser.add_argument("--max-sequence-train", type=int, default=100_000, help="Cap LSTM/TFT train rows")
    parser.add_argument("--max-sequences", type=int, default=900_000, help="Cap total built sequence samples before split")
    parser.add_argument(
        "--success-tolerance-hours",
        type=int,
        default=6,
        help="Evaluation-only tolerance after the target time; training labels remain exact-horizon onset labels",
    )
    parser.add_argument(
        "--max-false-alarm-rate",
        type=float,
        default=0.05,
        help="Threshold report: maximize recall while false alarm rate is at or below this value",
    )
    args = parser.parse_args()
    args.weather = not args.no_weather
    return args


def main():
    args = parse_args()
    run_name = args.name or next_test_folder_name()
    out_dir = os.path.join(IMAGES_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\nDrought onset classification")
    print(f"Horizon: t+{args.horizon}h")
    print(f"Drought threshold: {args.threshold}")
    print(f"Success tolerance: +{args.success_tolerance_hours}h (evaluation only)")
    print(f"Training tolerance: +{args.train_tolerance_hours}h")
    print(f"Threshold selection target: {args.threshold_selection}")
    print(f"Weather inputs: {'yes' if args.weather else 'no'}")
    if args.baseline_only:
        print("Mode: persistence baseline only")
    print(f"Output: {out_dir}\n")

    stations = load_all_stations(with_weather=args.weather)
    results = build_persistence_baseline(args, stations)
    results.update(build_linear_trend_baseline(args, stations))
    if not args.baseline_only:
        results.update(train_tabular_classifiers(args, stations))
        results["LSTM"] = train_lstm_classifier(args, stations)
        results["TFT"] = train_tft_classifier(args, stations)
    save_outputs(out_dir, results, args)
    print(f"\nSaved drought onset classification outputs to {out_dir}")


if __name__ == "__main__":
    main()
