"""
Train RF, XGBoost, LSTM, TFT twice: without weather inputs vs with weather + oracle
future weather (matching run_experiment.py). Uses per-station temporal splits and
restricts both regimes to stations that have complete merged weather columns so the
comparison is on the same station cohort.

Usage (from repo root):
    python scripts/weather_ablation_four_models.py [run_name]
    python scripts/weather_ablation_four_models.py [run_name] --horizon 120

Forecast horizon defaults to **120 hours** (oracle weather at t+24h,t+48h,...,t+120h for with-weather runs).
If run_name is omitted, uses the next numbered test folder like run_experiment.

Long horizons with weather expand LSTM/TFT tensors; by default LSTM/TFT datasets are capped
(--max-sequences, shuffled station order) so builds fit in RAM. Use --full-sequence-dataset to disable
the cap if you have enough memory.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor

from prepare import (
    DERIVED_WEATHER_COLS,
    WEATHER_COLS,
    build_lstm_dataset,
    build_tft_dataset,
    build_xgboost_dataset,
    get_feature_cols,
    load_all_stations,
    station_temporal_split,
    temporal_split_by_station,
)
from config import XGB_PARAMS

VERB_LSTM_TFT = 0  # set to 1 for Keras batch progress bars
HORIZON = 120  # updated at runtime from CLI --horizon
IMAGES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))

WX_COLS = list(WEATHER_COLS) + list(DERIVED_WEATHER_COLS)


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
    n = max(nums) + 1 if nums else 1
    return f"test {n}"


def stations_with_complete_weather() -> tuple[dict, dict]:
    """Pair of (stations_no_wx, stations_wx) on the same key set."""
    raw_wx = load_all_stations(with_weather=True)
    raw_nw = load_all_stations(with_weather=False)
    keys = [
        k
        for k in raw_wx
        if k in raw_nw
        and all(c in raw_wx[k][0].columns for c in WX_COLS)
    ]
    keys.sort()
    return {k: raw_nw[k] for k in keys}, {k: raw_wx[k] for k in keys}


def _build_tft(seq_len: int, n_features: int, n_static: int, future_steps=None, n_future_wx=None):
    seq_in = keras.Input(shape=(seq_len, n_features), name="sequence")
    sta_in = keras.Input(shape=(n_static,), name="static")
    fut_in = None
    if future_steps is not None and n_future_wx:
        fut_in = keras.Input(shape=(future_steps, n_future_wx), name="future_weather")
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
    if fut_in is not None:
        fw = layers.Flatten()(fut_in)
        fw = layers.Dense(16, activation="elu")(fw)
        fw = layers.LayerNormalization()(fw)
        x = layers.Concatenate()([x, fw])
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    inputs = [seq_in, sta_in] if fut_in is None else [seq_in, fut_in, sta_in]
    return keras.Model(inputs=inputs, outputs=out)


def _build_lstm_dual(seq_len: int, n_features: int, future_steps: int, n_future_wx: int):
    seq_in = keras.Input(shape=(seq_len, n_features), name="sequence")
    fut_in = keras.Input(shape=(future_steps, n_future_wx), name="future_weather")
    x = layers.LSTM(64, return_sequences=True)(seq_in)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    fw = layers.Flatten()(fut_in)
    fw = layers.Dense(32, activation="relu")(fw)
    x = layers.Concatenate()([x, fw])
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    return keras.Model([seq_in, fut_in], out)


def drought_recall(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.30) -> float:
    actual = (y_true < thr).astype(int)
    pred = (y_pred < thr).astype(int)
    return float(recall_score(actual, pred, zero_division=0))


def cap_train(rng, *arrays, max_n: int = 100_000):
    """Subset first dimension of each array to at most max_n rows (train only)."""
    arrays = tuple(arrays)
    n = len(arrays[0])
    if n <= max_n:
        return arrays
    sub = rng.choice(n, size=max_n, replace=False)
    sub.sort()
    return tuple(a[sub] for a in arrays)


def metrics_row(model: str, weather: bool, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": model,
        "Weather": "yes" if weather else "no",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "DroughtRecall": round(drought_recall(y_true, y_pred), 3),
        "Horizon": f"t+{HORIZON}h",
    }


def train_tabular(
    stations: dict,
    with_weather: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fc = get_feature_cols(with_weather=with_weather, horizon=HORIZON)
    data = build_xgboost_dataset(stations, horizon=HORIZON, with_weather=with_weather)
    X = data[fc].values
    y = data["target"].values
    tr, te = station_temporal_split(data)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]
    print(f"  [{label}] tabular rows={len(X)}  features={len(fc)}  test={te.sum()}")

    print(f"    Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, max_features="sqrt", random_state=0, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    pred_rf = rf.predict(X_te)
    del rf

    print(f"    XGBoost...")
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X[tr], y[tr])
    pred_xgb = xgb.predict(X_te)
    del xgb

    gc.collect()
    return y_te.copy(), pred_rf, pred_xgb


def lstm_prepare_and_train(
    stations: dict,
    with_weather: bool,
    label: str,
    rng_seed: int,
    max_sequences: int | None = None,
    max_sequences_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    if with_weather:
        X_seq, X_fut, y, keys = build_lstm_dataset(
            stations,
            horizon=HORIZON,
            with_weather=True,
            max_sequences=max_sequences,
            max_sequences_seed=max_sequences_seed,
        )
    else:
        X_seq, y, keys = build_lstm_dataset(
            stations,
            horizon=HORIZON,
            with_weather=False,
            max_sequences=max_sequences,
            max_sequences_seed=max_sequences_seed,
        )

    seq_tr, seq_te = temporal_split_by_station(keys)
    if with_weather:
        Xt_tr_full, xf_tr_full, yt_tr_full = X_seq[seq_tr], X_fut[seq_tr], y[seq_tr]
        Xt_te_raw, xf_te_raw, yt_te = X_seq[seq_te], X_fut[seq_te], y[seq_te]
        Xt_tr_full, xf_tr_full, yt_tr_full = cap_train(rng, Xt_tr_full, xf_tr_full, yt_tr_full)
        mu = Xt_tr_full.mean(axis=(0, 1), keepdims=True)
        sd = Xt_tr_full.std(axis=(0, 1), keepdims=True) + 1e-8
        XT_tr = (Xt_tr_full - mu) / sd
        XT_te = (Xt_te_raw - mu) / sd
        fm = xf_tr_full.mean(axis=(0, 1), keepdims=True)
        fs = xf_tr_full.std(axis=(0, 1), keepdims=True) + 1e-8
        Xf_tr = (xf_tr_full - fm) / fs
        Xf_te = (xf_te_raw - fm) / fs
        yt_tr = yt_tr_full
        n_fut_s, n_fut_w = Xf_tr.shape[1], Xf_tr.shape[2]
        model = _build_lstm_dual(XT_tr.shape[1], XT_tr.shape[2], n_fut_s, n_fut_w)
        model.compile(optimizer="adam", loss="mae")
        model.fit(
            [XT_tr, Xf_tr],
            yt_tr,
            epochs=20,
            batch_size=512,
            validation_split=0.1,
            verbose=VERB_LSTM_TFT,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ],
        )
        pred = model.predict([XT_te, Xf_te], verbose=0).flatten()
    else:
        Xt_tr_full, yt_tr_full = X_seq[seq_tr], y[seq_tr]
        Xt_te_raw, yt_te = X_seq[seq_te], y[seq_te]
        Xt_tr_full, yt_tr_full = cap_train(rng, Xt_tr_full, yt_tr_full)
        mu = Xt_tr_full.mean(axis=(0, 1), keepdims=True)
        sd = Xt_tr_full.std(axis=(0, 1), keepdims=True) + 1e-8
        XT_tr = (Xt_tr_full - mu) / sd
        XT_te = (Xt_te_raw - mu) / sd
        yt_tr = yt_tr_full
        model = keras.Sequential(
            [
                keras.Input(shape=(XT_tr.shape[1], XT_tr.shape[2])),
                layers.LSTM(64, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(32),
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")
        model.fit(
            XT_tr,
            yt_tr,
            epochs=20,
            batch_size=512,
            validation_split=0.1,
            verbose=VERB_LSTM_TFT,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ],
        )
        pred = model.predict(XT_te, verbose=0).flatten()

    del model
    gc.collect()
    print(f"  [{label}] LSTM test={len(yt_te)}")
    return yt_te, pred


def tft_prepare_and_train(
    stations: dict,
    with_weather: bool,
    label: str,
    rng_seed: int,
    max_sequences: int | None = None,
    max_sequences_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    if with_weather:
        Xt, xf, sta, yt, keys = build_tft_dataset(
            stations,
            horizon=HORIZON,
            with_weather=True,
            max_sequences=max_sequences,
            max_sequences_seed=max_sequences_seed,
        )
    else:
        Xt, sta, yt, keys = build_tft_dataset(
            stations,
            horizon=HORIZON,
            with_weather=False,
            max_sequences=max_sequences,
            max_sequences_seed=max_sequences_seed,
        )

    tr, te = temporal_split_by_station(keys)
    if with_weather:
        a, b, c, d = Xt[tr], xf[tr], sta[tr], yt[tr]
        a, b, c, d = cap_train(rng, a, b, c, d)
        am = a.mean(axis=(0, 1), keepdims=True)
        asd = a.std(axis=(0, 1), keepdims=True) + 1e-8
        A_tr = (a - am) / asd
        A_te = (Xt[te] - am) / asd
        bm = b.mean(axis=(0, 1), keepdims=True)
        bd = b.std(axis=(0, 1), keepdims=True) + 1e-8
        B_tr = (b - bm) / bd
        B_te = (xf[te] - bm) / bd
        sm = c.mean(axis=0)
        ss = c.std(axis=0) + 1e-8
        C_tr = (c - sm) / ss
        C_te = (sta[te] - sm) / ss
        y_tr = d
        n_f_s, n_f_w = B_tr.shape[1], B_tr.shape[2]
        tft = _build_tft(A_tr.shape[1], A_tr.shape[2], C_tr.shape[1], future_steps=n_f_s, n_future_wx=n_f_w)
        tft.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss="mae")
        tft.fit(
            [A_tr, B_tr, C_tr],
            y_tr,
            epochs=30,
            batch_size=512,
            validation_split=0.1,
            verbose=VERB_LSTM_TFT,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0
                ),
            ],
        )
        pred = tft.predict([A_te, B_te, C_te], verbose=0).flatten()
    else:
        a, b, d = Xt[tr], sta[tr], yt[tr]
        a, b, d = cap_train(rng, a, b, d)
        am = a.mean(axis=(0, 1), keepdims=True)
        asd = a.std(axis=(0, 1), keepdims=True) + 1e-8
        A_tr = (a - am) / asd
        A_te = (Xt[te] - am) / asd
        sm = b.mean(axis=0)
        ss = b.std(axis=0) + 1e-8
        B_tr = (b - sm) / ss
        B_te = (sta[te] - sm) / ss
        y_tr = d
        tft = _build_tft(A_tr.shape[1], A_tr.shape[2], B_tr.shape[1])
        tft.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss="mae")
        tft.fit(
            [A_tr, B_tr],
            y_tr,
            epochs=30,
            batch_size=512,
            validation_split=0.1,
            verbose=VERB_LSTM_TFT,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0
                ),
            ],
        )
        pred = tft.predict([A_te, B_te], verbose=0).flatten()

    yt_te = yt[te]
    del tft
    gc.collect()
    print(f"  [{label}] TFT test={len(yt_te)}")
    return yt_te, pred


def main():
    global HORIZON
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Folder name under images/ (defaults to next numbered test)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=120,
        help="Prediction horizon (hours); oracle future weather uses steps 24..H in steps of 24 (default 120)",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=900_000,
        help="Max LSTM/TFT training samples collected before stacking arrays (RAM). Ignored if --full-sequence-dataset.",
    )
    parser.add_argument(
        "--max-sequences-seed",
        type=int,
        default=0,
        help="RNG seed for shuffling station order when applying --max-sequences.",
    )
    parser.add_argument(
        "--full-sequence-dataset",
        action="store_true",
        help="Build full LSTM/TFT sequence sets (may require large RAM at long horizons with weather).",
    )
    args = parser.parse_args()
    HORIZON = int(args.horizon)
    if HORIZON < 24:
        raise SystemExit("--horizon must be at least 24 (oracle slices are spaced by 24h).")
    max_sequences = None if args.full_sequence_dataset else int(args.max_sequences)

    run_name = args.name or next_test_folder_name()
    out_dir = os.path.join(IMAGES_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\nWeather ablation — four models — same station cohort (complete wx merge)")
    print(f"Horizon t+{HORIZON}h")
    if max_sequences is not None:
        print(f"LSTM/TFT sequence cap: {max_sequences} (seed={args.max_sequences_seed})")
    else:
        print("LSTM/TFT sequence cap: disabled (full dataset)")
    print(f"Output -> {out_dir}\n")

    stations_nw, stations_w = stations_with_complete_weather()
    print(f"Stations in cohort: {len(stations_w)}")

    rows: list[dict] = []

    # ── NO WEATHER ─────────────────────────────────────
    print("\n=== NO WEATHER ===")
    y_te, rf_nw, xgb_nw = train_tabular(stations_nw, False, "no_wx_tab")
    rows.append(metrics_row("Random Forest", False, y_te, rf_nw))
    rows.append(metrics_row("XGBoost", False, y_te, xgb_nw))

    yt, pred = lstm_prepare_and_train(
        stations_nw,
        False,
        "no_wx_lstm",
        rng_seed=0,
        max_sequences=max_sequences,
        max_sequences_seed=args.max_sequences_seed,
    )
    rows.append(metrics_row("LSTM", False, yt, pred))

    yt, pred = tft_prepare_and_train(
        stations_nw,
        False,
        "no_wx_tft",
        rng_seed=1,
        max_sequences=max_sequences,
        max_sequences_seed=args.max_sequences_seed,
    )
    rows.append(metrics_row("TFT", False, yt, pred))

    # ── WITH WEATHER (+ oracle future windows) ────────
    print("\n=== WITH WEATHER (oracle future) ===")
    y_te_w, rf_w, xgb_w = train_tabular(stations_w, True, "wx_tab")
    rows.append(metrics_row("Random Forest", True, y_te_w, rf_w))
    rows.append(metrics_row("XGBoost", True, y_te_w, xgb_w))

    yt, pred = lstm_prepare_and_train(
        stations_w,
        True,
        "wx_lstm",
        rng_seed=0,
        max_sequences=max_sequences,
        max_sequences_seed=args.max_sequences_seed,
    )
    rows.append(metrics_row("LSTM", True, yt, pred))

    yt, pred = tft_prepare_and_train(
        stations_w,
        True,
        "wx_tft",
        rng_seed=1,
        max_sequences=max_sequences,
        max_sequences_seed=args.max_sequences_seed,
    )
    rows.append(metrics_row("TFT", True, yt, pred))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "weather_ablation_four_models.csv")
    df.to_csv(csv_path, index=False)

    print("\nFull table:")
    print(df.to_string(index=False))
    print("\nMAE by Model (no vs yes):")
    print(df.pivot_table(index="Model", columns="Weather", values="MAE").to_string())
    print(f"\nSaved {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    models = sorted(df["Model"].unique())
    x = np.arange(len(models))
    w = 0.35
    y_no = [df[(df.Model == m) & (df.Weather == "no")]["MAE"].values[0] for m in models]
    y_yes = [df[(df.Model == m) & (df.Weather == "yes")]["MAE"].values[0] for m in models]
    ax.bar(x - w / 2, y_no, w, label="No weather")
    ax.bar(x + w / 2, y_yes, w, label="With weather (+ oracle)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12)
    ax.set_ylabel("MAE")
    ax.set_title(f"Weather vs no weather — t+{HORIZON}h")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "weather_ablation_mae_bar.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    r2_no = [df[(df.Model == m) & (df.Weather == "no")]["R2"].values[0] for m in models]
    r2_yes = [df[(df.Model == m) & (df.Weather == "yes")]["R2"].values[0] for m in models]
    ax.bar(x - w / 2, r2_no, w, label="No weather")
    ax.bar(x + w / 2, r2_yes, w, label="With weather (+ oracle)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12)
    ax.set_ylabel("R^2")
    ax.set_title(f"Weather vs no weather — R2 — t+{HORIZON}h")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "weather_ablation_r2_bar.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
