"""
Station-holdout generalisation test.

Holds out 8 stations (stratified by country) from training entirely, trains
all four models on the remaining 40, and reports test metrics on the held-out
stations. Run twice — with and without weather — so the comparison is clean.

This is the *cross-station* generalisation question:
  "Does the model still work on sensors it has never seen during training?"

Compare these numbers to the per-station temporal split in test 10 / test 14
to see how much harder pure generalisation is.

Usage:
    python scripts/run_station_holdout.py
    python scripts/run_station_holdout.py "test 15 - station holdout"
"""

import argparse
import gc
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    recall_score, precision_score, f1_score, roc_auc_score,
)
from tensorflow import keras

from prepare import (
    load_all_stations,
    build_xgboost_dataset,
    build_lstm_dataset,
    build_tft_dataset,
    get_feature_cols,
)
from config import THRESHOLD, XGB_PARAMS
from xgboost import XGBRegressor

# Both model builders are inlined here. Importing from run_experiment.py would
# trigger that script's module-level argparse and break our CLI.


def _build_lstm_dual(seq_len, n_features, future_steps, n_future_wx):
    """LSTM with past sequence + oracle future weather (matches run_experiment.py)."""
    from tensorflow.keras import layers
    seq_in = keras.Input(shape=(seq_len, n_features), name='sequence')
    fut_in = keras.Input(shape=(future_steps, n_future_wx), name='future_weather')
    x = layers.LSTM(64, return_sequences=True)(seq_in)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    fw = layers.Flatten()(fut_in)
    fw = layers.Dense(32, activation='relu')(fw)
    x = layers.Concatenate()([x, fw])
    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1)(x)
    return keras.Model([seq_in, fut_in], out)


def _build_tft(seq_len, n_features, n_static, future_steps=None, n_future_wx=None):
    """TFT v3 — minimal targeted regularisation.

    v2 over-dropped (6 layers); the model couldn't learn. v3 keeps only two
    dropouts, placed where station-overfit actually happens:
      • Dropout(0.2) inside the static encoder — the dominant failure mode
        for cross-station generalisation (static feats act as station IDs).
      • Dropout(0.2) after the LSTM — regularises the temporal representation
        before it reaches attention.
    Everything else matches the original _build_tft from run_experiment.py.
    """
    from tensorflow.keras import layers
    seq_in = keras.Input(shape=(seq_len, n_features), name='sequence')
    sta_in = keras.Input(shape=(n_static,), name='static')
    fut_in = None
    if future_steps is not None and n_future_wx:
        fut_in = keras.Input(shape=(future_steps, n_future_wx), name='future_weather')

    s = layers.Dense(16, activation='elu')(sta_in)
    s = layers.Dropout(0.2)(s)
    s = layers.Dense(16)(s)
    s_exp = layers.RepeatVector(seq_len)(s)
    x = layers.Concatenate(axis=-1)([seq_in, s_exp])

    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)

    a = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    g = layers.Dense(64, activation='sigmoid')(a)
    x = layers.Add()([layers.Multiply()([g, a]), x])
    x = layers.LayerNormalization()(x)

    x = layers.Lambda(lambda t: t[:, -1, :])(x)

    if fut_in is not None:
        fw = layers.Flatten()(fut_in)
        fw = layers.Dense(16, activation='elu')(fw)
        fw = layers.LayerNormalization()(fw)
        x = layers.Concatenate()([x, fw])

    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1)(x)

    inputs = [seq_in, sta_in] if fut_in is None else [seq_in, fut_in, sta_in]
    return keras.Model(inputs=inputs, outputs=out)

HORIZON = 72
N_HOLDOUT = 8
SEED = 0
MAX_TRAIN_SEQ = 100_000

IMAGES_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'images')

keras.utils.set_random_seed(SEED)


# ── station country mapping (same boxes as app.py) ────────────────────────────
def _country_from_latlon(lat, lon):
    if -3.0 <= lat <= -1.0 and 28.5 <= lon <= 31.0:
        return 'Rwanda'
    if -1.5 < lat <= 4.5 and 29.5 <= lon < 35.5:
        return 'Uganda'
    if -5.0 <= lat <= 5.5 and 33.0 <= lon <= 42.5:
        return 'Kenya'
    return 'Other'


def pick_holdout_stations(stations, n=N_HOLDOUT, seed=SEED):
    """Stratified-by-country selection of n holdout station keys."""
    rng = np.random.default_rng(seed)
    by_country = {}
    for key, (_, meta) in stations.items():
        c = _country_from_latlon(meta['latitude'], meta['longitude'])
        by_country.setdefault(c, []).append(key)

    total = sum(len(v) for v in by_country.values())
    holdout = []
    remainders = []
    for country, keys in by_country.items():
        share = len(keys) / total * n
        take = int(np.floor(share))
        keys_sorted = sorted(keys)
        rng.shuffle(keys_sorted)
        holdout.extend(keys_sorted[:take])
        remainders.append((country, keys_sorted[take:], share - take))

    # Distribute leftover slots by largest fractional remainder.
    remainders.sort(key=lambda r: -r[2])
    i = 0
    while len(holdout) < n and i < len(remainders):
        country, leftover, _ = remainders[i]
        if leftover:
            holdout.append(leftover.pop(0))
        i = (i + 1) % len(remainders)

    return sorted(holdout)


# ── metrics helpers ───────────────────────────────────────────────────────────
def _regression_metrics(y_true, y_pred):
    actual = (y_true < THRESHOLD).astype(int)
    pred = (y_pred < THRESHOLD).astype(int)
    out = {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(mean_squared_error(y_true, y_pred) ** 0.5),
        'R2': float(r2_score(y_true, y_pred)),
        'Recall': float(recall_score(actual, pred, zero_division=0)),
        'Precision': float(precision_score(actual, pred, zero_division=0)),
        'F1': float(f1_score(actual, pred, zero_division=0)),
    }
    # ROC AUC needs both classes present.
    if len(np.unique(actual)) == 2:
        out['ROC AUC'] = float(roc_auc_score(actual, -y_pred))
    else:
        out['ROC AUC'] = np.nan
    return out


def _per_station(y_true, y_pred, station_keys):
    rows = []
    for s in np.unique(station_keys):
        m = station_keys == s
        if m.sum() < 50:
            continue
        rows.append({
            'station': s,
            'n': int(m.sum()),
            **_regression_metrics(y_true[m], y_pred[m]),
        })
    return pd.DataFrame(rows).sort_values('MAE')


# ── tabular path (XGB + RF) ───────────────────────────────────────────────────
def run_tabular(stations, holdout, with_weather):
    data = build_xgboost_dataset(stations, horizon=HORIZON, with_weather=with_weather)
    feat_cols = get_feature_cols(with_weather=with_weather, horizon=HORIZON)
    test_mask = data['station'].isin(holdout).values
    X = data[feat_cols].values
    y = data['target'].values
    X_tr, X_te = X[~test_mask], X[test_mask]
    y_tr, y_te = y[~test_mask], y[test_mask]
    keys_te = data['station'].values[test_mask]

    print(f'    tabular train={len(y_tr):,}  test={len(y_te):,}')

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, max_features='sqrt',
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)

    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X_tr, y_tr)
    xgb_pred = xgb.predict(X_te)

    return {
        'Random Forest': (y_te, rf_pred, keys_te),
        'XGBoost':       (y_te, xgb_pred, keys_te),
    }


# ── LSTM path ─────────────────────────────────────────────────────────────────
def run_lstm(stations, holdout, with_weather):
    if with_weather:
        X_seq, X_fut, y, keys = build_lstm_dataset(
            stations, horizon=HORIZON, with_weather=True,
            max_sequences_seed=SEED,
        )
    else:
        X_seq, y, keys = build_lstm_dataset(
            stations, horizon=HORIZON, with_weather=False,
            max_sequences_seed=SEED,
        )
        X_fut = None

    test_mask = np.isin(keys, list(holdout))
    train_mask = ~test_mask

    # Cap training sequences after the split so the test set is intact.
    rng = np.random.default_rng(SEED)
    train_idx = np.where(train_mask)[0]
    if len(train_idx) > MAX_TRAIN_SEQ:
        train_idx = np.sort(rng.choice(train_idx, MAX_TRAIN_SEQ, replace=False))

    X_tr, y_tr = X_seq[train_idx], y[train_idx]
    X_te, y_te = X_seq[test_mask], y[test_mask]
    keys_te = keys[test_mask]

    # Train-only normalisation.
    mu = X_tr.mean(axis=(0, 1), keepdims=True)
    sd = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    X_tr_n = (X_tr - mu) / sd
    X_te_n = (X_te - mu) / sd

    print(f'    lstm train={len(y_tr):,}  test={len(y_te):,}')

    if with_weather:
        fut_tr = X_fut[train_idx]
        fut_te = X_fut[test_mask]
        fmu = fut_tr.mean(axis=(0, 1), keepdims=True)
        fsd = fut_tr.std(axis=(0, 1), keepdims=True) + 1e-8
        fut_tr_n = (fut_tr - fmu) / fsd
        fut_te_n = (fut_te - fmu) / fsd

        model = _build_lstm_dual(X_tr.shape[1], X_tr.shape[2],
                                  fut_tr.shape[1], fut_tr.shape[2])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
                      loss='mae')
        model.fit(
            [X_tr_n, fut_tr_n], y_tr,
            epochs=20, batch_size=1024, validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,
                                              restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=2, min_lr=1e-6, verbose=0),
            ],
            verbose=1,
        )
        pred = model.predict([X_te_n, fut_te_n], verbose=0).flatten()
    else:
        model = keras.Sequential([
            keras.Input(shape=(X_tr.shape[1], X_tr.shape[2])),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
                      loss='mae')
        model.fit(
            X_tr_n, y_tr,
            epochs=20, batch_size=1024, validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,
                                              restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=2, min_lr=1e-6, verbose=0),
            ],
            verbose=1,
        )
        pred = model.predict(X_te_n, verbose=0).flatten()

    return y_te, pred, keys_te


# ── TFT path ──────────────────────────────────────────────────────────────────
def run_tft(stations, holdout, with_weather):
    if with_weather:
        X_seq, X_fut, X_static, y, keys = build_tft_dataset(
            stations, horizon=HORIZON, with_weather=True,
            max_sequences_seed=SEED,
        )
    else:
        X_seq, X_static, y, keys = build_tft_dataset(
            stations, horizon=HORIZON, with_weather=False,
            max_sequences_seed=SEED,
        )
        X_fut = None

    test_mask = np.isin(keys, list(holdout))
    train_mask = ~test_mask

    rng = np.random.default_rng(SEED + 1)
    train_idx = np.where(train_mask)[0]
    if len(train_idx) > MAX_TRAIN_SEQ:
        train_idx = np.sort(rng.choice(train_idx, MAX_TRAIN_SEQ, replace=False))

    X_tr, y_tr = X_seq[train_idx], y[train_idx]
    X_te, y_te = X_seq[test_mask], y[test_mask]
    stat_tr, stat_te = X_static[train_idx], X_static[test_mask]
    keys_te = keys[test_mask]

    mu = X_tr.mean(axis=(0, 1), keepdims=True)
    sd = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    X_tr_n = (X_tr - mu) / sd
    X_te_n = (X_te - mu) / sd
    smu = stat_tr.mean(axis=0)
    ssd = stat_tr.std(axis=0) + 1e-8
    stat_tr_n = (stat_tr - smu) / ssd
    stat_te_n = (stat_te - smu) / ssd

    print(f'    tft  train={len(y_tr):,}  test={len(y_te):,}')

    if with_weather:
        fut_tr = X_fut[train_idx]
        fut_te = X_fut[test_mask]
        fmu = fut_tr.mean(axis=(0, 1), keepdims=True)
        fsd = fut_tr.std(axis=(0, 1), keepdims=True) + 1e-8
        fut_tr_n = (fut_tr - fmu) / fsd
        fut_te_n = (fut_te - fmu) / fsd

        model = _build_tft(
            X_tr.shape[1], X_tr.shape[2], stat_tr.shape[1],
            future_steps=fut_tr.shape[1], n_future_wx=fut_tr.shape[2],
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
                      loss='mae')
        model.fit(
            [X_tr_n, fut_tr_n, stat_tr_n], y_tr,
            epochs=30, batch_size=1024, validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                              restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=2, min_lr=1e-6, verbose=0),
            ],
            verbose=1,
        )
        pred = model.predict([X_te_n, fut_te_n, stat_te_n], verbose=0).flatten()
    else:
        model = _build_tft(X_tr.shape[1], X_tr.shape[2], stat_tr.shape[1])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
                      loss='mae')
        model.fit(
            [X_tr_n, stat_tr_n], y_tr,
            epochs=30, batch_size=1024, validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                              restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=2, min_lr=1e-6, verbose=0),
            ],
            verbose=1,
        )
        pred = model.predict([X_te_n, stat_te_n], verbose=0).flatten()

    return y_te, pred, keys_te


# ── orchestrator ──────────────────────────────────────────────────────────────
def _next_test_name():
    existing = [d for d in os.listdir(IMAGES_ROOT)
                if os.path.isdir(os.path.join(IMAGES_ROOT, d)) and d.startswith('test ')]
    nums = []
    for d in existing:
        try:
            nums.append(int(d.split(' ')[1]))
        except (IndexError, ValueError):
            pass
    n = max(nums) + 1 if nums else 1
    return f'test {n} - station holdout'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?', default=None)
    ap.add_argument('--models', default='all',
                    help='Comma-separated subset: rf,xgb,lstm,tft. Default: all.')
    ap.add_argument('--into', default=None,
                    help='Existing test folder to merge rerun results into '
                         '(e.g. "test 17 - station holdout"). Required if --models is a subset.')
    args = ap.parse_args()

    requested = {m.strip().lower() for m in args.models.split(',')} if args.models != 'all' \
        else {'rf', 'xgb', 'lstm', 'tft'}
    subset_run = requested != {'rf', 'xgb', 'lstm', 'tft'}

    if subset_run and not args.into:
        raise SystemExit('--into is required when running a subset of models')

    run_name = args.into if subset_run else (args.name or _next_test_name())
    out_dir = os.path.join(IMAGES_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n{"="*60}\nRun: {run_name}'
          f'{"  (subset rerun: " + ",".join(sorted(requested)) + ")" if subset_run else ""}'
          f'\nOutput: images/{run_name}/\n{"="*60}\n')

    print('Loading stations (with weather, used for both configs)...')
    stations = load_all_stations(with_weather=True)
    print(f'  loaded {len(stations)} stations')

    holdout = pick_holdout_stations(stations, n=N_HOLDOUT, seed=SEED)
    train_keys = sorted(set(stations.keys()) - set(holdout))

    print(f'\nHoldout stations ({len(holdout)}):')
    for k in holdout:
        meta = stations[k][1]
        c = _country_from_latlon(meta['latitude'], meta['longitude'])
        print(f'  {k:<60} {c}')
    print(f'\nTraining on {len(train_keys)} stations.\n')

    pd.DataFrame({
        'station': holdout,
        'country': [_country_from_latlon(*[stations[k][1][c] for c in ('latitude', 'longitude')]) for k in holdout],
        'network': [stations[k][1]['network'] for k in holdout],
    }).to_csv(os.path.join(out_dir, 'holdout_stations.csv'), index=False)

    all_metrics = []
    all_per_station = []

    run_tabular_flag = any(m in requested for m in ('rf', 'xgb'))
    rf_xgb_filter = {m for m in ('rf', 'xgb') if m in requested}

    for with_weather in (False, True):
        tag = 'with weather' if with_weather else 'no weather'
        print(f'\n{"="*60}\nCONFIG: {tag}\n{"="*60}')

        results = {}

        if run_tabular_flag:
            print(f'\n  Tabular models...')
            tabular = run_tabular(stations, holdout, with_weather)
            gc.collect()
            if 'rf' in rf_xgb_filter:
                results['Random Forest'] = tabular['Random Forest']
            if 'xgb' in rf_xgb_filter:
                results['XGBoost'] = tabular['XGBoost']

        if 'lstm' in requested:
            print(f'\n  LSTM...')
            y_l, p_l, k_l = run_lstm(stations, holdout, with_weather)
            gc.collect()
            results['LSTM'] = (y_l, p_l, k_l)

        if 'tft' in requested:
            print(f'\n  TFT...')
            y_t, p_t, k_t = run_tft(stations, holdout, with_weather)
            gc.collect()
            results['TFT'] = (y_t, p_t, k_t)

        for name, (yt, yp, kk) in results.items():
            m = _regression_metrics(yt, yp)
            row = {'Model': name, 'Weather': tag,
                   'n_test_samples': int(len(yt)),
                   'n_test_stations': int(len(np.unique(kk))),
                   **{k: round(v, 4) for k, v in m.items()}}
            all_metrics.append(row)
            print(f'    {name:<14}  MAE={m["MAE"]:.4f}  R²={m["R2"]:.4f}  '
                  f'Recall={m["Recall"]:.3f}  ROC AUC={m["ROC AUC"]:.3f}')

            ps = _per_station(yt, yp, kk)
            ps.insert(0, 'Weather', tag)
            ps.insert(0, 'Model', name)
            all_per_station.append(ps)

    metrics_df = pd.DataFrame(all_metrics)
    per_station_df = pd.concat(all_per_station, ignore_index=True)

    metrics_path = os.path.join(out_dir, 'metrics.csv')
    per_station_path = os.path.join(out_dir, 'per_station_metrics.csv')

    if subset_run and os.path.exists(metrics_path):
        # Merge: drop existing rows for the (Model × Weather) pairs we just ran,
        # then append the new ones. Keeps numbers from models we didn't rerun.
        existing = pd.read_csv(metrics_path)
        ran_pairs = metrics_df[['Model', 'Weather']].apply(tuple, axis=1).tolist()
        keep = ~existing[['Model', 'Weather']].apply(tuple, axis=1).isin(ran_pairs)
        metrics_df = pd.concat([existing[keep], metrics_df], ignore_index=True)

        if os.path.exists(per_station_path):
            existing_ps = pd.read_csv(per_station_path)
            keep_ps = ~existing_ps[['Model', 'Weather']].apply(tuple, axis=1).isin(ran_pairs)
            per_station_df = pd.concat([existing_ps[keep_ps], per_station_df], ignore_index=True)

    metrics_df.to_csv(metrics_path, index=False)
    per_station_df.to_csv(per_station_path, index=False)

    # Summary plot: MAE per model, with vs without weather.
    fig, ax = plt.subplots(figsize=(10, 5))
    models = ['Random Forest', 'XGBoost', 'LSTM', 'TFT']
    x = np.arange(len(models))
    w = 0.38
    nw = metrics_df[metrics_df['Weather'] == 'no weather'].set_index('Model').reindex(models)
    ww = metrics_df[metrics_df['Weather'] == 'with weather'].set_index('Model').reindex(models)
    ax.bar(x - w/2, nw['MAE'], w, label='No weather', color='steelblue')
    ax.bar(x + w/2, ww['MAE'], w, label='With weather', color='darkorange')
    for i, (a, b) in enumerate(zip(nw['MAE'], ww['MAE'])):
        ax.text(i - w/2, a, f'{a:.4f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + w/2, b, f'{b:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('MAE on held-out stations (m³/m³)')
    ax.set_title(f'Station holdout — {len(holdout)} unseen stations, t+{HORIZON}h')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mae_by_model.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\n{"="*60}\nSaved to images/{run_name}/\n{"="*60}')
    print('\nFinal metrics:')
    print(metrics_df.to_string(index=False))


if __name__ == '__main__':
    main()
