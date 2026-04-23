"""
Experiment runner — trains RF, XGBoost, and LSTM and saves plots + metrics.
Each run is saved to a new folder under images/ for progression tracking.

Usage:
    python scripts/run_experiment.py                        # auto-names next test
    python scripts/run_experiment.py "test 3 - added NDVI" # custom name
"""

import os
import sys
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    recall_score, precision_score, f1_score,
)
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras import layers

from prepare import load_all_stations, build_xgboost_dataset, build_lstm_dataset, build_tft_dataset, get_feature_cols, station_temporal_split
from config import THRESHOLD, XGB_PARAMS

HORIZON    = 72
IMAGES_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'images')
COLORS      = ['steelblue', 'darkorange', 'green', 'purple']


def _next_test_name():
    existing = [
        d for d in os.listdir(IMAGES_ROOT)
        if os.path.isdir(os.path.join(IMAGES_ROOT, d)) and d.startswith('test ')
    ]
    nums = []
    for d in existing:
        try:
            nums.append(int(d.split(' ')[1]))
        except (IndexError, ValueError):
            pass
    n = max(nums) + 1 if nums else 1
    return f"test {n}"


def classification_metrics(y_true, y_pred):
    actual = (y_true < THRESHOLD).astype(int)
    pred   = (y_pred < THRESHOLD).astype(int)
    score  = 1 - y_pred
    fpr, tpr, _  = roc_curve(actual, score)
    prec, rec, _ = precision_recall_curve(actual, score)
    return {
        'recall':    recall_score(actual, pred),
        'precision': precision_score(actual, pred),
        'f1':        f1_score(actual, pred),
        'roc_auc':   auc(fpr, tpr),
        'pr_auc':    auc(rec, prec),
        'fpr': fpr, 'tpr': tpr,
        'prec': prec, 'rec': rec,
        'cm':  confusion_matrix(actual, pred),
    }


def load_all_metrics():
    """Load metrics.csv from every previous test folder for comparison."""
    rows = []
    for d in sorted(os.listdir(IMAGES_ROOT)):
        path = os.path.join(IMAGES_ROOT, d, 'metrics.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.insert(0, 'Test', d)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_tft(seq_len, n_static):
    """
    TFT-inspired Keras model.
    Static features are broadcast and concatenated at each timestep so the LSTM
    sees station context at every step. Multi-head self-attention + gated residual
    then captures temporal dependencies before a dense output head.
    """
    seq_in = keras.Input(shape=(seq_len, 1), name='sequence')
    sta_in = keras.Input(shape=(n_static,),  name='static')

    # Static covariate encoder
    s = layers.Dense(16, activation='elu')(sta_in)
    s = layers.Dense(16)(s)

    # Broadcast static embedding across every timestep and concatenate
    s_exp = layers.RepeatVector(seq_len)(s)                    # (batch, seq, 16)
    x = layers.Concatenate(axis=-1)([seq_in, s_exp])           # (batch, seq, 17)

    # Temporal LSTM
    x = layers.LSTM(64, return_sequences=True)(x)

    # Multi-head temporal self-attention
    a = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)

    # Gated residual connection (simplified GLU)
    g = layers.Dense(64, activation='sigmoid')(a)
    x = layers.Add()([layers.Multiply()([g, a]), x])
    x = layers.LayerNormalization()(x)

    # Use the last timestep — carries full sequence history
    x   = layers.Lambda(lambda t: t[:, -1, :])(x)
    x   = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1)(x)

    return keras.Model(inputs=[seq_in, sta_in], outputs=out)


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('name', nargs='?', default=None, help='Experiment name (defaults to next test number)')
args = parser.parse_args()

RUN_NAME   = args.name or _next_test_name()
IMAGES_DIR = os.path.join(IMAGES_ROOT, RUN_NAME)
os.makedirs(IMAGES_DIR, exist_ok=True)

print(f'\n{"="*60}')
print(f'Run: {RUN_NAME}')
print(f'Output: images/{RUN_NAME}/')
print(f'{"="*60}\n')

# ── load data ─────────────────────────────────────────────────────────────────
print('Loading data with weather features...')
stations     = load_all_stations(with_weather=True)
feature_cols = get_feature_cols(with_weather=True, horizon=HORIZON)

data  = build_xgboost_dataset(stations, horizon=HORIZON, with_weather=True)
X     = data[feature_cols].values
y     = data['target'].values
train_mask, test_mask = station_temporal_split(data)
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
print(f'  Tabular dataset: {X.shape}  ({len(feature_cols)} features)')

X_seq, y_seq = build_lstm_dataset(stations, horizon=HORIZON)
rng  = np.random.default_rng(0)
idx  = rng.choice(len(X_seq), size=min(100_000, len(X_seq)), replace=False)
idx.sort()
X_seq, y_seq  = X_seq[idx], y_seq[idx]
X_mean, X_std = X_seq.mean(), X_seq.std()
X_norm = (X_seq - X_mean) / X_std
sp     = int(len(X_seq) * 0.8)
X_train_seq = X_norm[:sp].reshape(-1, X_seq.shape[1], 1)
X_test_seq  = X_norm[sp:].reshape(-1, X_seq.shape[1], 1)
y_train_seq, y_test_seq = y_seq[:sp], y_seq[sp:]
print(f'  Sequence dataset: {X_seq.shape}')

X_tft, X_static, y_tft = build_tft_dataset(stations, horizon=HORIZON)
rng2  = np.random.default_rng(1)
idx2  = rng2.choice(len(X_tft), size=min(100_000, len(X_tft)), replace=False)
idx2.sort()
X_tft, X_static, y_tft = X_tft[idx2], X_static[idx2], y_tft[idx2]
X_tft_mean, X_tft_std  = X_tft.mean(), X_tft.std()
X_tft_norm             = (X_tft - X_tft_mean) / X_tft_std
X_static_norm          = (X_static - X_static.mean(axis=0)) / (X_static.std(axis=0) + 1e-8)
sp3 = int(len(X_tft) * 0.8)
X_train_tft    = X_tft_norm[:sp3].reshape(-1, X_tft.shape[1], 1)
X_test_tft     = X_tft_norm[sp3:].reshape(-1, X_tft.shape[1], 1)
X_train_static = X_static_norm[:sp3]
X_test_static  = X_static_norm[sp3:]
y_train_tft, y_test_tft = y_tft[:sp3], y_tft[sp3:]
print(f'  TFT dataset:      {X_tft.shape}')

# ── train models ──────────────────────────────────────────────────────────────
print(f'\nTraining Random Forest (t+{HORIZON}h, with weather)...')
rf = RandomForestRegressor(n_estimators=200, max_depth=10, max_features='sqrt',
                           random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
del X_train
gc.collect()
print('  done')

print(f'Training XGBoost (t+{HORIZON}h, with weather)...')
xgb = XGBRegressor(**XGB_PARAMS)
xgb.fit(X[train_mask], y[train_mask])
y_pred_xgb = xgb.predict(X_test)
del X, y
gc.collect()
print('  done')

print(f'Training LSTM (t+{HORIZON}h)...')
lstm = keras.Sequential([
    keras.Input(shape=(X_train_seq.shape[1], 1)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])
lstm.compile(optimizer='adam', loss='mae')
lstm.fit(X_train_seq, y_train_seq, epochs=20, batch_size=512, validation_split=0.1,
         callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                   restore_best_weights=True)],
         verbose=1)
y_pred_lstm = lstm.predict(X_test_seq).flatten()
del X_train_seq
gc.collect()
print('  done')

print(f'Training TFT (t+{HORIZON}h)...')
tft = _build_tft(X_train_tft.shape[1], X_train_static.shape[1])
tft.compile(optimizer='adam', loss='mae')
tft.fit(
    [X_train_tft, X_train_static], y_train_tft,
    epochs=20, batch_size=512, validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                              restore_best_weights=True)],
    verbose=1,
)
y_pred_tft = tft.predict([X_test_tft, X_test_static]).flatten()
del X_train_tft, X_train_static
gc.collect()
print('  done')

y_persist = data['sm_lag_72h'].values[test_mask]

models = {
    'Random Forest': (y_test,     y_pred_rf),
    'XGBoost':       (y_test,     y_pred_xgb),
    'LSTM':          (y_test_seq, y_pred_lstm),
    'TFT':           (y_test_tft, y_pred_tft),
}

# ── metrics ───────────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print(f'METRICS — {RUN_NAME}')
print(f'{"="*65}')

rows = []
for name, (y_true, y_pred) in models.items():
    cm   = classification_metrics(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    rows.append({
        'Model': name, 'Horizon': f't+{HORIZON}h', 'Features': 'weather+lag',
        'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R²': round(r2, 4),
        'Recall': round(cm['recall'], 3), 'ROC AUC': round(cm['roc_auc'], 3),
    })
    print(f"  {name:<20} MAE={mae:.4f}  R²={r2:.4f}  Recall={cm['recall']:.3f}  ROC AUC={cm['roc_auc']:.3f}")

p_mae = mean_absolute_error(y_test, y_persist)
p_r2  = r2_score(y_test, y_persist)
print(f"  {'Persistence':<20} MAE={p_mae:.4f}  R²={p_r2:.4f}")

pd.DataFrame(rows).to_csv(os.path.join(IMAGES_DIR, 'metrics.csv'), index=False)

# ── progression comparison ────────────────────────────────────────────────────
all_metrics = load_all_metrics()
if not all_metrics.empty:
    print(f'\n{"="*65}')
    print('PROGRESSION ACROSS ALL TESTS')
    print(f'{"="*65}')
    print(all_metrics[['Test', 'Model', 'MAE', 'R²', 'Recall', 'ROC AUC']].to_string(index=False))

# ── plots ─────────────────────────────────────────────────────────────────────
print('\nGenerating plots...')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), COLORS):
    cm = classification_metrics(y_true, y_pred)
    ax.plot(cm['fpr'], cm['tpr'], color=color, lw=2, label=f"AUC={cm['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {name}')
    ax.legend()
plt.suptitle(f'ROC Curves — {RUN_NAME}')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved roc_curves.png')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), COLORS):
    cm = classification_metrics(y_true, y_pred)
    ax.plot(cm['rec'], cm['prec'], color=color, lw=2, label=f"AUC={cm['pr_auc']:.3f}")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall — {name}')
    ax.legend()
plt.suptitle(f'Precision-Recall Curves — {RUN_NAME}')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'precision_recall.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved precision_recall.png')

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, (y_true, y_pred)) in zip(axes, models.items()):
    cm = classification_metrics(y_true, y_pred)
    ConfusionMatrixDisplay(cm['cm'], display_labels=['Above', 'Below']).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.suptitle(f'Confusion Matrices — {RUN_NAME} (threshold={THRESHOLD})')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved confusion_matrices.png')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), COLORS):
    s = np.random.choice(len(y_true), size=min(5000, len(y_true)), replace=False)
    ax.scatter(y_true[s], y_pred[s], alpha=0.2, s=5, color=color)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', lw=0.8)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(name)
plt.suptitle(f'Predicted vs Actual — {RUN_NAME}')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'predicted_vs_actual.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved predicted_vs_actual.png')

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), COLORS):
    residuals = y_true - y_pred
    ax.hist(residuals, bins=60, color=color, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Residual (actual − predicted)')
    ax.set_ylabel('Count')
    ax.set_title(f'{name}  (mean={residuals.mean():.4f})')
plt.suptitle(f'Residual Distributions — {RUN_NAME}')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'residuals.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved residuals.png')

feat_imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 8))
feat_imp.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Importance')
ax.set_title(f'XGBoost Feature Importance — {RUN_NAME}')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved feature_importance.png')

# ── horizon comparison ────────────────────────────────────────────────────────
print('\nRunning horizon comparison (this takes a few minutes)...')

horizons    = [24, 48, 72, 96, 120, 168]
results     = []
stations_nw = load_all_stations(with_weather=False)
stations_w  = load_all_stations(with_weather=True)

for h in horizons:
    # Feature cols are horizon-specific: weather version adds t+24h…t+h columns
    fc_nw = get_feature_cols(with_weather=False, horizon=h)
    fc_w  = get_feature_cols(with_weather=True,  horizon=h)

    data_nw             = build_xgboost_dataset(stations_nw, horizon=h)
    X_nw                = data_nw[fc_nw].values
    y_nw                = data_nw['target'].values
    tr_nw, te_nw        = station_temporal_split(data_nw)
    m_nw                = XGBRegressor(**XGB_PARAMS)
    m_nw.fit(X_nw[tr_nw], y_nw[tr_nw])
    pred_nw             = m_nw.predict(X_nw[te_nw])
    persist             = data_nw['sm_lag_24h'].values[te_nw]

    data_w              = build_xgboost_dataset(stations_w, horizon=h, with_weather=True)
    X_w                 = data_w[fc_w].values
    y_w                 = data_w['target'].values
    tr_w, te_w          = station_temporal_split(data_w)
    m_w                 = XGBRegressor(**XGB_PARAMS)
    m_w.fit(X_w[tr_w], y_w[tr_w])
    pred_w              = m_w.predict(X_w[te_w])

    cm_nw = classification_metrics(y_nw[te_nw], pred_nw)
    cm_w  = classification_metrics(y_w[te_w],   pred_w)

    results.append({
        'horizon':        h,
        'persist_mae':    round(mean_absolute_error(y_nw[te_nw], persist), 4),
        'persist_r2':     round(r2_score(y_nw[te_nw], persist), 4),
        'xgb_nw_mae':     round(mean_absolute_error(y_nw[te_nw], pred_nw), 4),
        'xgb_nw_r2':      round(r2_score(y_nw[te_nw], pred_nw), 4),
        'xgb_nw_recall':  round(cm_nw['recall'], 4),
        'xgb_nw_roc_auc': round(cm_nw['roc_auc'], 4),
        'xgb_w_mae':      round(mean_absolute_error(y_w[te_w], pred_w), 4),
        'xgb_w_r2':       round(r2_score(y_w[te_w], pred_w), 4),
        'xgb_w_recall':   round(cm_w['recall'], 4),
        'xgb_w_roc_auc':  round(cm_w['roc_auc'], 4),
    })
    print(f'  t+{h}h done')

hr = pd.DataFrame(results)
hr['mae_gap']     = hr['xgb_nw_mae']     - hr['xgb_w_mae']
hr['r2_gap']      = hr['xgb_w_r2']       - hr['xgb_nw_r2']
hr['recall_gap']  = hr['xgb_w_recall']   - hr['xgb_nw_recall']
hr['roc_auc_gap'] = hr['xgb_w_roc_auc']  - hr['xgb_nw_roc_auc']
hr.to_csv(os.path.join(IMAGES_DIR, 'horizon_comparison.csv'), index=False)

# ── performance vs horizon ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(hr['horizon'], hr['persist_mae'],  'k--',              marker='o', label='Persistence')
ax1.plot(hr['horizon'], hr['xgb_nw_mae'],   color='steelblue',  marker='o', label='XGBoost (no weather)')
ax1.plot(hr['horizon'], hr['xgb_w_mae'],    color='darkorange', marker='o', label='XGBoost (with weather)')
ax1.set_xlabel('Prediction Horizon (hours)')
ax1.set_ylabel('MAE (m³/m³)')
ax1.set_title('MAE vs Horizon')
ax1.legend()

ax2.plot(hr['horizon'], hr['persist_r2'],  'k--',              marker='o', label='Persistence')
ax2.plot(hr['horizon'], hr['xgb_nw_r2'],   color='steelblue',  marker='o', label='XGBoost (no weather)')
ax2.plot(hr['horizon'], hr['xgb_w_r2'],    color='darkorange', marker='o', label='XGBoost (with weather)')
ax2.set_xlabel('Prediction Horizon (hours)')
ax2.set_ylabel('R²')
ax2.set_title('R² vs Horizon')
ax2.legend()

plt.suptitle(f'Model Performance vs Prediction Horizon — {RUN_NAME}')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'horizon_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved horizon_comparison.png')

# ── weather impact gap ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def _bar(ax, x, y, color, ylabel, title):
    ax.bar(x, y, color=color, width=10)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for xi, v in zip(x, y):
        ax.text(xi, v + abs(y.max()) * 0.02, f'{v:+.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)

_bar(axes[0, 0], hr['horizon'], hr['mae_gap'],
     'darkorange', 'MAE reduction (m³/m³)', 'Weather Impact on MAE\n(positive = weather helps)')
_bar(axes[0, 1], hr['horizon'], hr['r2_gap'],
     'steelblue',  'R² gain', 'Weather Impact on R²\n(positive = weather helps)')
_bar(axes[1, 0], hr['horizon'], hr['recall_gap'],
     'green',      'Recall gain', 'Weather Impact on Recall\n(catching below-threshold events)')
_bar(axes[1, 1], hr['horizon'], hr['roc_auc_gap'],
     'purple',     'ROC AUC gain', 'Weather Impact on ROC AUC')

plt.suptitle(f'Benefit of Weather Features Across Horizons — {RUN_NAME}')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'weather_impact.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved weather_impact.png')

# ── recall vs horizon ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(hr['horizon'], hr['xgb_nw_recall'],  color='steelblue',  marker='o', label='XGBoost (no weather)')
ax1.plot(hr['horizon'], hr['xgb_w_recall'],   color='darkorange', marker='o', label='XGBoost (with weather)')
ax1.set_xlabel('Prediction Horizon (hours)')
ax1.set_ylabel('Recall')
ax1.set_title('Recall vs Horizon')
ax1.legend()

ax2.plot(hr['horizon'], hr['xgb_nw_roc_auc'], color='steelblue',  marker='o', label='XGBoost (no weather)')
ax2.plot(hr['horizon'], hr['xgb_w_roc_auc'],  color='darkorange', marker='o', label='XGBoost (with weather)')
ax2.set_xlabel('Prediction Horizon (hours)')
ax2.set_ylabel('ROC AUC')
ax2.set_title('ROC AUC vs Horizon')
ax2.legend()

plt.suptitle(f'Classification Performance vs Horizon — {RUN_NAME}')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'recall_vs_horizon.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved recall_vs_horizon.png')

print(f'\nAll outputs saved to images/{RUN_NAME}/')
print('\nHorizon comparison table:')
print(hr.to_string(index=False))
