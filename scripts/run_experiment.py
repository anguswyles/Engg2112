"""
Test 2 — weather-augmented models at t+72h horizon.
Compares against Test 1 (no weather, t+24h) and persistence baselines.
Saves all plots and CSVs to images/test 2/

Run from project root:  python scripts/run_experiment.py
"""

import os
import sys
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

from prepare import load_all_stations, build_xgboost_dataset, build_lstm_dataset, get_feature_cols
from config import THRESHOLD, XGB_PARAMS

HORIZON    = 72
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images', 'test 2')
COLORS     = ['steelblue', 'darkorange', 'green']

os.makedirs(IMAGES_DIR, exist_ok=True)

# Test 1 reference metrics (no weather, t+24h) for delta comparison
TEST1 = {
    'Random Forest': {'MAE': 0.0090, 'RMSE': 0.0175, 'R2': 0.9595},
    'XGBoost':       {'MAE': 0.0081, 'RMSE': 0.0171, 'R2': 0.9612},
    'LSTM':          {'MAE': 0.0096, 'RMSE': 0.0235, 'R2': 0.9308},
    'Persistence':   {'MAE': 0.0120, 'R2':   0.9261},
}


def classification_metrics(y_true, y_pred):
    actual = (y_true < THRESHOLD).astype(int)
    pred   = (y_pred < THRESHOLD).astype(int)
    score  = 1 - y_pred
    fpr, tpr, _   = roc_curve(actual, score)
    prec, rec, _  = precision_recall_curve(actual, score)
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


# ── load data ─────────────────────────────────────────────────────────────────
print('Loading data with weather features...')
stations     = load_all_stations(with_weather=True)
feature_cols = get_feature_cols(with_weather=True)

data  = build_xgboost_dataset(stations, horizon=HORIZON, with_weather=True)
X     = data[feature_cols].values
y     = data['target'].values
split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
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

# ── train models ──────────────────────────────────────────────────────────────
print('\nTraining Random Forest (t+72h, with weather)...')
rf = RandomForestRegressor(n_estimators=200, max_depth=10, max_features='sqrt',
                           random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
del X_train
gc.collect()
print('  done')

print('Training XGBoost (t+72h, with weather)...')
xgb = XGBRegressor(**XGB_PARAMS)
xgb.fit(X[:split], y[:split])
y_pred_xgb = xgb.predict(X_test)
del X, y
gc.collect()
print('  done')

print('Training LSTM (t+72h)...')
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

y_persist = data['sm_lag_72h'].values[split:]

models = {
    'Random Forest': (y_test,     y_pred_rf),
    'XGBoost':       (y_test,     y_pred_xgb),
    'LSTM':          (y_test_seq, y_pred_lstm),
}

# ── metrics table ─────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('METRICS — Test 2 (weather features, t+72h)')
print('='*65)

rows = []
for name, (y_true, y_pred) in models.items():
    cm   = classification_metrics(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    rows.append({
        'Model': name, 'Horizon': 't+72h', 'Features': 'weather+lag',
        'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R²': round(r2, 4),
        'Recall': round(cm['recall'], 3), 'ROC AUC': round(cm['roc_auc'], 3),
    })
    t1        = TEST1.get(name, {})
    mae_delta = f"({mae - t1['MAE']:+.4f})" if t1.get('MAE') else ''
    r2_delta  = f"({r2  - t1['R2']:+.4f})"  if t1.get('R2')  else ''
    print(f"  {name:<20} MAE={mae:.4f} {mae_delta:<12} R²={r2:.4f} {r2_delta:<12} Recall={cm['recall']:.3f}  ROC AUC={cm['roc_auc']:.3f}")

p_mae = mean_absolute_error(y_test, y_persist)
p_r2  = r2_score(y_test, y_persist)
print(f"  {'Persistence':<20} MAE={p_mae:.4f} ({p_mae - TEST1['Persistence']['MAE']:+.4f} vs t+24h)  R²={p_r2:.4f} ({p_r2 - TEST1['Persistence']['R2']:+.4f} vs t+24h)")

pd.DataFrame(rows).to_csv(os.path.join(IMAGES_DIR, 'metrics.csv'), index=False)
print(f'\nMetrics saved to images/test 2/metrics.csv')

# ── plots ─────────────────────────────────────────────────────────────────────
print('\nGenerating plots...')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), COLORS):
    cm = classification_metrics(y_true, y_pred)
    ax.plot(cm['fpr'], cm['tpr'], color=color, lw=2, label=f"AUC={cm['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {name}')
    ax.legend()
plt.suptitle('ROC Curves — Test 2 (weather features, t+72h)')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved roc_curves.png')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), COLORS):
    cm = classification_metrics(y_true, y_pred)
    ax.plot(cm['rec'], cm['prec'], color=color, lw=2, label=f"AUC={cm['pr_auc']:.3f}")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall — {name}')
    ax.legend()
plt.suptitle('Precision-Recall Curves — Test 2 (weather features, t+72h)')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'precision_recall.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved precision_recall.png')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, (y_true, y_pred)) in zip(axes, models.items()):
    cm = classification_metrics(y_true, y_pred)
    ConfusionMatrixDisplay(cm['cm'], display_labels=['Above', 'Below']).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.suptitle(f'Confusion Matrices — Test 2 (threshold={THRESHOLD})')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved confusion_matrices.png')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), COLORS):
    s = np.random.choice(len(y_true), size=min(5000, len(y_true)), replace=False)
    ax.scatter(y_true[s], y_pred[s], alpha=0.2, s=5, color=color)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', lw=0.8)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(name)
plt.suptitle('Predicted vs Actual — Test 2 (weather features, t+72h)')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'predicted_vs_actual.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved predicted_vs_actual.png')

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), COLORS):
    residuals = y_true - y_pred
    ax.hist(residuals, bins=60, color=color, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Residual (actual − predicted)')
    ax.set_ylabel('Count')
    ax.set_title(f'{name}  (mean={residuals.mean():.4f})')
plt.suptitle('Residual Distributions — Test 2 (weather features, t+72h)')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'residuals.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved residuals.png')

feat_imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 8))
feat_imp.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('XGBoost Feature Importance (weather features, t+72h)')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved feature_importance.png')

# ── horizon comparison ────────────────────────────────────────────────────────
print('\nRunning horizon comparison (this takes a few minutes)...')

horizons    = [24, 48, 72, 120, 168]
results     = []
stations_nw = load_all_stations(with_weather=False)
stations_w  = load_all_stations(with_weather=True)
fc_nw       = get_feature_cols(with_weather=False)
fc_w        = get_feature_cols(with_weather=True)

for h in horizons:
    data_nw = build_xgboost_dataset(stations_nw, horizon=h)
    X_nw    = data_nw[fc_nw].values
    y_nw    = data_nw['target'].values
    sp_nw   = int(len(X_nw) * 0.8)
    m_nw    = XGBRegressor(**XGB_PARAMS)
    m_nw.fit(X_nw[:sp_nw], y_nw[:sp_nw])
    pred_nw = m_nw.predict(X_nw[sp_nw:])
    persist = data_nw['sm_lag_24h'].values[sp_nw:]

    data_w = build_xgboost_dataset(stations_w, horizon=h, with_weather=True)
    X_w    = data_w[fc_w].values
    y_w    = data_w['target'].values
    sp_w   = int(len(X_w) * 0.8)
    m_w    = XGBRegressor(**XGB_PARAMS)
    m_w.fit(X_w[:sp_w], y_w[:sp_w])
    pred_w = m_w.predict(X_w[sp_w:])

    results.append({
        'horizon':     h,
        'persist_mae': round(mean_absolute_error(y_nw[sp_nw:], persist), 4),
        'persist_r2':  round(r2_score(y_nw[sp_nw:], persist), 4),
        'xgb_nw_mae':  round(mean_absolute_error(y_nw[sp_nw:], pred_nw), 4),
        'xgb_nw_r2':   round(r2_score(y_nw[sp_nw:], pred_nw), 4),
        'xgb_w_mae':   round(mean_absolute_error(y_w[sp_w:], pred_w), 4),
        'xgb_w_r2':    round(r2_score(y_w[sp_w:], pred_w), 4),
    })
    print(f'  t+{h}h done')

hr = pd.DataFrame(results)
hr.to_csv(os.path.join(IMAGES_DIR, 'horizon_comparison.csv'), index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(hr['horizon'], hr['persist_mae'],  'k--',          marker='o', label='Persistence')
ax1.plot(hr['horizon'], hr['xgb_nw_mae'],   color='steelblue',   marker='o', label='XGBoost (no weather)')
ax1.plot(hr['horizon'], hr['xgb_w_mae'],    color='darkorange',  marker='o', label='XGBoost (with weather)')
ax1.set_xlabel('Prediction Horizon (hours)')
ax1.set_ylabel('MAE (m³/m³)')
ax1.set_title('MAE vs Horizon')
ax1.legend()

ax2.plot(hr['horizon'], hr['persist_r2'],  'k--',          marker='o', label='Persistence')
ax2.plot(hr['horizon'], hr['xgb_nw_r2'],   color='steelblue',   marker='o', label='XGBoost (no weather)')
ax2.plot(hr['horizon'], hr['xgb_w_r2'],    color='darkorange',  marker='o', label='XGBoost (with weather)')
ax2.set_xlabel('Prediction Horizon (hours)')
ax2.set_ylabel('R²')
ax2.set_title('R² vs Horizon')
ax2.legend()

plt.suptitle('Model Performance vs Prediction Horizon — with and without weather features')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'horizon_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  saved horizon_comparison.png')

print(f'\nAll outputs saved to images/test 2/')
print('\nHorizon comparison table:')
print(hr.to_string(index=False))
