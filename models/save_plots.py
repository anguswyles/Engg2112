import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(__file__))

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
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras import layers
from prepare import load_all_stations, build_xgboost_dataset, build_lstm_dataset

THRESHOLD  = 0.30
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images', 'test 1')
os.makedirs(IMAGES_DIR, exist_ok=True)

print('Loading data...')
stations = load_all_stations()

data = build_xgboost_dataset(stations)
feature_cols = [
    'sm_value', 'sm_lag_24h', 'sm_lag_48h', 'sm_lag_72h',
    'hour', 'month', 'dayofyear',
    'latitude', 'longitude', 'elevation_m', 'depth_m'
]
X_tab = data[feature_cols].values
y_tab = data['target'].values
split = int(len(X_tab) * 0.8)
X_train_tab, X_test_tab = X_tab[:split], X_tab[split:]
y_train_tab, y_test_tab = y_tab[:split], y_tab[split:]

print('Training Random Forest...')
rf = RandomForestRegressor(n_estimators=200, max_depth=10, max_features='sqrt', random_state=0, n_jobs=-1)
rf.fit(X_train_tab, y_train_tab)
y_pred_rf = rf.predict(X_test_tab)
del X_train_tab
gc.collect()
print('done')

print('Training XGBoost...')
xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0, tree_method='hist', nthread=-1)
xgb.fit(X_train_tab if False else X_tab[:split], y_train_tab if False else y_tab[:split])
xgb.fit(X_tab[:split], y_tab[:split])
y_pred_xgb = xgb.predict(X_test_tab)
del X_tab, y_tab
gc.collect()
print('done')

print('Training LSTM...')
X_seq, y_seq = build_lstm_dataset(stations)
rng = np.random.default_rng(0)
idx = rng.choice(len(X_seq), size=min(100_000, len(X_seq)), replace=False)
idx.sort()
X_seq, y_seq = X_seq[idx], y_seq[idx]
X_mean, X_std = X_seq.mean(), X_seq.std()
X_seq_norm = (X_seq - X_mean) / X_std
split_seq = int(len(X_seq) * 0.8)
X_train_seq = X_seq_norm[:split_seq].reshape(-1, X_seq.shape[1], 1)
X_test_seq  = X_seq_norm[split_seq:].reshape(-1, X_seq.shape[1], 1)
y_train_seq, y_test_seq = y_seq[:split_seq], y_seq[split_seq:]

lstm = keras.Sequential([
    keras.Input(shape=(X_train_seq.shape[1], 1)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])
lstm.compile(optimizer='adam', loss='mae')
lstm.fit(X_train_seq, y_train_seq, epochs=10, batch_size=512,
         validation_split=0.1,
         callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
         verbose=1)
y_pred_lstm = lstm.predict(X_test_seq).flatten()
del X_train_seq, X_seq_norm
gc.collect()
print('done')

models = {
    'Random Forest': (y_test_tab, y_pred_rf),
    'XGBoost':       (y_test_tab, y_pred_xgb),
    'LSTM':          (y_test_seq, y_pred_lstm),
}
colors = ['steelblue', 'darkorange', 'green']

rows = []
for name, (y_true, y_pred) in models.items():
    rows.append({
        'Model': name,
        'MAE':   round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE':  round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
        'R²':    round(r2_score(y_true, y_pred), 4),
    })
print('\n', pd.DataFrame(rows).set_index('Model').to_string())

print('\nGenerating plots...')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), colors):
    actual_bin  = (y_true < THRESHOLD).astype(int)
    fpr, tpr, _ = roc_curve(actual_bin, 1 - y_pred)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {name}')
    ax.legend()
plt.suptitle('ROC Curves')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved roc_curves.png')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), colors):
    actual_bin = (y_true < THRESHOLD).astype(int)
    precision, recall, _ = precision_recall_curve(actual_bin, 1 - y_pred)
    ax.plot(recall, precision, color=color, lw=2, label=f'AUC = {auc(recall, precision):.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall — {name}')
    ax.legend()
plt.suptitle('Precision-Recall Curves')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/precision_recall.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved precision_recall.png')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, (y_true, y_pred)) in zip(axes, models.items()):
    actual_bin = (y_true < THRESHOLD).astype(int)
    pred_bin   = (y_pred < THRESHOLD).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(actual_bin, pred_bin), display_labels=['Above', 'Below']).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.suptitle(f'Confusion Matrices (threshold = {THRESHOLD})')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved confusion_matrices.png')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), colors):
    sample = np.random.choice(len(y_true), size=min(5000, len(y_true)), replace=False)
    ax.scatter(y_true[sample], y_pred[sample], alpha=0.2, s=5, color=color)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', lw=0.8)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(name)
plt.suptitle('Predicted vs Actual Soil Moisture (m³/m³)')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved predicted_vs_actual.png')

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, (y_true, y_pred)), color in zip(axes, models.items(), colors):
    residuals = y_true - y_pred
    ax.hist(residuals, bins=60, color=color, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Residual (actual − predicted)')
    ax.set_ylabel('Count')
    ax.set_title(f'{name}  (mean={residuals.mean():.4f})')
plt.suptitle('Residual Distributions')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved residuals.png')

print(f'\nAll images saved to {IMAGES_DIR}')
