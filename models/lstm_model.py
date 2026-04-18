import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, recall_score, precision_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers

from prepare import load_all_stations, build_lstm_dataset

THRESHOLD = 0.30

stations  = load_all_stations()
X, y      = build_lstm_dataset(stations)

# subsample to 100k sequences to keep training time reasonable
rng  = np.random.default_rng(0)
idx  = rng.choice(len(X), size=min(100_000, len(X)), replace=False)
idx.sort()
X, y = X[idx], y[idx]

X_mean, X_std = X.mean(), X.std()
X_norm        = (X - X_mean) / X_std

split    = int(len(X) * 0.8)
X_train  = X_norm[:split].reshape(-1, X.shape[1], 1)
X_test   = X_norm[split:].reshape(-1, X.shape[1], 1)
y_train  = y[:split]
y_test   = y[split:]

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

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
    epochs=20,
    batch_size=512,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

y_pred = model.predict(X_test).flatten()

print(f"\nMAE:       {mean_absolute_error(y_test, y_pred):.4f} m³/m³")
print(f"R²:        {r2_score(y_test, y_pred):.4f}")

urgent_actual = (y_test < THRESHOLD).astype(int)
urgent_pred   = (y_pred < THRESHOLD).astype(int)
print(f"Recall:    {recall_score(urgent_actual, urgent_pred):.3f}")
print(f"Precision: {precision_score(urgent_actual, urgent_pred):.3f}")
print(f"F1:        {f1_score(urgent_actual, urgent_pred):.3f}")
