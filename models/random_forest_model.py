import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, recall_score, precision_score, f1_score

from prepare import load_all_stations, build_xgboost_dataset

THRESHOLD = 0.30

stations = load_all_stations()
data     = build_xgboost_dataset(stations)

feature_cols = [
    'sm_value', 'sm_lag_24h', 'sm_lag_48h', 'sm_lag_72h',
    'hour', 'month', 'dayofyear',
    'latitude', 'longitude', 'elevation_m', 'depth_m'
]

X      = data[feature_cols].values
y      = data['target'].values
groups = data['station'].values

# subsample to 100k rows to keep grid search tractable
rng  = np.random.default_rng(0)
idx  = rng.choice(len(X), size=min(100_000, len(X)), replace=False)
idx.sort()
X, y, groups = X[idx], y[idx], groups[idx]

split   = int(len(X) * 0.8)
X_train = X[:split]
X_test  = X[split:]
y_train = y[:split]
y_test  = y[split:]
g_train = groups[:split]

param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [5, 10],
    'max_features': ['sqrt'],
}

gkf       = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_train, y_train, g_train))

search = GridSearchCV(
    RandomForestRegressor(random_state=0, n_jobs=-1),
    param_grid,
    cv=cv_splits,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)

best   = search.best_estimator_
y_pred = best.predict(X_test)

print(f"\nBest params: {search.best_params_}")
print(f"MAE:         {mean_absolute_error(y_test, y_pred):.4f} m³/m³")
print(f"R²:          {r2_score(y_test, y_pred):.4f}")

urgent_actual = (y_test < THRESHOLD).astype(int)
urgent_pred   = (y_pred < THRESHOLD).astype(int)
print(f"Recall:      {recall_score(urgent_actual, urgent_pred):.3f}")
print(f"Precision:   {precision_score(urgent_actual, urgent_pred):.3f}")
print(f"F1:          {f1_score(urgent_actual, urgent_pred):.3f}")
