THRESHOLD = 0.30

HORIZON = 24
WINDOW  = 120
LAGS    = (24, 48, 72, 96, 120)

WEATHER_COLS = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'PRECTOTCORR', 'WS2M', 'ALLSKY_SFC_SW_DWN']

# Derived from WEATHER_COLS — computed in prepare.py, not fetched from API
DERIVED_WEATHER_COLS = ['ET0']

FEATURE_COLS_BASE = [
    'sm_value',
    'sm_lag_24h', 'sm_lag_48h', 'sm_lag_72h', 'sm_lag_96h', 'sm_lag_120h',
    'sm_rolling_mean_72h', 'sm_rolling_std_72h',
    'sm_rolling_mean_168h', 'sm_rolling_std_168h',
    'hour', 'month', 'dayofyear',
    'latitude', 'longitude', 'elevation_m', 'depth_m',
]

XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=0, tree_method='hist', nthread=-1, verbosity=0,
)

XGB_PARAM_GRID = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [3, 5, 6],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

RF_PARAM_GRID = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [5, 10, None],
    'max_features':     ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4],
}
