THRESHOLD = 0.30

HORIZON = 24
WINDOW  = 72
LAGS    = (24, 48, 72)

WEATHER_COLS = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'PRECTOTCORR', 'WS2M', 'ALLSKY_SFC_SW_DWN']

FEATURE_COLS_BASE = [
    'sm_value', 'sm_lag_24h', 'sm_lag_48h', 'sm_lag_72h',
    'hour', 'month', 'dayofyear',
    'latitude', 'longitude', 'elevation_m', 'depth_m',
]

XGB_PARAMS = dict(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    random_state=0, tree_method='hist', nthread=-1, verbosity=0,
)

XGB_PARAM_GRID = {
    'n_estimators':  [100, 300],
    'max_depth':     [3, 6],
    'learning_rate': [0.05, 0.1],
}

RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth':    [5, 10],
    'max_features': ['sqrt'],
}
