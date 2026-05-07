import os
import sys
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from config import LAGS, HORIZON, WINDOW, WEATHER_COLS, DERIVED_WEATHER_COLS, FEATURE_COLS_BASE

DATA_DIR    = os.path.join(_here, '..', 'data', 'soil_data', 'ismn_data')
WEATHER_DIR = os.path.join(_here, '..', 'data', 'weather_data', 'stations')


def load_all_stations(with_weather=False):
    stations = {}
    for network in os.listdir(DATA_DIR):
        net_path = os.path.join(DATA_DIR, network)
        if not os.path.isdir(net_path):
            continue
        for f in os.listdir(net_path):
            if not f.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(net_path, f), parse_dates=['datetime_utc'])
            meta = {
                'latitude':    df['latitude'].iloc[0],
                'longitude':   df['longitude'].iloc[0],
                'elevation_m': df['elevation_m'].iloc[0],
                'depth_m':     df['depth_from_m'].iloc[0],
                'network':     network,
            }
            df = df[df['ismn_flag_good']][['datetime_utc', 'sm_value']].copy()
            df = df.set_index('datetime_utc').sort_index()
            df.index = df.index.tz_convert(None)
            df = df[~df.index.duplicated(keep='first')]
            df = df.resample('1h').interpolate(method='time', limit=6)

            if with_weather:
                station_name = f[:-4]
                weather_path = os.path.join(WEATHER_DIR, network, f"{network}_{station_name}_weather.csv")
                if os.path.exists(weather_path):
                    w = pd.read_csv(weather_path, index_col='date', parse_dates=True)
                    w.index = pd.to_datetime(w.index).normalize()
                    df['date'] = df.index.normalize()
                    df = df.merge(w[WEATHER_COLS], left_on='date', right_index=True, how='left')
                    df = df.drop(columns='date')
                    # Hargreaves-Samani reference evapotranspiration (mm/day)
                    trange = (df['T2M_MAX'] - df['T2M_MIN']).clip(lower=0)
                    df['ET0'] = (0.0023 * (df['ALLSKY_SFC_SW_DWN'] / 0.408)
                                 * (df['T2M'] + 17.8) * np.sqrt(trange))

            stations[f"{network}/{f[:-4]}"] = (df, meta)
    return stations


def temporal_split_by_station(stations_arr, train_frac=0.8):
    """
    Per-station chronological split primitive. Given an array of station
    identifiers (one per row/sample, in time order within each station),
    return boolean train/test masks of the same length where the first
    train_frac of each station's samples are train and the rest are test.
    """
    n          = len(stations_arr)
    train_mask = np.zeros(n, dtype=bool)
    positions  = np.arange(n)
    stations_arr = np.asarray(stations_arr)
    for station in np.unique(stations_arr):
        pos = positions[stations_arr == station]
        sp  = int(len(pos) * train_frac)
        train_mask[pos[:sp]] = True
    return train_mask, ~train_mask


def station_temporal_split(data, train_frac=0.8):
    """
    Split each station's rows independently at train_frac by chronological
    order, rather than cutting the globally-sorted dataframe in one place.

    Because build_xgboost_dataset ends with sort_index(), each station's rows
    already appear in time order within the combined frame, so the first
    train_frac positions for each station are always the earliest readings.

    Returns boolean arrays (train_mask, test_mask) of length len(data).
    """
    return temporal_split_by_station(data['station'].values, train_frac)


def get_feature_cols(with_weather=False, horizon=None):
    """
    Return the feature column list for the tabular (XGBoost/RF) dataset.

    When with_weather=True the current-time weather columns are included, plus
    one set of forward-shifted weather columns per 24-hour step up to the
    prediction horizon (e.g. horizon=72 → t+24h and t+48h and t+72h windows).
    This mirrors treating observed future weather as a stand-in for a perfect
    weather forecast over the prediction window.
    """
    h            = horizon or HORIZON
    cols         = list(FEATURE_COLS_BASE)
    all_wx_cols  = list(WEATHER_COLS) + list(DERIVED_WEATHER_COLS)
    if with_weather:
        cols += all_wx_cols
        for step in range(24, h + 1, 24):
            cols += [f'{c}_t+{step}h' for c in all_wx_cols]
    return cols


def build_xgboost_dataset(stations, horizon=None, with_weather=False):
    h      = horizon or HORIZON
    frames = []
    for key, (df, meta) in stations.items():
        if len(df) < WINDOW + h + 10:
            continue
        feat = df.copy()
        feat['target'] = feat['sm_value'].shift(-h)
        for lag in LAGS:
            feat[f'sm_lag_{lag}h'] = feat['sm_value'].shift(lag)
        for w in (72, 168):
            feat[f'sm_rolling_mean_{w}h'] = feat['sm_value'].shift(1).rolling(w).mean()
            feat[f'sm_rolling_std_{w}h']  = feat['sm_value'].shift(1).rolling(w).std()
        feat['hour']      = feat.index.hour
        feat['month']     = feat.index.month
        feat['dayofyear'] = feat.index.dayofyear
        # Forward weather windows: give the model observed weather for each
        # 24-hour day between t and t+horizon, simulating a perfect forecast.
        if with_weather:
            for step in range(24, h + 1, 24):
                for col in list(WEATHER_COLS) + list(DERIVED_WEATHER_COLS):
                    if col in feat.columns:
                        feat[f'{col}_t+{step}h'] = feat[col].shift(-step)
        for k, v in meta.items():
            feat[k] = v
        feat['station'] = key
        frames.append(feat.dropna())
    return pd.concat(frames).sort_index()


def build_lstm_dataset(stations, horizon=None, with_weather=False):
    """Returns (X, y, sample_stations) where X has shape (n, WINDOW, n_features).
    n_features=1 (sm only) or 8 (sm + 7 weather channels).
    sample_stations is a (n,) array of station keys, one per sample, used to
    apply per-station temporal splits.
    Daily weather values are forward-filled to hourly resolution.
    Stations without complete weather data are skipped when with_weather=True.
    """
    h = horizon or HORIZON
    feat_cols = ['sm_value'] + (list(WEATHER_COLS) + list(DERIVED_WEATHER_COLS) if with_weather else [])
    all_X, all_y, all_keys = [], [], []
    for key, (df, _) in stations.items():
        if not all(c in df.columns for c in feat_cols):
            continue
        arr = df[feat_cols].ffill().bfill().dropna().values.astype(np.float32)
        if len(arr) < WINDOW + h + 10:
            continue
        for i in range(WINDOW, len(arr) - h):
            all_X.append(arr[i - WINDOW:i])
            all_y.append(arr[i + h - 1, 0])
            all_keys.append(key)
    return (np.array(all_X, dtype=np.float32),
            np.array(all_y, dtype=np.float32),
            np.array(all_keys))


def build_tft_dataset(stations, horizon=None, with_weather=False):
    """Returns (X_seq, X_static, y, sample_stations).
    X_seq shape: (n, WINDOW, n_features) — same channels as build_lstm_dataset.
    X_static shape: (n, 4) — per-station lat/lon/elevation/depth.
    sample_stations is a (n,) array of station keys, one per sample.
    """
    h = horizon or HORIZON
    feat_cols = ['sm_value'] + (list(WEATHER_COLS) + list(DERIVED_WEATHER_COLS) if with_weather else [])
    all_X, all_static, all_y, all_keys = [], [], [], []
    for key, (df, meta) in stations.items():
        if not all(c in df.columns for c in feat_cols):
            continue
        arr = df[feat_cols].ffill().bfill().dropna().values.astype(np.float32)
        if len(arr) < WINDOW + h + 10:
            continue
        static = np.array([meta['latitude'], meta['longitude'],
                           meta['elevation_m'], meta['depth_m']], dtype=np.float32)
        for i in range(WINDOW, len(arr) - h):
            all_X.append(arr[i - WINDOW:i])
            all_static.append(static)
            all_y.append(arr[i + h - 1, 0])
            all_keys.append(key)
    return (np.array(all_X, dtype=np.float32),
            np.array(all_static, dtype=np.float32),
            np.array(all_y, dtype=np.float32),
            np.array(all_keys))
