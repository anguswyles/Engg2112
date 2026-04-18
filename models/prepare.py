import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ismn_data')
LAGS     = (24, 48, 72)
HORIZON  = 24   # predict soil moisture 24h ahead
WINDOW   = 72   # LSTM input window


def load_all_stations():
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
            df = df[df['ismn_flag_good'] == True][['datetime_utc', 'sm_value']].copy()
            df = df.set_index('datetime_utc').sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df = df.resample('1h').interpolate(method='time', limit=6)
            stations[f"{network}/{f[:-4]}"] = (df, meta)
    return stations


def build_xgboost_dataset(stations):
    frames = []
    for key, (df, meta) in stations.items():
        if len(df) < WINDOW + HORIZON + 10:
            continue
        feat = df.copy()
        feat['target'] = feat['sm_value'].shift(-HORIZON)
        for lag in LAGS:
            feat[f'sm_lag_{lag}h'] = feat['sm_value'].shift(lag)
        feat['hour']      = feat.index.hour
        feat['month']     = feat.index.month
        feat['dayofyear'] = feat.index.dayofyear
        for k, v in meta.items():
            feat[k] = v
        feat['station'] = key
        feat = feat.dropna()
        frames.append(feat)
    return pd.concat(frames).sort_index()


def build_lstm_dataset(stations):
    all_X, all_y = [], []
    for key, (df, _) in stations.items():
        sm = df['sm_value'].dropna().values
        if len(sm) < WINDOW + HORIZON + 10:
            continue
        for i in range(WINDOW, len(sm) - HORIZON):
            all_X.append(sm[i - WINDOW:i])
            all_y.append(sm[i + HORIZON - 1])
    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)
