"""
One-off asset builder for the Streamlit demo app.

Trains XGBoost (with weather) at t+72h on the per-station temporal split, saves:
  app_assets/xgboost_t72h.joblib              — trained model
  app_assets/feature_cols.json                — feature column order
  app_assets/test_predictions.parquet         — (station, datetime, y_true, y_pred_xgb,
                                                 y_pred_persist, sm_now) for the test set
  app_assets/station_meta.parquet             — lat/lon/elev/depth/network per station
  app_assets/sample_features.parquet          — full feature row for a few thousand
                                                 test points so the app can run XGB live
  app_assets/feature_importance.parquet       — XGB feature importance

Usage:  python scripts/build_app_assets.py
"""

import os
import sys
import json
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from prepare import load_all_stations, build_xgboost_dataset, get_feature_cols, station_temporal_split
from config import XGB_PARAMS, THRESHOLD

HORIZON = 72
HERE = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(HERE, '..', 'app_assets')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print('Loading data with weather features...')
    stations = load_all_stations(with_weather=True)
    print(f'  {len(stations)} stations')

    feature_cols = get_feature_cols(with_weather=True, horizon=HORIZON)
    data = build_xgboost_dataset(stations, horizon=HORIZON, with_weather=True)
    print(f'  Tabular dataset: {data.shape}, {len(feature_cols)} features')

    X = data[feature_cols].values
    y = data['target'].values
    train_mask, test_mask = station_temporal_split(data)

    print(f'  Train rows: {train_mask.sum():,}  Test rows: {test_mask.sum():,}')

    print('Training XGBoost (t+72h, with weather)...')
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X[train_mask], y[train_mask])
    y_pred = xgb.predict(X[test_mask])
    print('  trained.')

    test_data = data.iloc[np.where(test_mask)[0]].reset_index().rename(columns={'datetime_utc': 'datetime'})

    preds = pd.DataFrame({
        'station':       test_data['station'].values,
        'datetime':      test_data['datetime'].values,
        'y_true':        y[test_mask],
        'y_pred_xgb':    y_pred,
        'y_pred_persist': test_data['sm_lag_24h'].values,
        'sm_now':        test_data['sm_value'].values,
        'latitude':      test_data['latitude'].values,
        'longitude':     test_data['longitude'].values,
    })

    # Save model
    model_path = os.path.join(OUT_DIR, 'xgboost_t72h.joblib')
    joblib.dump(xgb, model_path)
    print(f'  Saved {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)')

    # Save feature cols (order matters for live predictions)
    with open(os.path.join(OUT_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)

    # Save test predictions
    preds_path = os.path.join(OUT_DIR, 'test_predictions.parquet')
    preds.to_parquet(preds_path, index=False)
    print(f'  Saved {preds_path} ({len(preds):,} rows)')

    # Save full feature rows for a *sample* of test points (so app can call xgb.predict
    # with user-tweaked features). Keep it under ~30 MB by subsampling.
    n_sample = min(30_000, test_mask.sum())
    rng = np.random.default_rng(0)
    test_idx = np.where(test_mask)[0]
    sample_idx = rng.choice(test_idx, size=n_sample, replace=False)
    sample_idx.sort()
    sample_df = data.iloc[sample_idx].reset_index().rename(columns={'datetime_utc': 'datetime'})
    keep_cols = ['datetime', 'station'] + feature_cols + ['target', 'sm_lag_24h']
    keep_cols = list(dict.fromkeys(keep_cols))   # dedupe but preserve order
    sample_df = sample_df[keep_cols]
    sample_path = os.path.join(OUT_DIR, 'sample_features.parquet')
    sample_df.to_parquet(sample_path, index=False)
    print(f'  Saved {sample_path} ({len(sample_df):,} rows, {os.path.getsize(sample_path)/1024/1024:.1f} MB)')

    # Save station metadata
    meta_rows = []
    for key, (df, meta) in stations.items():
        good_df = df.dropna(subset=['sm_value'])
        meta_rows.append({
            'station':     key,
            'network':     meta['network'],
            'latitude':    meta['latitude'],
            'longitude':   meta['longitude'],
            'elevation_m': meta['elevation_m'],
            'depth_m':     meta['depth_m'],
            'n_readings':  int(len(good_df)),
            'start':       good_df.index.min(),
            'end':         good_df.index.max(),
        })
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(os.path.join(OUT_DIR, 'station_meta.parquet'), index=False)
    print(f'  Saved station_meta.parquet ({len(meta_df)} stations)')

    # Feature importance
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb.feature_importances_,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    fi.to_parquet(os.path.join(OUT_DIR, 'feature_importance.parquet'), index=False)
    print('  Saved feature_importance.parquet')

    # Save a long-window raw timeseries per station so the app can show actual SM history.
    # We keep this lean: 4h downsample, plus the original hourly only for the test segment.
    raw_rows = []
    for key, (df, _meta) in stations.items():
        if df.empty:
            continue
        s = df[['sm_value']].copy()
        s = s[s['sm_value'].notna()]
        if s.empty:
            continue
        # Downsample full history to 6-hourly for context plots
        s6 = s.resample('6h').mean().dropna()
        s6['station'] = key
        s6['resolution'] = '6h'
        raw_rows.append(s6.reset_index().rename(columns={'datetime_utc': 'datetime'}))
    raw = pd.concat(raw_rows, ignore_index=True)
    raw_path = os.path.join(OUT_DIR, 'raw_timeseries.parquet')
    raw.to_parquet(raw_path, index=False)
    print(f'  Saved {raw_path} ({len(raw):,} rows, {os.path.getsize(raw_path)/1024/1024:.1f} MB)')

    # Save weather-feature correlation snapshot (one row per weather var)
    wx_cols_present = ['T2M','T2M_MAX','T2M_MIN','RH2M','PRECTOTCORR','WS2M','ALLSKY_SFC_SW_DWN','ET0']
    cor_rows = []
    for c in wx_cols_present:
        if c in data.columns:
            mask = data[c].notna() & data['sm_value'].notna()
            if mask.sum() < 100:
                continue
            r = float(np.corrcoef(data.loc[mask, c], data.loc[mask, 'sm_value'])[0, 1])
            cor_rows.append({'weather_var': c, 'pearson_r': r, 'n': int(mask.sum())})
    pd.DataFrame(cor_rows).to_parquet(os.path.join(OUT_DIR, 'weather_correlations.parquet'), index=False)
    print('  Saved weather_correlations.parquet')

    # Free memory
    del data, X, y
    gc.collect()

    print('\nDone. Assets written to', OUT_DIR)


if __name__ == '__main__':
    main()
