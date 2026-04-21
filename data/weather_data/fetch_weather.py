"""
Fetches daily weather data from NASA POWER API for each ISMN station.
Variables pulled: temperature (min/max/mean), humidity, precipitation,
                  wind speed, and solar radiation.

Output: one CSV per station saved to data/weather_data/stations/{NETWORK}/
        e.g. TAHMO/TAHMO_Mahali_Mzuri_weather.csv

Coverage check is run at the end to verify date and location alignment
with the ISMN soil moisture data.
"""

import os
import time
import requests
import pandas as pd

ISMN_DIR  = os.path.join(os.path.dirname(__file__), '..', 'soil_data', 'ismn_data')
OUT_DIR   = os.path.join(os.path.dirname(__file__), 'stations')
os.makedirs(OUT_DIR, exist_ok=True)

NASA_POWER_URL = 'https://power.larc.nasa.gov/api/temporal/daily/point'

PARAMETERS = ','.join([
    'T2M',        # mean temperature at 2m (°C)
    'T2M_MAX',    # max temperature
    'T2M_MIN',    # min temperature
    'RH2M',       # relative humidity at 2m (%)
    'PRECTOTCORR',# precipitation (mm/day)
    'WS2M',       # wind speed at 2m (m/s)
    'ALLSKY_SFC_SW_DWN',  # solar radiation (MJ/m²/day)
])


def fetch_station(name, lat, lon, start, end):
    params = {
        'parameters': PARAMETERS,
        'community':  'AG',
        'longitude':  lon,
        'latitude':   lat,
        'start':      start.strftime('%Y%m%d'),
        'end':        end.strftime('%Y%m%d'),
        'format':     'JSON',
    }
    resp = requests.get(NASA_POWER_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()['properties']['parameter']
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df.index.name = 'date'
    df.columns.name = None
    return df


def load_station_meta():
    meta = []
    for network in os.listdir(ISMN_DIR):
        net_path = os.path.join(ISMN_DIR, network)
        if not os.path.isdir(net_path):
            continue
        for f in os.listdir(net_path):
            if not f.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(net_path, f), parse_dates=['datetime_utc'])
            meta.append({
                'station':   f[:-4],
                'network':   network,
                'latitude':  df['latitude'].iloc[0],
                'longitude': df['longitude'].iloc[0],
                'start':     df['datetime_utc'].min(),
                'end':       df['datetime_utc'].max(),
            })
    return meta


print('Loading station metadata...')
stations = load_station_meta()
print(f'Found {len(stations)} stations\n')

failed = []
for i, s in enumerate(stations, 1):
    net_out_dir = os.path.join(OUT_DIR, s['network'])
    os.makedirs(net_out_dir, exist_ok=True)
    out_path = os.path.join(net_out_dir, f"{s['network']}_{s['station']}_weather.csv")
    if os.path.exists(out_path):
        print(f"[{i}/{len(stations)}] {s['station']} — already exists, skipping")
        continue
    try:
        print(f"[{i}/{len(stations)}] Fetching {s['station']} "
              f"({s['latitude']:.4f}, {s['longitude']:.4f}) "
              f"{s['start'].date()} → {s['end'].date()} ...", end=' ', flush=True)
        df = fetch_station(s['station'], s['latitude'], s['longitude'],
                           s['start'], s['end'])
        df.to_csv(out_path)
        print(f"saved ({len(df)} days)")
        time.sleep(1)  # be polite to the API
    except Exception as e:
        print(f"FAILED: {e}")
        failed.append(s['station'])

print(f'\n{"="*55}')
print('COVERAGE CHECK')
print(f'{"="*55}')

ok, issues = 0, 0
for s in stations:
    out_path = os.path.join(OUT_DIR, s['network'], f"{s['network']}_{s['station']}_weather.csv")
    if not os.path.exists(out_path):
        print(f"  MISSING  {s['station']}")
        issues += 1
        continue
    w = pd.read_csv(out_path, index_col='date', parse_dates=True)
    sm_start = pd.Timestamp(s['start']).tz_convert(None).normalize() if pd.Timestamp(s['start']).tzinfo else pd.Timestamp(s['start']).normalize()
    sm_end   = pd.Timestamp(s['end']).tz_convert(None).normalize()   if pd.Timestamp(s['end']).tzinfo   else pd.Timestamp(s['end']).normalize()
    w_start  = w.index.min()
    w_end    = w.index.max()
    date_ok  = w_start <= sm_start and w_end >= sm_end
    null_pct = w.isnull().mean().mean()
    status   = 'OK' if date_ok and null_pct < 0.05 else 'WARN'
    if status == 'OK':
        ok += 1
    else:
        issues += 1
    print(f"  {status:<5} {s['network']}/{s['station']:<45} "
          f"weather: {w_start.date()}→{w_end.date()}  "
          f"nulls: {null_pct:.1%}")

print(f'\n{ok} stations OK, {issues} issues')
if failed:
    print(f'Failed to fetch: {failed}')
