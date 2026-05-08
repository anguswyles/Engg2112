# ============================================================
#  ENGG2112 – Weather Feature Impact Analysis
#  Run from repo root: python weather_analysis_final.py
#  Outputs: 6 PNG figures saved to ./outputs/
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

# ── Paths (edit BASE to point at your repo root) ─────────────
BASE = '.'           # <-- change this if running from elsewhere
OUT  = './outputs'
os.makedirs(OUT, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────
C_NO_WX = '#4C72B0'   # blue  – no weather
C_WX    = '#DD8452'   # orange – with weather
C_DIFF  = '#55A868'   # green  – positive delta
C_NEG   = '#C44E52'   # red    – negative delta
GREY    = '#8C8C8C'

# ── Test folders ──────────────────────────────────────────────
test_dirs = {
    'T2: wx features': 'test 2 - weather features t+72h',
    'T3: ext lags':    'test 3 - extended lags 5 day',
    'T5: wx seqs':     'test 5 - weather sequences',
    'T6: rolling':     'test 6 - rolling features',
    'T7: ET0+tuned':   'test 7 - ET0 dropout tuned XGB',
    'T8: oracle wx':   'test 8 - oracle wx LSTM TFT',
    'T8: unified':     'test 8 - unified splits',
}

# ── Load CSVs ─────────────────────────────────────────────────
all_metrics  = {}
horizon_data = {}
for label, folder in test_dirs.items():
    img_dir = os.path.join(BASE, 'images', folder)
    mp = os.path.join(img_dir, 'metrics.csv')
    hp = os.path.join(img_dir, 'horizon_comparison.csv')
    if os.path.exists(mp):
        all_metrics[label] = pd.read_csv(mp)
    if os.path.exists(hp):
        df = pd.read_csv(hp)
        if 'xgb_nw_r2' in df.columns and 'xgb_w_r2' in df.columns:
            horizon_data[label] = df

# ═══════════════════════════════════════════════════════════════
# FIG 1 – R² line plots: weather vs no-weather per test
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)
axes = axes.flatten()

for i, (label, df) in enumerate(horizon_data.items()):
    ax = axes[i]
    ax.plot(df['horizon'], df['xgb_nw_r2'], 'o-', color=C_NO_WX, lw=2,
            label='XGB – No Weather', markersize=6)
    ax.plot(df['horizon'], df['xgb_w_r2'], 's--', color=C_WX, lw=2,
            label='XGB – With Weather', markersize=6)
    ax.fill_between(df['horizon'], df['xgb_nw_r2'], df['xgb_w_r2'],
                    alpha=0.15, color=C_DIFF)
    max_delta = (df['xgb_w_r2'] - df['xgb_nw_r2']).max()
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Horizon (h)')
    ax.set_ylabel('R²')
    ax.legend(fontsize=7.5)
    ax.set_ylim(0.7, 1.0)
    ax.grid(alpha=0.3)
    ax.annotate(f'max Δ={max_delta:.4f}', xy=(0.97, 0.05),
                xycoords='axes fraction', ha='right', fontsize=8,
                color=C_DIFF, fontweight='bold')

for j in range(len(horizon_data), len(axes)):
    axes[j].set_visible(False)

fig.suptitle('R² – Weather vs No-Weather XGBoost Across Forecast Horizons',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_r2_weather_vs_noweather.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1")

# ═══════════════════════════════════════════════════════════════
# FIG 2 – Heatmaps: ΔR² and ΔMAE across all tests × horizons
# ═══════════════════════════════════════════════════════════════
# Keep only tests with the same number of horizon rows
ref_horizons = None
valid = {}
for label, df in horizon_data.items():
    if ref_horizons is None:
        ref_horizons = df['horizon'].values
    if len(df) == len(ref_horizons):
        valid[label] = df

r2_gains  = pd.DataFrame(index=ref_horizons)
mae_gains = pd.DataFrame(index=ref_horizons)
for label, df in valid.items():
    r2_gains[label]  = df['xgb_w_r2'].values  - df['xgb_nw_r2'].values
    mae_gains[label] = df['xgb_nw_mae'].values - df['xgb_w_mae'].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

im1 = ax1.imshow(r2_gains.values.T, aspect='auto', cmap='RdYlGn', vmin=-0.01, vmax=0.06)
ax1.set_xticks(range(len(r2_gains.index)))
ax1.set_xticklabels([f't+{h}h' for h in r2_gains.index], fontsize=9)
ax1.set_yticks(range(len(r2_gains.columns)))
ax1.set_yticklabels(r2_gains.columns, fontsize=9)
ax1.set_title('ΔR² Gain from Weather\n(green = helpful, red = harmful)', fontweight='bold')
plt.colorbar(im1, ax=ax1, label='ΔR²')
for yi, col in enumerate(r2_gains.columns):
    for xi in range(len(r2_gains.index)):
        ax1.text(xi, yi, f'{r2_gains.iloc[xi][col]:.4f}',
                 ha='center', va='center', fontsize=7)

im2 = ax2.imshow(mae_gains.values.T, aspect='auto', cmap='RdYlGn', vmin=-0.001, vmax=0.008)
ax2.set_xticks(range(len(mae_gains.index)))
ax2.set_xticklabels([f't+{h}h' for h in mae_gains.index], fontsize=9)
ax2.set_yticks(range(len(mae_gains.columns)))
ax2.set_yticklabels(mae_gains.columns, fontsize=9)
ax2.set_title('ΔMAE Reduction from Weather\n(green = lower MAE, better)', fontweight='bold')
plt.colorbar(im2, ax=ax2, label='ΔMAE')
for yi, col in enumerate(mae_gains.columns):
    for xi in range(len(mae_gains.index)):
        ax2.text(xi, yi, f'{mae_gains.iloc[xi][col]:.5f}',
                 ha='center', va='center', fontsize=7)

fig.suptitle('Weather Feature Impact Heatmap – XGBoost', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_weather_impact_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2")

# ═══════════════════════════════════════════════════════════════
# FIG 3 – % improvement from weather (T8 unified, canonical)
# ═══════════════════════════════════════════════════════════════
final_hz = horizon_data.get('T8: unified', list(horizon_data.values())[-1])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metric_defs = [
    ('R²',     'xgb_nw_r2',     'xgb_w_r2',     False),  # higher = better
    ('MAE',    'xgb_nw_mae',    'xgb_w_mae',    True),   # lower = better
]
if 'xgb_nw_recall' in final_hz.columns:
    metric_defs.append(('Recall', 'xgb_nw_recall', 'xgb_w_recall', False))

for ax, (mname, nwcol, wcol, lower_better) in zip(axes, metric_defs):
    nw  = final_hz[nwcol].values
    w   = final_hz[wcol].values
    pct = ((nw - w) / nw * 100) if lower_better else ((w - nw) / np.abs(nw) * 100)
    horizons = final_hz['horizon'].values
    colors = [C_DIFF if v >= 0 else C_NEG for v in pct]
    bars = ax.bar([f't+{h}h' for h in horizons], pct, color=colors, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title(f'{mname} % Change with Weather\n(positive = weather helps)', fontweight='bold')
    ax.set_ylabel('% change')
    for bar, val in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.01 if val >= 0 else -0.08),
                f'{val:+.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlabel('Forecast Horizon')

fig.suptitle('% Improvement from Weather Features (T8 Unified XGBoost)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_pct_improvement.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3")

# ═══════════════════════════════════════════════════════════════
# FIG 4 – All models, all metrics, across tests
# ═══════════════════════════════════════════════════════════════
model_rows = []
for label, df in all_metrics.items():
    for _, row in df.iterrows():
        model_rows.append({'test': label, **row.to_dict()})
mdf = pd.DataFrame(model_rows)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, ['R²', 'MAE', 'ROC AUC']):
    if metric not in mdf.columns:
        continue
    for model, color, marker in [
        ('XGBoost',       C_NO_WX, 'o'),
        ('LSTM',          C_WX,    's'),
        ('TFT',           C_DIFF,  '^'),
        ('Random Forest', GREY,    'D'),
    ]:
        sub = mdf[(mdf['Model'] == model)].dropna(subset=[metric])
        if sub.empty:
            continue
        xs = list(range(len(sub)))
        ax.scatter(xs, sub[metric], label=model, color=color, marker=marker, s=70, zorder=3)
        ax.plot(xs, sub[metric].values, color=color, alpha=0.4, lw=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels(sub['test'].values, rotation=35, ha='right', fontsize=7)
    ax.set_title(f'{metric} across Tests', fontweight='bold')
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle('All Models – Performance Across Tests', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_model_comparison_across_tests.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4")

# ═══════════════════════════════════════════════════════════════
# FIG 5 – Raw weather–soil moisture correlations from station data
# ═══════════════════════════════════════════════════════════════
SOIL_DIR    = os.path.join(BASE, 'data', 'soil_data', 'ismn_data')
WEATHER_DIR = os.path.join(BASE, 'data', 'weather_data', 'stations')
WEATHER_COLS = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'PRECTOTCORR', 'WS2M', 'ALLSKY_SFC_SW_DWN']

combined = []
for network in os.listdir(SOIL_DIR):
    net_path = os.path.join(SOIL_DIR, network)
    if not os.path.isdir(net_path):
        continue
    for f in sorted(os.listdir(net_path))[:15]:   # sample up to 15 stations/network
        if not f.endswith('.csv'):
            continue
        try:
            soil = pd.read_csv(os.path.join(net_path, f), parse_dates=['datetime_utc'])
            soil = soil[soil['ismn_flag_good']][['datetime_utc', 'sm_value']].copy()
            soil['date'] = pd.to_datetime(soil['datetime_utc']).dt.tz_localize(None).dt.normalize()
            soil_daily   = soil.groupby('date')['sm_value'].mean().reset_index()
            wx_path = os.path.join(WEATHER_DIR, network, f"{network}_{f[:-4]}_weather.csv")
            if not os.path.exists(wx_path):
                continue
            wx     = pd.read_csv(wx_path, parse_dates=['date'])
            merged = soil_daily.merge(wx, on='date', how='inner')
            combined.append(merged)
        except Exception:
            pass

all_data = pd.concat(combined, ignore_index=True)

wx_labels = {
    'T2M':            'Avg Temp (°C)',
    'T2M_MAX':        'Max Temp (°C)',
    'T2M_MIN':        'Min Temp (°C)',
    'RH2M':           'Rel. Humidity (%)',
    'PRECTOTCORR':    'Precipitation (mm/day)',
    'WS2M':           'Wind Speed (m/s)',
    'ALLSKY_SFC_SW_DWN': 'Solar Radiation (MJ/m²/day)',
}

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
last_mask = None
for i, (col, clabel) in enumerate(wx_labels.items()):
    ax = axes[i]
    if col not in all_data.columns:
        ax.set_visible(False)
        continue
    mask = all_data['sm_value'].notna() & all_data[col].notna()
    last_mask = mask
    x = all_data.loc[mask, col]
    y = all_data.loc[mask, 'sm_value']
    hb = ax.hexbin(x, y, gridsize=35, cmap='YlOrRd', mincnt=1)
    m, b, r, p, _ = stats.linregress(x, y)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m * xr + b, color='navy', lw=2)
    ax.set_xlabel(clabel, fontsize=9)
    ax.set_ylabel('Soil Moisture (m³/m³)', fontsize=9)
    sig   = '***' if p < 0.001 else ('*' if p < 0.05 else 'ns')
    tcolor = C_NEG if abs(r) < 0.2 else ('darkgreen' if abs(r) > 0.5 else 'darkorange')
    ax.set_title(f'{col}   r={r:.3f} {sig}', fontsize=10, fontweight='bold', color=tcolor)
    plt.colorbar(hb, ax=ax, label='count')

axes[-1].set_visible(False)
n_pts = last_mask.sum() if last_mask is not None else 0
fig.suptitle(
    f'Weather Features vs Soil Moisture Correlation\n'
    f'(n≈{n_pts:,} station-days  |  weak r → limited predictive value)',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_weather_sm_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5")

# ═══════════════════════════════════════════════════════════════
# FIG 6 – Numeric summary table (T8 unified, all horizons)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

rows = []
for h_row in final_hz.itertuples():
    h    = h_row.horizon
    nwr2 = h_row.xgb_nw_r2;  wr2 = h_row.xgb_w_r2
    nwm  = h_row.xgb_nw_mae; wm  = h_row.xgb_w_mae
    dr2  = wr2 - nwr2
    dm   = nwm - wm
    rows.append([
        f't+{h}h',
        f'{nwr2:.4f}', f'{wr2:.4f}', f'{dr2:+.4f}', f'{dr2/abs(nwr2)*100:+.3f}%',
        f'{nwm:.4f}',  f'{wm:.4f}',  f'{dm:+.5f}',  f'{dm/nwm*100:+.3f}%',
    ])

col_names = ['Horizon', 'R² no wx', 'R² wx', 'ΔR²', 'ΔR² %',
             'MAE no wx', 'MAE wx', 'ΔMAE', 'ΔMAE %']

table = ax.table(cellText=rows, colLabels=col_names, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.9)

for j in range(len(col_names)):
    table[0, j].set_facecolor('#2C3E50')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(rows) + 1):
    for j_col in [3, 4, 7, 8]:   # delta columns
        try:
            v  = float(rows[i - 1][j_col].replace('%', ''))
            fc = '#d4efdf' if v > 0 else ('#fadbd8' if v < 0 else 'white')
        except ValueError:
            fc = 'white'
        table[i, j_col].set_facecolor(fc)

ax.set_title(
    'XGBoost: Weather vs No-Weather Performance (T8 Unified Splits, t+24h → t+168h)',
    fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6")
print("\n✅ All figures saved to", OUT)
