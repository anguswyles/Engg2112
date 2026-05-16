"""
Soil Moisture Forecaster — Plotly Dash demo.
ENGG2112 Data Farmers.

Launch:  python app.py
Then visit http://localhost:8050
"""

import json
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error


# ──────────────────────────────────────────────────────────────────────
# Paths and constants
# ──────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
APP_ASSETS = HERE / 'app_assets'
IMAGES_DIR = HERE / 'images'

THRESHOLD = 0.30
HORIZON_HOURS = 72


# ──────────────────────────────────────────────────────────────────────
# Design system — single source of truth
# ──────────────────────────────────────────────────────────────────────

class T:
    """Tokens. One palette, one type system, one set of spacings."""

    # Colours
    INK         = '#1A1A1A'   # primary text
    INK_SOFT    = '#555555'   # secondary text
    INK_FAINT   = '#8A8A8A'   # tertiary text / muted labels
    LINE        = '#E0DDD8'   # borders + grid (slightly warmer than before)
    LINE_SOFT   = '#EBE8E3'   # subtle dividers
    PANEL       = '#FFFFFF'   # card / content backgrounds (pure white — lifts off BG)
    BG          = '#F0EDE8'   # page background (warm parchment — gives the white cards depth)

    ACCENT      = '#C25E2A'   # primary accent (terracotta / soil)
    ACCENT_SOFT = '#F4E4D7'   # accent tint for fills
    SIGNAL      = '#1F4E5F'   # secondary signal (deep teal)
    SIGNAL_SOFT = '#DCE7EB'

    OK          = '#3A6B3A'   # green for "healthy"
    OK_SOFT     = '#E5EFE2'
    WARN        = '#B8860B'   # amber for "watch"
    WARN_SOFT   = '#F7EFD7'
    ALERT       = '#A23B3B'   # red for "drought"
    ALERT_SOFT  = '#F2DCDC'

    # Type
    FONT = (
        'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", '
        'Roboto, Helvetica, Arial, sans-serif'
    )


# Plotly template
# Margins: left/bottom are generous so tick labels never clip; top minimal (titles live in HTML)
_MARGIN = dict(l=60, r=24, t=16, b=52)

pio.templates['min'] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=T.FONT, size=13, color=T.INK),
        colorway=[T.ACCENT, T.SIGNAL, T.OK, T.WARN, T.ALERT, T.INK_SOFT],
        plot_bgcolor=T.PANEL,
        paper_bgcolor='rgba(0,0,0,0)',   # transparent — shows page BG behind chart
        margin=_MARGIN,
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=T.LINE,
                   linewidth=1, ticks='outside', ticklen=5,
                   tickcolor=T.LINE, tickfont=dict(size=12, color=T.INK_SOFT),
                   automargin=True),
        yaxis=dict(showgrid=True, gridcolor=T.LINE, gridwidth=1,
                   zeroline=False, showline=True, linecolor=T.LINE, linewidth=1,
                   ticks='outside', ticklen=5, tickcolor=T.LINE,
                   tickfont=dict(size=12, color=T.INK_SOFT),
                   automargin=True),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0,
                    font=dict(size=12, color=T.INK_SOFT)),
        hoverlabel=dict(bgcolor=T.PANEL, bordercolor=T.LINE,
                        font=dict(family=T.FONT, size=12, color=T.INK)),
    )
)
pio.templates.default = 'min'

PLOTLY_CFG = dict(displayModeBar=False, responsive=True)


# ──────────────────────────────────────────────────────────────────────
# Data loaders (run once at import; small enough)
# ──────────────────────────────────────────────────────────────────────

def _country_from_latlon(row):
    lat, lon = row['latitude'], row['longitude']
    if -3.0 <= lat <= -1.0 and 28.5 <= lon <= 31.0: return 'Rwanda'
    if -1.5 < lat <= 4.5 and 29.5 <= lon < 35.5:    return 'Uganda'
    if -5.0 <= lat <= 5.5 and 33.0 <= lon <= 42.5:  return 'Kenya'
    return 'Other'


print('Loading model + assets...')
XGB_MODEL = joblib.load(APP_ASSETS / 'xgboost_t72h.joblib')
with open(APP_ASSETS / 'feature_cols.json') as f:
    FEATURE_COLS = json.load(f)

TEST_PREDS = pd.read_parquet(APP_ASSETS / 'test_predictions.parquet')
TEST_PREDS['datetime'] = pd.to_datetime(TEST_PREDS['datetime'])

SAMPLE_FEATS = pd.read_parquet(APP_ASSETS / 'sample_features.parquet')
SAMPLE_FEATS['datetime'] = pd.to_datetime(SAMPLE_FEATS['datetime'])

META = pd.read_parquet(APP_ASSETS / 'station_meta.parquet')
META['country'] = META.apply(_country_from_latlon, axis=1)
META['short_name'] = META['station'].str.split('/', n=1).str[1].str.replace('_', ' ', regex=False)

FI = pd.read_parquet(APP_ASSETS / 'feature_importance.parquet')

RAW_TS = pd.read_parquet(APP_ASSETS / 'raw_timeseries.parquet')
RAW_TS['datetime'] = pd.to_datetime(RAW_TS['datetime'])

WX_CORR = pd.read_parquet(APP_ASSETS / 'weather_correlations.parquet')


def _read_first(filenames):
    for f in filenames:
        p = IMAGES_DIR / f
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


HORIZON_DF = _read_first([
    'test 10 - current models rerun/horizon_comparison.csv',
    'test 9 - TFT fix + threshold opt/horizon_comparison.csv',
    'test 7 - ET0 dropout tuned XGB/horizon_comparison.csv',
])

ONSET_DF = _read_first([
    'test 10 - current models rerun/onset_analysis.csv',
    'test 9 - TFT fix + threshold opt/onset_analysis.csv',
])

METRICS_DF = _read_first([
    'test 10 - current models rerun/metrics.csv',
    'test 9 - TFT fix + threshold opt/metrics.csv',
    'test 7 - ET0 dropout tuned XGB/metrics.csv',
])
if 'R²' in METRICS_DF.columns:
    METRICS_DF = METRICS_DF.rename(columns={'R²': 'R2'})

WX_ABL_DF = _read_first(['test 11 - weather vs no wx t+120h/weather_ablation_four_models.csv'])

print(f'  ready ({len(META)} stations, {len(TEST_PREDS):,} test predictions)')


STATIONS_SORTED = sorted(META['station'].tolist())
STATION_LABELS = {s: s.split('/', 1)[1].replace('_', ' ') for s in STATIONS_SORTED}


# ──────────────────────────────────────────────────────────────────────
# Reusable components
# ──────────────────────────────────────────────────────────────────────

def section_label(text):
    """Small uppercase label above a section title."""
    return html.Div(
        text.upper(),
        style={
            'fontSize': '11px', 'letterSpacing': '0.15em', 'fontWeight': 600,
            'color': T.INK_FAINT, 'marginBottom': '10px',
        },
    )


def page_title(eyebrow, title, lede=None):
    """Standard page header — eyebrow label, big title, optional one-line lede."""
    children = [
        section_label(eyebrow),
        html.H1(title, style={
            'fontSize': '40px', 'fontWeight': 600, 'letterSpacing': '-0.02em',
            'color': T.INK, 'marginBottom': '12px', 'lineHeight': 1.1,
        }),
    ]
    if lede:
        children.append(html.P(lede, style={
            'fontSize': '17px', 'color': T.INK_SOFT, 'maxWidth': '720px',
            'lineHeight': 1.5, 'marginBottom': 0,
        }))
    return html.Div(children, style={'marginBottom': '40px'})


def section_title(title, sub=None):
    children = [html.H2(title, style={
        'fontSize': '24px', 'fontWeight': 600, 'letterSpacing': '-0.01em',
        'color': T.INK, 'marginBottom': '8px',
    })]
    if sub:
        children.append(html.P(sub, style={
            'fontSize': '15px', 'color': T.INK_SOFT, 'marginBottom': '24px',
            'lineHeight': 1.5,
        }))
    else:
        children[-1].style['marginBottom'] = '24px'
    return html.Div(children, style={'marginBottom': '8px'})


def card(children, padding='28px'):
    return html.Div(
        children,
        style={
            'background': T.PANEL, 'border': f'1px solid {T.LINE}',
            'padding': padding, 'height': '100%',
        },
    )


def stat(label, value, sub=None, value_color=None):
    return html.Div(
        [
            html.Div(label.upper(), style={
                'fontSize': '11px', 'letterSpacing': '0.12em', 'fontWeight': 600,
                'color': T.INK_FAINT, 'marginBottom': '14px',
            }),
            html.Div(value, style={
                'fontSize': '36px', 'fontWeight': 600, 'letterSpacing': '-0.02em',
                'color': value_color or T.INK, 'lineHeight': 1, 'marginBottom': '8px',
            }),
            html.Div(sub or '', style={
                'fontSize': '13px', 'color': T.INK_SOFT, 'lineHeight': 1.4,
            }),
        ],
        style={
            'background': T.BG, 'border': f'1px solid {T.LINE}',
            'padding': '28px', 'height': '100%',
        },
    )


def verdict_panel(level, headline, body):
    color = {'green': T.OK, 'amber': T.WARN, 'red': T.ALERT}[level]
    bg = {'green': T.OK_SOFT, 'amber': T.WARN_SOFT, 'red': T.ALERT_SOFT}[level]
    return html.Div(
        [
            html.Div(headline, style={
                'fontSize': '22px', 'fontWeight': 600, 'color': color,
                'marginBottom': '10px', 'letterSpacing': '-0.01em',
            }),
            html.Div(body, style={
                'fontSize': '15px', 'color': T.INK, 'lineHeight': 1.55,
            }),
        ],
        style={
            'background': bg, 'borderLeft': f'3px solid {color}',
            'padding': '24px 28px',
        },
    )


def divider(margin='48px'):
    return html.Div(style={
        'height': '1px', 'background': T.LINE, 'margin': f'{margin} 0',
    })


def graph(figure, height=420, **kwargs):
    # Only set height — let the template margins stand so axes are never clipped.
    # automargin=True on axes handles label overflow automatically.
    figure.update_layout(height=height)
    return dcc.Graph(figure=figure, config=PLOTLY_CFG, **kwargs)


def graph_id(graph_id, height=420):
    return dcc.Graph(id=graph_id, config=PLOTLY_CFG, style={'height': f'{height}px'})


# ──────────────────────────────────────────────────────────────────────
# Chart builders
# ──────────────────────────────────────────────────────────────────────

def station_map(selected=None, color_field='country'):
    df = META.copy()
    df['marker_size'] = np.where(df['station'] == selected, 18, 9)

    color_map = {
        'Kenya':   T.ACCENT,
        'Uganda':  T.SIGNAL,
        'Rwanda':  T.OK,
        'COSMOS':  T.WARN,
        'TAHMO':   T.SIGNAL,
        'Other':   T.INK_FAINT,
    }

    fig = px.scatter_map(
        df,
        lat='latitude', lon='longitude',
        color=color_field,
        size='marker_size', size_max=18,
        hover_name='short_name',
        hover_data={
            'country': True, 'network': True,
            'elevation_m': ':.0f', 'n_readings': ':,',
            'latitude': ':.3f', 'longitude': ':.3f',
            'marker_size': False,
        },
        color_discrete_map=color_map,
        zoom=4.4,
        opacity=0.85,
    )
    fig.update_layout(
        map_style='carto-positron',
        map_center=dict(lat=0.5, lon=36),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation='h', yanchor='bottom', y=0.01, xanchor='left', x=0.01,
            bgcolor='rgba(255,255,255,0.92)', borderwidth=0,
            font=dict(size=12, color=T.INK_SOFT),
            itemsizing='constant',
        ),
        legend_title_text='',
    )
    return fig


def classify_recommendation(sm_now, sm_pred):
    if sm_pred < THRESHOLD and sm_now < THRESHOLD:
        return ('red',
                'Drought stress now and in 3 days',
                f'Soil moisture is {sm_now:.3f} now and predicted to remain at {sm_pred:.3f}. '
                f'The threshold is {THRESHOLD}. Crops are already water-stressed — '
                'irrigate immediately if water is available.')
    if sm_pred < THRESHOLD and sm_now >= THRESHOLD:
        return ('red',
                'Drought stress incoming within 3 days',
                f'Soil is healthy now ({sm_now:.3f}) but forecast to drop to {sm_pred:.3f} — '
                f'below the {THRESHOLD} threshold. Plan to irrigate in the next 24 to 48 hours. '
                'This is the most valuable warning the model provides.')
    if sm_pred < THRESHOLD + 0.04 and sm_now >= THRESHOLD:
        return ('amber',
                'Borderline — watch closely',
                f'Soil moisture is {sm_now:.3f} now and predicted to fall to {sm_pred:.3f}. '
                f'Close to the {THRESHOLD} threshold. Have irrigation ready.')
    if sm_pred < THRESHOLD + 0.04:
        return ('amber',
                'Currently dry, holding steady',
                f'Soil moisture is {sm_now:.3f} now and predicted {sm_pred:.3f}. '
                'Already near the threshold — irrigation likely needed this week.')
    return ('green',
            'Soil moisture healthy',
            f'Currently {sm_now:.3f}, predicted {sm_pred:.3f} in 3 days. '
            f'Well above the {THRESHOLD} threshold. No irrigation needed.')


# ──────────────────────────────────────────────────────────────────────
# Pages — each returns html children for the content area
# ──────────────────────────────────────────────────────────────────────

# ── HOME ─────────────────────────────────────────────────────────────

def page_home():
    # Hero
    hero = html.Div(
        [
            section_label('ENGG2112  ·  Data Farmers'),
            html.H1(
                'Will the soil be dry on Thursday?',
                style={
                    'fontSize': '54px', 'fontWeight': 600, 'letterSpacing': '-0.03em',
                    'color': T.INK, 'lineHeight': 1.05, 'marginBottom': '24px',
                    'maxWidth': '900px',
                },
            ),
            html.P(
                'A machine-learning system that warns smallholder farmers in East Africa '
                'three days before their soil dries below the drought stress threshold, '
                'so they irrigate in advance rather than after their crops have wilted.',
                style={
                    'fontSize': '19px', 'color': T.INK_SOFT, 'lineHeight': 1.55,
                    'maxWidth': '780px', 'marginBottom': 0,
                },
            ),
        ],
        style={'marginBottom': '64px'},
    )

    # Stat row
    stats = dbc.Row(
        [
            dbc.Col(stat('Sensor stations', '48', 'Kenya, Uganda, Rwanda'), md=3),
            dbc.Col(stat('Hourly readings', '1.1M', 'quality-flagged "Good"'), md=3),
            dbc.Col(stat('Forecast horizon', '7 days', 'best at 1–3 days', T.ACCENT), md=3),
            dbc.Col(stat('Drought events caught', '57%', 'baseline catches 27%', T.ACCENT), md=3),
        ],
        className='gx-4 gy-4',
    )

    # Example forecast chart
    sample_key = 'TAHMO/Kibanda_Hydromet' if 'TAHMO/Kibanda_Hydromet' in TEST_PREDS['station'].unique() else TEST_PREDS['station'].iloc[0]
    s = TEST_PREDS[TEST_PREDS['station'] == sample_key].sort_values('datetime').head(720)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s['datetime'], y=s['y_true'],
        mode='lines', name='Actual soil moisture',
        line=dict(color=T.SIGNAL, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=s['datetime'], y=s['y_pred_xgb'],
        mode='lines', name='Model forecast at t+72h',
        line=dict(color=T.ACCENT, width=2, dash='dot'),
    ))
    fig.add_hline(y=THRESHOLD, line_color=T.ALERT, line_width=1, line_dash='dot')
    fig.add_annotation(
        x=s['datetime'].iloc[-1], y=THRESHOLD, text=f'Drought threshold ({THRESHOLD})',
        showarrow=False, yshift=-14, xshift=-8, xanchor='right',
        font=dict(size=11, color=T.ALERT),
    )
    fig.update_layout(
        xaxis_title=None, yaxis_title='Soil moisture (m³/m³)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    chart_section = html.Div(
        [
            section_title(
                'The problem in one picture',
                f'Thirty days at one station ({STATION_LABELS[sample_key]}). The dotted line '
                'is what the model would have forecast 3 days earlier.',
            ),
            graph(fig, height=420),
        ],
        style={'marginTop': '64px'},
    )

    # Why this matters columns
    why_col = lambda label, title, body: html.Div(
        [
            section_label(label),
            html.H3(title, style={
                'fontSize': '18px', 'fontWeight': 600, 'color': T.INK,
                'marginBottom': '12px', 'letterSpacing': '-0.01em',
            }),
            html.P(body, style={
                'fontSize': '14px', 'color': T.INK_SOFT, 'lineHeight': 1.55, 'marginBottom': 0,
            }),
        ],
        style={'paddingRight': '24px'},
    )

    why = html.Div(
        [
            section_title('Why this matters'),
            dbc.Row(
                [
                    dbc.Col(why_col('01', 'The farmer\'s dilemma',
                        'Most smallholders irrigate reactively. By the time leaves wilt, the yield '
                        'damage is already done. A 3-day warning lets water arrive before stress sets in.'), md=4),
                    dbc.Col(why_col('02', 'The data we have',
                        'ISMN soil sensors give hourly ground-truth moisture. NASA POWER satellites '
                        'give daily weather everywhere. Combined, they cover places no weather station does.'), md=4),
                    dbc.Col(why_col('03', 'What the model adds',
                        'A naive "tomorrow looks like today" forecast catches only 27% of upcoming droughts. '
                        'Our best model catches 57% — twice as many, with fewer false alarms.'), md=4),
                ],
                className='gx-4 gy-4',
            ),
        ],
        style={'marginTop': '64px'},
    )

    return html.Div([hero, stats, chart_section, why])


# ── THE DATA ─────────────────────────────────────────────────────────

def page_data():
    map_section = html.Div(
        [
            section_title(
                'Where the sensors are',
                'Each pin is a soil moisture station. TAHMO covers most of East Africa; '
                'COSMOS uses cosmic-ray probes for deeper readings.',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    section_label('Colour by'),
                                    dcc.RadioItems(
                                        id='map-color-radio',
                                        options=[
                                            {'label': ' Country', 'value': 'country'},
                                            {'label': ' Network', 'value': 'network'},
                                        ],
                                        value='country',
                                        labelStyle={'display': 'block', 'marginBottom': '10px',
                                                    'fontSize': '14px', 'color': T.INK},
                                        inputStyle={'marginRight': '6px'},
                                    ),
                                ],
                                style={'marginBottom': '32px'},
                            ),
                            html.Div(id='map-summary'),
                        ],
                        md=3,
                    ),
                    dbc.Col(graph_id('map-graph', height=520), md=9),
                ],
                className='gx-4',
            ),
        ],
    )

    station_section = html.Div(
        [
            section_title(
                'Inspect any station',
                'Pick a station to see its full history, the distribution of its readings, and how soil moisture cycles through the year.',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_label('Station'),
                            dcc.Dropdown(
                                id='station-deep-dropdown',
                                options=[{'label': STATION_LABELS[s], 'value': s} for s in STATIONS_SORTED],
                                value='TAHMO/Kibanda_Hydromet' if 'TAHMO/Kibanda_Hydromet' in STATIONS_SORTED else STATIONS_SORTED[0],
                                clearable=False,
                                style={'fontSize': '14px'},
                            ),
                            html.Div(id='station-meta-table', style={'marginTop': '24px'}),
                        ],
                        md=3,
                    ),
                    dbc.Col(graph_id('station-history', height=380), md=9),
                ],
                className='gx-4 mb-4',
            ),
            dbc.Row(
                [
                    dbc.Col(graph_id('station-histogram', height=320), md=6),
                    dbc.Col(graph_id('station-seasonality', height=320), md=6),
                ],
                className='gx-4',
            ),
        ],
        style={'marginTop': '64px'},
    )

    # Weather correlations
    corr = WX_CORR.copy()
    corr['label'] = corr['weather_var'].map({
        'T2M': 'Mean temperature', 'T2M_MAX': 'Max temperature',
        'T2M_MIN': 'Min temperature', 'RH2M': 'Relative humidity',
        'PRECTOTCORR': 'Precipitation', 'WS2M': 'Wind speed',
        'ALLSKY_SFC_SW_DWN': 'Solar radiation', 'ET0': 'Evapotranspiration',
    })
    corr = corr.sort_values('pearson_r', key=lambda s: s.abs(), ascending=True)
    colors = [T.ALERT if v < 0 else T.OK for v in corr['pearson_r']]

    fig = go.Figure(go.Bar(
        x=corr['pearson_r'], y=corr['label'], orientation='h',
        marker_color=colors, marker_line_width=0,
        text=[f'{v:+.3f}' for v in corr['pearson_r']], textposition='outside',
        textfont=dict(size=12, color=T.INK_SOFT),
    ))
    fig.add_vline(x=0, line_color=T.LINE, line_width=1)
    fig.update_layout(
        xaxis_title='Pearson correlation with soil moisture',
        yaxis_title=None,
        xaxis=dict(range=[-0.38, 0.3]),
        margin=dict(l=160, r=40, t=16, b=52),   # room for label names on left
    )

    weather_section = html.Div(
        [
            section_title(
                'How much does weather actually relate to soil moisture?',
                'Pearson correlations between each weather feature and raw soil moisture. '
                'No single variable is a strong predictor on its own.',
            ),
            graph(fig, height=400),
            html.Div(
                'Humidity is the strongest single signal (r ≈ -0.30), but every variable on its '
                'own is weak. The model wins by combining all of them with recent soil moisture history.',
                style={'fontSize': '14px', 'color': T.INK_SOFT, 'marginTop': '20px',
                       'paddingLeft': '16px', 'borderLeft': f'2px solid {T.LINE}',
                       'fontStyle': 'italic'},
            ),
        ],
        style={'marginTop': '64px'},
    )

    return html.Div([
        page_title('The data', 'Two complementary sources',
                   'ISMN soil sensors provide hourly ground-truth readings. NASA POWER satellites '
                   'fill in daily weather across the same locations.'),
        map_section,
        station_section,
        weather_section,
    ])


# ── THE MODELS ───────────────────────────────────────────────────────

def page_models():
    # Architecture explainer + R² bar
    arch_rows = [
        ('Persistence', 'Baseline', 'Predicts soil moisture in 3 days equals soil moisture 24 hours ago. No ML — a sanity check.'),
        ('Random Forest', 'Tree ensemble', '200 decision trees voting on the answer. Sees hand-crafted lag and weather features.'),
        ('XGBoost', 'Gradient boosting', 'Each tree corrects the previous one. Same features as RF. Very competitive.'),
        ('LSTM', 'Highest R²', 'Reads the last 5 days of moisture and weather hour by hour. Highest R² in the final run.'),
        ('TFT', 'Transformer', 'LSTM + multi-head attention + static station features (lat, lon, elevation, depth).'),
    ]

    arch_table = html.Div([
        html.Div(
            [
                html.Div(label, style={
                    'fontSize': '14px', 'fontWeight': 600, 'color': T.INK,
                    'flex': '0 0 140px',
                }),
                html.Div(role.upper(), style={
                    'fontSize': '11px', 'letterSpacing': '0.1em', 'color': T.INK_FAINT,
                    'flex': '0 0 130px',
                }),
                html.Div(desc, style={
                    'fontSize': '14px', 'color': T.INK_SOFT, 'lineHeight': 1.55,
                    'flex': '1 1 auto',
                }),
            ],
            style={
                'display': 'flex', 'padding': '20px 0',
                'borderBottom': f'1px solid {T.LINE}',
            },
        )
        for label, role, desc in arch_rows
    ])

    # R² bar — highlight the actual winner from the data
    if not METRICS_DF.empty:
        show = METRICS_DF[METRICS_DF['Model'].isin(['Random Forest', 'XGBoost', 'LSTM', 'TFT'])].copy()
        order = ['Random Forest', 'XGBoost', 'LSTM', 'TFT']
        show = show.set_index('Model').reindex([m for m in order if m in show['Model'].values]).reset_index()
        r2_vals = show['R2'].astype(float)
        best_model = show.loc[r2_vals.idxmax(), 'Model']
        colors = [T.ACCENT if m == best_model else T.SIGNAL_SOFT for m in show['Model']]
        edge   = [T.ACCENT if m == best_model else T.SIGNAL for m in show['Model']]
        txt_c  = [T.BG if m == best_model else T.SIGNAL for m in show['Model']]

        fig_r2 = go.Figure(go.Bar(
            x=show['Model'], y=r2_vals,
            marker=dict(color=colors, line=dict(color=edge, width=1.2)),
            text=[f'{v:.4f}' for v in r2_vals],
            textposition='inside', insidetextanchor='middle',
            textfont=dict(size=14, color=txt_c, family=T.FONT),
        ))
        fig_r2.update_layout(
            yaxis=dict(title='R² at t+72h', range=[0.85, 1.0]),
            xaxis_title=None, showlegend=False,
        )
        winner_str = f'Higher is better. {best_model} leads at R² {r2_vals.max():.4f}.'
    else:
        fig_r2 = go.Figure()
        winner_str = 'Higher R² is better.'

    arch_section = dbc.Row(
        [
            dbc.Col([
                section_title('Four architectures, same problem',
                              'Each model was given the same train/test split and evaluated on the same horizon.'),
                arch_table,
            ], md=7),
            dbc.Col([
                section_title('Performance at t+72h', winner_str),
                graph(fig_r2, height=380),
            ], md=5),
        ],
        className='gx-5',
    )

    # Feature importance section
    fi = FI.head(20).iloc[::-1].copy()

    def categorise(f):
        if f == 'sm_value':           return 'Soil moisture (now)'
        if f.startswith('sm_lag'):    return 'Soil moisture (past)'
        if f.startswith('sm_rolling'):return 'Rolling stats'
        if f in ('hour', 'month', 'dayofyear'): return 'Time'
        if f in ('latitude','longitude','elevation_m','depth_m'): return 'Station'
        if 't+' in f: return 'Forecast weather'
        return 'Current weather'

    fi['category'] = fi['feature'].apply(categorise)
    cat_colors = {
        'Soil moisture (now)':  T.ACCENT,
        'Soil moisture (past)': '#E0997E',
        'Rolling stats':        '#F4E4D7',
        'Time':                 T.WARN,
        'Station':              T.SIGNAL,
        'Current weather':      T.OK,
        'Forecast weather':     '#A8C5A8',
    }
    fi['color'] = fi['category'].map(cat_colors)

    fig_fi = go.Figure(go.Bar(
        x=fi['importance'], y=fi['feature'], orientation='h',
        marker=dict(color=fi['color']),
        customdata=fi[['category']].values,
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<br>%{customdata[0]}<extra></extra>',
    ))
    fig_fi.update_layout(
        xaxis_title='Importance', yaxis_title=None,
        margin=dict(l=200, r=24, t=16, b=52),   # wider left for long feature names
    )

    # Category pie
    fi_full = FI.copy()
    fi_full['category'] = fi_full['feature'].apply(categorise)
    cat_summary = fi_full.groupby('category')['importance'].sum().reset_index().sort_values('importance', ascending=False)
    fig_pie = go.Figure(go.Pie(
        labels=cat_summary['category'], values=cat_summary['importance'],
        marker=dict(colors=[cat_colors.get(c, T.LINE) for c in cat_summary['category']],
                    line=dict(color=T.BG, width=1)),
        hole=0.55,
        textinfo='label+percent', textposition='outside',
        textfont=dict(size=12, color=T.INK_SOFT),
    ))
    fig_pie.update_layout(showlegend=False)

    fi_section = html.Div(
        [
            section_title(
                'What XGBoost actually looks at',
                'Feature importance from the trained model. Most signal comes from the recent past.',
            ),
            dbc.Row(
                [
                    dbc.Col(graph(fig_fi, height=580), md=7),
                    dbc.Col([
                        graph(fig_pie, height=400),
                        html.Div(
                            'Soil moisture is highly autocorrelated. Knowing the current and recent '
                            'readings tells you most of what you need. Weather adds small but reliable '
                            'lift at longer horizons; station identity helps a little.',
                            style={'fontSize': '14px', 'color': T.INK_SOFT, 'lineHeight': 1.55,
                                   'paddingLeft': '16px', 'borderLeft': f'2px solid {T.LINE}',
                                   'fontStyle': 'italic', 'marginTop': '16px'},
                        ),
                    ], md=5),
                ],
                className='gx-5',
            ),
        ],
        style={'marginTop': '80px'},
    )

    # Worked example
    worked_section = html.Div(
        [
            section_title(
                'A worked example',
                'Pick a test point at random, see exactly what XGBoost was given as input, and what it predicted.',
            ),
            html.Div(
                [
                    html.Button(
                        'Try another example',
                        id='worked-example-btn',
                        n_clicks=0,
                        style={
                            'background': T.INK, 'color': T.BG, 'border': 'none',
                            'padding': '12px 24px', 'fontSize': '14px', 'fontWeight': 500,
                            'fontFamily': T.FONT, 'cursor': 'pointer', 'letterSpacing': '0.02em',
                        },
                    ),
                ],
                style={'marginBottom': '32px'},
            ),
            html.Div(id='worked-example-content'),
        ],
        style={'marginTop': '80px'},
    )

    return html.Div([
        page_title('The models', 'Four ways to forecast soil moisture',
                   'Random Forest, XGBoost, LSTM and TFT, all trained side-by-side and benchmarked against a naive persistence baseline.'),
        arch_section,
        fi_section,
        worked_section,
    ])


# ── LIVE DEMO ────────────────────────────────────────────────────────

def page_demo():
    default_station = 'TAHMO/Kibanda_Hydromet' if 'TAHMO/Kibanda_Hydromet' in STATIONS_SORTED else STATIONS_SORTED[0]
    candidates = TEST_PREDS.groupby('station').agg(
        n=('y_true', 'count'), std=('y_true', 'std'),
    ).query('n > 1000').sort_values('std', ascending=False)
    if len(candidates):
        default_station = candidates.index[0]

    return html.Div(
        [
            page_title('Live demo', 'Pick a farm. Pick a day. Get a forecast.',
                       'The model gives you a 3-day soil moisture forecast and an irrigation recommendation, '
                       'using the same XGBoost that won the benchmarks.'),

            # Step 1 — Station picker
            html.Div(
                [
                    section_label('Step 01 — Choose a station'),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id='demo-station-dropdown',
                                                options=[{'label': STATION_LABELS[s], 'value': s} for s in STATIONS_SORTED],
                                                value=default_station,
                                                clearable=False,
                                                style={'fontSize': '14px'},
                                            ),
                                        ],
                                        style={'marginBottom': '24px'},
                                    ),
                                    html.Div(id='demo-station-info'),
                                ],
                                md=3,
                            ),
                            dbc.Col(graph_id('demo-map', height=440), md=9),
                        ],
                        className='gx-4',
                    ),
                ],
                style={'marginBottom': '72px'},
            ),

            # Step 2 — Date slider
            html.Div(
                [
                    section_label('Step 02 — Pick today\'s date'),
                    html.P(
                        'The model will forecast soil moisture 3 days from this point. '
                        'Drag the handle to scrub through the test period.',
                        style={'fontSize': '15px', 'color': T.INK_SOFT,
                               'marginBottom': '24px', 'maxWidth': '720px'},
                    ),
                    html.Div(id='demo-date-display', style={
                        'fontSize': '14px', 'color': T.INK_SOFT, 'marginBottom': '16px',
                        'textAlign': 'center', 'fontWeight': 500,
                    }),
                    html.Div(
                        dcc.Slider(
                            id='demo-date-slider',
                            min=0, max=100, value=40, step=1,
                            marks=None,
                            tooltip={'always_visible': False},
                            updatemode='drag',
                        ),
                        style={'padding': '0 16px'},
                    ),
                ],
                style={'marginBottom': '72px'},
            ),

            # Step 3 — Verdict
            html.Div(
                [
                    section_label('Step 03 — Recommendation'),
                    html.Div(id='demo-verdict', style={'marginBottom': '32px'}),
                    dbc.Row(id='demo-stat-row', className='gx-4 gy-4'),
                ],
                style={'marginBottom': '72px'},
            ),

            # Step 4 — Context chart
            html.Div(
                [
                    section_label('Step 04 — The forecast in context'),
                    html.P(
                        'Solid blue is what actually happened. Orange is what the model predicted '
                        '3 days in advance. Grey is the naive baseline.',
                        style={'fontSize': '15px', 'color': T.INK_SOFT,
                               'marginBottom': '24px', 'maxWidth': '720px'},
                    ),
                    graph_id('demo-context-chart', height=460),
                ],
                style={'marginBottom': '72px'},
            ),

            # Step 5 — Model vs persistence
            html.Div(
                [
                    section_label('Step 05 — Model vs the naive guess'),
                    html.P(
                        'How much better did the model do than just assuming "tomorrow looks like today"?',
                        style={'fontSize': '15px', 'color': T.INK_SOFT,
                               'marginBottom': '24px', 'maxWidth': '720px'},
                    ),
                    dbc.Row(id='demo-comparison-row', className='gx-4'),
                ],
            ),
        ]
    )


# ── RESULTS ──────────────────────────────────────────────────────────

def page_results():
    # Onset analysis section
    onset_section = html.Div()
    if not ONSET_DF.empty:
        show_onset = ONSET_DF[ONSET_DF['Model'].isin(['Persistence', 'Random Forest', 'XGBoost'])].copy()
        show_onset = show_onset.set_index('Model').reindex(['Persistence', 'Random Forest', 'XGBoost'])

        stats_cards = dbc.Row(
            [
                dbc.Col(
                    stat(
                        model,
                        f'{row["Catch rate"]*100:.1f}%',
                        f'{int(row["Caught"]):,} / {int(row["Onsets in test"]):,} caught  ·  '
                        f'{int(row["False alarms"]):,} false alarms',
                        value_color=(T.INK_SOFT if model == 'Persistence' else
                                     T.SIGNAL  if model == 'Random Forest' else
                                     T.ACCENT),
                    ),
                    md=4,
                )
                for model, row in show_onset.iterrows() if pd.notna(row['Catch rate'])
            ],
            className='gx-4 gy-4',
        )

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Drought onsets caught', 'False alarms'),
                            horizontal_spacing=0.18)
        models = show_onset.index.tolist()
        colors_bar = [T.INK_FAINT if m == 'Persistence' else T.SIGNAL if m == 'Random Forest' else T.ACCENT for m in models]

        fig.add_trace(go.Bar(
            x=models, y=show_onset['Catch rate'] * 100,
            marker_color=colors_bar, marker_line_width=0,
            text=[f'{v*100:.1f}%' for v in show_onset['Catch rate']],
            textposition='outside', textfont=dict(size=13, color=T.INK_SOFT),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=models, y=show_onset['False alarm rate'] * 100,
            marker_color=colors_bar, marker_line_width=0,
            text=[f'{v*100:.1f}%' for v in show_onset['False alarm rate']],
            textposition='outside', textfont=dict(size=13, color=T.INK_SOFT),
            showlegend=False,
        ), row=1, col=2)
        fig.update_yaxes(title_text='% of upcoming droughts caught', row=1, col=1, range=[0, 82],
                         showgrid=True, gridcolor=T.LINE)
        fig.update_yaxes(title_text='% false alarm rate', row=1, col=2, range=[0, 14],
                         showgrid=True, gridcolor=T.LINE)
        fig.update_xaxes(showline=False, showgrid=False)
        fig.update_annotations(font=dict(size=13, color=T.INK))
        fig.update_layout(
            plot_bgcolor=T.PANEL, paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=24, t=40, b=52),
            font=dict(family=T.FONT),
        )

        onset_section = html.Div(
            [
                section_title(
                    'The number that matters most',
                    'A drought onset event is when soil is healthy now but will fall below the threshold '
                    'within 3 days. Missing one means thirsty crops. Catching one means time to irrigate.',
                ),
                stats_cards,
                html.Div(style={'height': '40px'}),
                graph(fig, height=440),
            ]
        )

    # Economic value calculator
    economic_section = html.Div(
        [
            section_title('What this means in dollars',
                          'A back-of-envelope estimate of how much yield the model could protect, '
                          'based on FAO maize yield-loss methodology.'),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div('Drought onset events per growing season',
                                     style={'fontSize': '14px', 'color': T.INK_SOFT, 'marginBottom': '12px'}),
                            dcc.Slider(id='econ-events-slider', min=1, max=6, value=3, step=1,
                                       marks={i: str(i) for i in range(1, 7)},
                                       tooltip={'always_visible': False}),
                            html.Div(style={'height': '32px'}),
                            html.Div('Farm size (hectares)',
                                     style={'fontSize': '14px', 'color': T.INK_SOFT, 'marginBottom': '12px'}),
                            dcc.Slider(id='econ-farm-slider', min=0.5, max=5.0, value=1.0, step=0.5,
                                       marks={i: str(i) for i in [1, 2, 3, 4, 5]},
                                       tooltip={'always_visible': False}),
                        ],
                        md=6,
                    ),
                    dbc.Col(html.Div(id='econ-result'), md=6),
                ],
                className='gx-5 gy-4',
            ),
        ],
        style={'marginTop': '80px'},
    )

    # Horizon trade-off
    horizon_section = html.Div(
        [
            section_title('How far ahead can we forecast?',
                          'Every horizon is harder than the one before. Pick a metric to see the trade-off.'),
            html.Div(
                dcc.RadioItems(
                    id='horizon-metric-radio',
                    options=[
                        {'label': ' R²', 'value': 'r2'},
                        {'label': ' MAE', 'value': 'mae'},
                        {'label': ' Recall (catching drought)', 'value': 'recall'},
                    ],
                    value='r2',
                    inline=True,
                    labelStyle={'marginRight': '32px', 'fontSize': '14px', 'color': T.INK},
                    inputStyle={'marginRight': '6px'},
                ),
                style={'marginBottom': '24px'},
            ),
            graph_id('horizon-chart', height=460),
        ],
        style={'marginTop': '80px'},
    )

    # Weather ablation
    wabl_section = html.Div()
    if not WX_ABL_DF.empty:
        wabl_w  = WX_ABL_DF[WX_ABL_DF['Weather'] == 'yes'].set_index('Model')
        wabl_nw = WX_ABL_DF[WX_ABL_DF['Weather'] == 'no' ].set_index('Model')
        models = ['Random Forest', 'XGBoost', 'LSTM', 'TFT']
        models = [m for m in models if m in wabl_w.index and m in wabl_nw.index]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='No weather', x=models, y=[wabl_nw.loc[m, 'R2'] for m in models],
            marker_color=T.SIGNAL_SOFT, marker_line=dict(color=T.SIGNAL, width=1),
            text=[f'{wabl_nw.loc[m, "R2"]:.3f}' for m in models], textposition='outside',
            textfont=dict(size=12, color=T.INK_SOFT),
        ))
        fig.add_trace(go.Bar(
            name='With weather', x=models, y=[wabl_w.loc[m, 'R2'] for m in models],
            marker_color=T.ACCENT_SOFT, marker_line=dict(color=T.ACCENT, width=1),
            text=[f'{wabl_w.loc[m, "R2"]:.3f}' for m in models], textposition='outside',
            textfont=dict(size=12, color=T.INK_SOFT),
        ))
        fig.update_layout(
            barmode='group',
            yaxis=dict(title='R² at t+120h', range=[0.83, 0.95]),
            xaxis_title=None,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=24, t=40, b=52),
        )

        wabl_section = html.Div(
            [
                section_title('Does weather actually help?',
                              'At t+120h (5 days), every model gains from weather features — most by 2-3 R² points.'),
                graph(fig, height=420),
                html.Div(
                    'The benefit is small at short horizons (soil inertia dominates) and grows as we '
                    'forecast further out. At t+24h, weather adds under 0.5 R² points.',
                    style={'fontSize': '14px', 'color': T.INK_SOFT, 'lineHeight': 1.55,
                           'marginTop': '20px', 'paddingLeft': '16px',
                           'borderLeft': f'2px solid {T.LINE}', 'fontStyle': 'italic'},
                ),
            ],
            style={'marginTop': '80px'},
        )

    return html.Div([
        page_title('Results', 'Before, after, and the trade-offs',
                   'Drought-event detection, performance vs horizon, and the impact of adding weather features.'),
        onset_section,
        economic_section,
        horizon_section,
        wabl_section,
    ])


# ── CONCLUSIONS ──────────────────────────────────────────────────────

def page_conclusions():
    findings = [
        ('01', 'The model works', 'R² = 0.93 at t+72h. Average error is 0.015 m³/m³ — about ±5% of the drought threshold. Three days in advance.'),
        ('02', 'It catches what matters', 'When soil will fall from healthy to dry in 3 days, the model catches 57% of those events. The naive baseline catches 27%.'),
        ('03', 'XGBoost beat the neural nets', 'Gradient-boosted trees outperformed LSTM and TFT in most experiments. Hand-crafted lag features are hard to beat for slow physical variables.'),
    ]

    findings_section = html.Div([
        section_title('Headline findings'),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            section_label(num),
                            html.H3(title, style={
                                'fontSize': '20px', 'fontWeight': 600, 'color': T.INK,
                                'marginBottom': '14px', 'letterSpacing': '-0.01em',
                            }),
                            html.P(body, style={
                                'fontSize': '15px', 'color': T.INK_SOFT, 'lineHeight': 1.55,
                                'marginBottom': 0,
                            }),
                        ],
                        style={'paddingRight': '24px'},
                    ),
                    md=4,
                )
                for num, title, body in findings
            ],
            className='gx-4 gy-4',
        ),
    ])

    # Journey
    journey = pd.DataFrame([
        ('Test 1',  'Baseline, no weather, t+24h',                     0.97),
        ('Test 2',  'Weather features added at t+72h',                 0.90),
        ('Test 3',  'Extended lags to 5 days',                         0.898),
        ('Test 4',  'TFT model introduced',                            0.898),
        ('Test 5',  'Weather added to LSTM and TFT sequences',         0.924),
        ('Test 6',  'Rolling-window features added',                   0.923),
        ('Test 7',  'ET0 + tuned XGB + LSTM dropout + norm fix',       0.9305),
        ('Test 8',  'Oracle future weather for LSTM and TFT',          0.9302),
        ('Test 9',  'TFT fix + threshold optimisation',                0.9302),
        ('Test 10', 'Final unified run',                                0.9302),
    ], columns=['iteration', 'change', 'r2'])

    fig_j = go.Figure()
    fig_j.add_trace(go.Scatter(
        x=journey['iteration'], y=journey['r2'],
        mode='lines+markers',
        line=dict(color=T.ACCENT, width=2.5),
        marker=dict(size=10, color=T.ACCENT,
                    line=dict(color=T.BG, width=2)),
        customdata=journey['change'],
        hovertemplate='<b>%{x}</b><br>R²: %{y:.4f}<br>%{customdata}<extra></extra>',
    ))
    fig_j.update_layout(
        yaxis=dict(title='XGBoost R² at t+72h', range=[0.88, 0.97]),
        xaxis=dict(title=None, tickangle=-30),
    )

    journey_section = html.Div(
        [
            section_title('Ten iterations', 'How XGBoost R² evolved across the experiment log. Hover any point for the change.'),
            graph(fig_j, height=400),
        ],
        style={'marginTop': '80px'},
    )

    # Future work
    future = [
        ('Real weather forecasts', 'Right now the model uses observed future weather. The next version uses real numerical weather predictions, with their own errors.'),
        ('Cross-station generalisation', 'Train on 40 stations, test on 8 unseen ones. Quantifies how well the model deploys to a new farm.'),
        ('Soil texture features', 'Clay vs sand vs loam fundamentally changes how soil holds water. SoilGrids provides this globally.'),
        ('Vegetation index (NDVI)', 'Plant transpiration is the main moisture sink between rains. Monthly satellite NDVI would add this signal.'),
        ('Uncertainty quantification', 'Quantile XGBoost can produce prediction ranges, not just point estimates. Farmers need ranges.'),
        ('Seasonal performance', 'Disaggregate metrics by wet and dry season. Performance likely varies between them.'),
    ]

    future_section = html.Div(
        [
            section_title('What we\'d do next'),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H3(title, style={
                                    'fontSize': '16px', 'fontWeight': 600, 'color': T.INK,
                                    'marginBottom': '10px',
                                }),
                                html.P(body, style={
                                    'fontSize': '14px', 'color': T.INK_SOFT, 'lineHeight': 1.55,
                                    'marginBottom': 0,
                                }),
                            ],
                            style={'padding': '24px', 'background': T.PANEL,
                                   'border': f'1px solid {T.LINE}', 'height': '100%'},
                        ),
                        md=4,
                    )
                    for title, body in future
                ],
                className='gx-4 gy-4',
            ),
        ],
        style={'marginTop': '80px'},
    )

    # Final station picker
    final_section = html.Div(
        [
            section_title('One last look', 'Pick any station for a final summary of the model\'s performance there.'),
            dbc.Row(
                [
                    dbc.Col([
                        section_label('Station'),
                        dcc.Dropdown(
                            id='final-station-dropdown',
                            options=[{'label': STATION_LABELS[s], 'value': s} for s in STATIONS_SORTED],
                            value=STATIONS_SORTED[0],
                            clearable=False,
                            style={'fontSize': '14px'},
                        ),
                    ], md=4),
                    dbc.Col(html.Div(id='final-station-stats'), md=8),
                ],
                className='gx-4',
            ),
        ],
        style={'marginTop': '80px'},
    )

    # Team
    team_section = html.Div(
        [
            html.Div(
                'Built by',
                style={
                    'fontSize': '11px', 'letterSpacing': '0.15em', 'color': T.INK_FAINT,
                    'fontWeight': 600, 'textTransform': 'uppercase', 'marginBottom': '12px',
                    'textAlign': 'center',
                },
            ),
            html.H3('The ENGG2112 Data Farmers', style={
                'fontSize': '22px', 'fontWeight': 600, 'color': T.INK,
                'letterSpacing': '-0.01em', 'textAlign': 'center', 'marginBottom': '20px',
            }),
            html.P(
                'Angus  ·  Domain Researcher    James  ·  Project Lead    '
                'Oscar  ·  ML Engineer    Byron  ·  Data Engineer',
                style={'fontSize': '14px', 'color': T.INK_SOFT,
                       'textAlign': 'center', 'letterSpacing': '0.02em', 'marginBottom': 0},
            ),
        ],
        style={
            'marginTop': '96px', 'padding': '48px',
            'background': T.PANEL, 'border': f'1px solid {T.LINE}',
        },
    )

    return html.Div([
        page_title('Conclusions', 'What we built, what we found, and what comes next.'),
        findings_section,
        journey_section,
        future_section,
        final_section,
        team_section,
    ])


# ──────────────────────────────────────────────────────────────────────
# App layout — sidebar + content router
# ──────────────────────────────────────────────────────────────────────

NAV_ITEMS = [
    ('home',        'Home'),
    ('data',        'The Data'),
    ('models',      'The Models'),
    ('demo',        'Live Demo'),
    ('results',     'Results'),
    ('conclusions', 'Conclusions'),
]


def sidebar():
    return html.Div(
        [
            html.Div(
                [
                    html.Div('SOIL MOISTURE', style={
                        'fontSize': '11px', 'letterSpacing': '0.2em',
                        'color': T.INK_FAINT, 'fontWeight': 600,
                    }),
                    html.Div('Forecaster', style={
                        'fontSize': '22px', 'fontWeight': 600, 'color': T.INK,
                        'letterSpacing': '-0.01em', 'marginTop': '4px',
                    }),
                ],
                style={'marginBottom': '48px', 'padding': '36px 32px 0 32px'},
            ),
            html.Div(
                [
                    html.Div(
                        nav_label,
                        id={'type': 'nav-link', 'page': page_id},
                        n_clicks=0,
                        className='nav-link-item',
                        style={
                            'padding': '12px 32px',
                            'fontSize': '14px', 'color': T.INK_SOFT,
                            'cursor': 'pointer', 'borderLeft': '3px solid transparent',
                            'fontFamily': T.FONT, 'fontWeight': 500,
                            'transition': 'all 0.15s ease',
                        },
                    )
                    for page_id, nav_label in NAV_ITEMS
                ],
            ),
            html.Div(
                [
                    html.Div('ENGG2112', style={
                        'fontSize': '10px', 'letterSpacing': '0.2em', 'color': T.INK_FAINT,
                        'fontWeight': 600,
                    }),
                    html.Div('Data Farmers · 2026', style={
                        'fontSize': '12px', 'color': T.INK_FAINT, 'marginTop': '4px',
                    }),
                ],
                style={'position': 'absolute', 'bottom': '36px', 'left': '32px'},
            ),
        ],
        style={
            'position': 'fixed', 'left': 0, 'top': 0, 'bottom': 0,
            'width': '240px', 'background': T.PANEL,   # white sidebar on warm BG
            'borderRight': f'1px solid {T.LINE}',
            'fontFamily': T.FONT,
            'boxShadow': '2px 0 8px rgba(0,0,0,0.04)',
        },
    )


app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    ],
    suppress_callback_exceptions=True,
    title='Soil Moisture Forecaster',
)

# Custom index for global typography
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            html, body { margin: 0; padding: 0; background: ''' + T.BG + '''; }  /* warm parchment BG */
            body { font-family: ''' + T.FONT + '''; color: ''' + T.INK + '''; line-height: 1.5;
                   -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
            .nav-link-item:hover { color: ''' + T.INK + '''; background: ''' + T.LINE_SOFT + '''; }
            .nav-link-item.active {
                color: ''' + T.ACCENT + ''' !important;
                border-left: 3px solid ''' + T.ACCENT + ''' !important;
                font-weight: 600 !important;
                background: ''' + T.BG + ''' !important;
            }
            /* Slider styling */
            .rc-slider-track { background: ''' + T.ACCENT + ''' !important; }
            .rc-slider-rail  { background: ''' + T.LINE + ''' !important; }
            .rc-slider-handle {
                border-color: ''' + T.ACCENT + ''' !important;
                background: ''' + T.BG + ''' !important;
                box-shadow: none !important;
            }
            .rc-slider-handle:active, .rc-slider-handle:hover, .rc-slider-handle:focus {
                border-color: ''' + T.ACCENT + ''' !important;
                box-shadow: 0 0 0 4px ''' + T.ACCENT_SOFT + ''' !important;
            }
            .rc-slider-dot-active { border-color: ''' + T.ACCENT + ''' !important; }
            .rc-slider-mark-text { color: ''' + T.INK_FAINT + ''' !important; font-size: 11px !important;
                                   font-family: ''' + T.FONT + ''' !important; }
            /* Dropdown styling */
            .Select-control, .dash-dropdown .Select-control {
                border: 1px solid ''' + T.LINE + ''' !important;
                border-radius: 0 !important;
                background: ''' + T.BG + ''' !important;
                min-height: 40px !important;
            }
            .Select-control:hover { border-color: ''' + T.INK_FAINT + ''' !important; }
            .is-focused:not(.is-open) > .Select-control {
                border-color: ''' + T.ACCENT + ''' !important;
                box-shadow: none !important;
            }
            .Select-menu-outer { border-radius: 0 !important; border-color: ''' + T.LINE + ''' !important; }
            .VirtualizedSelectFocusedOption { background: ''' + T.PANEL + ''' !important; }
            ::selection { background: ''' + T.ACCENT_SOFT + '''; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div(
    [
        dcc.Store(id='current-page', data='home'),
        sidebar(),
        html.Div(
            html.Div(
                id='page-content',
                style={
                    'background': T.PANEL,
                    'padding': '72px 80px',
                    'minHeight': '100vh',
                },
            ),
            style={
                'marginLeft': '240px',
                'padding': '24px 32px 48px 32px',
                'maxWidth': '1260px',
                'fontFamily': T.FONT,
            },
        ),
    ],
    style={'background': T.BG, 'minHeight': '100vh'},
)


# ──────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────

@app.callback(
    Output('current-page', 'data'),
    Input({'type': 'nav-link', 'page': dash.ALL}, 'n_clicks'),
    State('current-page', 'data'),
    prevent_initial_call=True,
)
def on_nav_click(_clicks, current):
    triggered = dash.callback_context.triggered_id
    if triggered and isinstance(triggered, dict):
        return triggered.get('page', current)
    return current


@app.callback(
    Output('page-content', 'children'),
    Input('current-page', 'data'),
)
def route_page(page_id):
    return {
        'home':        page_home,
        'data':        page_data,
        'models':      page_models,
        'demo':        page_demo,
        'results':     page_results,
        'conclusions': page_conclusions,
    }.get(page_id, page_home)()


# Active nav highlight via clientside callback (CSS class toggle)
app.clientside_callback(
    '''
    function(page) {
        setTimeout(function() {
            document.querySelectorAll('.nav-link-item').forEach(function(el) {
                el.classList.remove('active');
            });
            document.querySelectorAll('[id]').forEach(function(el) {
                try {
                    var idObj = JSON.parse(el.id);
                    if (idObj.type === 'nav-link' && idObj.page === page) {
                        el.classList.add('active');
                    }
                } catch (e) {}
            });
        }, 30);
        return '';
    }
    ''',
    Output('current-page', 'title'),
    Input('current-page', 'data'),
)


# ── DATA page callbacks ──────────────────────────────────────────────

@app.callback(
    Output('map-graph', 'figure'),
    Output('map-summary', 'children'),
    Input('map-color-radio', 'value'),
)
def update_map(color_field):
    fig = station_map(color_field=color_field)
    summary = META.groupby(color_field).agg(
        stations=('station', 'count'),
        readings=('n_readings', 'sum'),
    ).reset_index()

    rows = [
        html.Tr([
            html.Td(name, style={'fontSize': '13px', 'color': T.INK, 'padding': '10px 0'}),
            html.Td(f'{n}', style={'fontSize': '13px', 'color': T.INK_SOFT, 'textAlign': 'right'}),
            html.Td(f'{r:,}', style={'fontSize': '13px', 'color': T.INK_SOFT, 'textAlign': 'right'}),
        ], style={'borderBottom': f'1px solid {T.LINE}'})
        for name, n, r in zip(summary[color_field], summary['stations'], summary['readings'])
    ]
    table = html.Div([
        section_label('Summary'),
        html.Table([
            html.Thead(html.Tr([
                html.Th(color_field.title(), style={'fontSize': '11px', 'color': T.INK_FAINT,
                                                     'fontWeight': 600, 'padding': '8px 0',
                                                     'borderBottom': f'1px solid {T.LINE}',
                                                     'letterSpacing': '0.1em', 'textTransform': 'uppercase'}),
                html.Th('Stns', style={'fontSize': '11px', 'color': T.INK_FAINT, 'fontWeight': 600,
                                       'textAlign': 'right', 'padding': '8px 0',
                                       'borderBottom': f'1px solid {T.LINE}',
                                       'letterSpacing': '0.1em', 'textTransform': 'uppercase'}),
                html.Th('Readings', style={'fontSize': '11px', 'color': T.INK_FAINT, 'fontWeight': 600,
                                            'textAlign': 'right', 'padding': '8px 0',
                                            'borderBottom': f'1px solid {T.LINE}',
                                            'letterSpacing': '0.1em', 'textTransform': 'uppercase'}),
            ])),
            html.Tbody(rows),
        ], style={'width': '100%'}),
    ])
    return fig, table


@app.callback(
    Output('station-meta-table', 'children'),
    Output('station-history', 'figure'),
    Output('station-histogram', 'figure'),
    Output('station-seasonality', 'figure'),
    Input('station-deep-dropdown', 'value'),
)
def update_station_deep(station):
    info = META[META['station'] == station].iloc[0]
    ts = RAW_TS[RAW_TS['station'] == station].sort_values('datetime')

    meta_table = html.Div(
        [
            html.Div([
                html.Span(label, style={'fontSize': '12px', 'color': T.INK_FAINT,
                                        'letterSpacing': '0.08em', 'textTransform': 'uppercase'}),
                html.Span(value, style={'fontSize': '13px', 'color': T.INK, 'float': 'right'}),
            ], style={'padding': '10px 0', 'borderBottom': f'1px solid {T.LINE}'})
            for label, value in [
                ('Network',    info['network']),
                ('Country',    info['country']),
                ('Latitude',   f'{info["latitude"]:.3f}°'),
                ('Longitude',  f'{info["longitude"]:.3f}°'),
                ('Elevation',  f'{info["elevation_m"]:.0f} m'),
                ('Depth',      f'{info["depth_m"]:.2f} m'),
                ('Readings',   f'{info["n_readings"]:,}'),
            ]
        ]
    )

    # History
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=ts['datetime'], y=ts['sm_value'],
        mode='lines', line=dict(color=T.SIGNAL, width=1.2),
        fill='tozeroy', fillcolor='rgba(31, 78, 95, 0.12)',
        showlegend=False,
    ))
    fig_hist.add_hline(y=THRESHOLD, line_color=T.ALERT, line_width=1, line_dash='dot')
    fig_hist.update_layout(yaxis_title='Soil moisture (m³/m³)', xaxis_title=None)

    # Histogram
    fig_dist = go.Figure(go.Histogram(
        x=ts['sm_value'], nbinsx=40,
        marker=dict(color=T.SIGNAL_SOFT, line=dict(color=T.SIGNAL, width=0.5)),
    ))
    fig_dist.add_vline(x=THRESHOLD, line_color=T.ALERT, line_width=1, line_dash='dot')
    fig_dist.update_layout(
        xaxis_title='Soil moisture (m³/m³)', yaxis_title='Count',
        showlegend=False,
    )

    # Seasonality
    ts2 = ts.copy()
    ts2['month'] = ts2['datetime'].dt.month
    monthly = ts2.groupby('month')['sm_value'].agg(['mean', 'std']).reset_index()
    monthly['month_name'] = monthly['month'].apply(lambda m: pd.Timestamp(2024, m, 1).strftime('%b'))
    fig_seas = go.Figure()
    fig_seas.add_trace(go.Scatter(
        x=monthly['month_name'], y=monthly['mean'],
        mode='lines+markers',
        line=dict(color=T.ACCENT, width=2.5),
        marker=dict(size=8, color=T.ACCENT, line=dict(color=T.BG, width=1.5)),
        error_y=dict(type='data', array=monthly['std'], visible=True,
                     color=T.ACCENT_SOFT, thickness=1.5, width=4),
        showlegend=False,
    ))
    fig_seas.add_hline(y=THRESHOLD, line_color=T.ALERT, line_width=1, line_dash='dot')
    fig_seas.update_layout(xaxis_title=None, yaxis_title='Soil moisture (m³/m³)')

    return meta_table, fig_hist, fig_dist, fig_seas


# ── MODELS page worked example ───────────────────────────────────────

@app.callback(
    Output('worked-example-content', 'children'),
    Input('worked-example-btn', 'n_clicks'),
)
def worked_example(n_clicks):
    rng = np.random.default_rng(42 + n_clicks)
    idx = int(rng.integers(0, len(SAMPLE_FEATS)))
    sample = SAMPLE_FEATS.iloc[idx]

    station_name = STATION_LABELS.get(sample['station'], sample['station'])
    dt = pd.to_datetime(sample['datetime'])

    X_row = sample[FEATURE_COLS].values.reshape(1, -1).astype(float)
    pred = float(XGB_MODEL.predict(X_row)[0])
    actual = float(sample['target'])
    err = pred - actual

    # Stat cards
    stats_row = dbc.Row(
        [
            dbc.Col(stat('Soil moisture now', f'{sample["sm_value"]:.3f}',
                         f'{station_name}, {dt.strftime("%d %b %Y, %H:%M")}'), md=4),
            dbc.Col(stat('Predicted at t+72h', f'{pred:.3f}', 'XGBoost output', T.ACCENT), md=4),
            dbc.Col(stat('Actual at t+72h', f'{actual:.3f}', f'error of {err:+.4f}'), md=4),
        ],
        className='gx-4 gy-4',
    )

    # Weather inputs
    wx = {
        'Temperature (°C)':     sample.get('T2M'),
        'Max temp (°C)':        sample.get('T2M_MAX'),
        'Humidity (%)':         sample.get('RH2M'),
        'Precipitation (mm)':   sample.get('PRECTOTCORR'),
        'Wind (m/s)':           sample.get('WS2M'),
        'Solar (MJ/m²/day)':    sample.get('ALLSKY_SFC_SW_DWN'),
        'ET0 (mm/day)':         sample.get('ET0'),
    }
    wx_rows = [
        html.Div([
            html.Span(k, style={'fontSize': '13px', 'color': T.INK_SOFT}),
            html.Span(f'{v:.2f}' if pd.notna(v) else '—',
                      style={'fontSize': '13px', 'color': T.INK, 'float': 'right',
                             'fontVariantNumeric': 'tabular-nums'}),
        ], style={'padding': '10px 0', 'borderBottom': f'1px solid {T.LINE}'})
        for k, v in wx.items()
    ]

    # Verdict
    level = 'green' if abs(err) < 0.02 else ('amber' if abs(err) < 0.05 else 'red')
    body = (f'Predicted {pred:.3f}, actual was {actual:.3f}. '
            f'{"Will stay above" if pred >= THRESHOLD else "Predicted to fall below"} the '
            f'{THRESHOLD} drought threshold.')
    verdict = verdict_panel(level, f'Error: {err:+.4f} m³/m³', body)

    return html.Div([
        stats_row,
        html.Div(style={'height': '40px'}),
        dbc.Row(
            [
                dbc.Col([
                    section_label('Weather context on the day'),
                    html.Div(wx_rows, style={'marginTop': '12px'}),
                ], md=6),
                dbc.Col([
                    section_label('The verdict'),
                    html.Div(verdict, style={'marginTop': '12px'}),
                    html.Div(
                        f'XGBoost was given {len(FEATURE_COLS)} features for this single point — '
                        'soil moisture at 5 past timesteps, rolling means, current weather, '
                        'forecast weather at +24h/+48h/+72h, time of day, time of year, and '
                        'static station info. 300 decision trees voted.',
                        style={'fontSize': '13px', 'color': T.INK_SOFT, 'lineHeight': 1.55,
                               'marginTop': '24px', 'paddingLeft': '14px',
                               'borderLeft': f'2px solid {T.LINE}'},
                    ),
                ], md=6),
            ],
            className='gx-5',
        ),
    ])


# ── LIVE DEMO callbacks ──────────────────────────────────────────────

@app.callback(
    Output('demo-station-info', 'children'),
    Output('demo-map', 'figure'),
    Input('demo-station-dropdown', 'value'),
)
def update_demo_station(station):
    info = META[META['station'] == station].iloc[0]
    info_panel = html.Div(
        [
            html.Div([
                html.Span(label, style={'fontSize': '12px', 'color': T.INK_FAINT,
                                        'letterSpacing': '0.08em', 'textTransform': 'uppercase'}),
                html.Span(value, style={'fontSize': '13px', 'color': T.INK, 'float': 'right',
                                        'fontVariantNumeric': 'tabular-nums'}),
            ], style={'padding': '10px 0', 'borderBottom': f'1px solid {T.LINE}'})
            for label, value in [
                ('Country',   info['country']),
                ('Network',   info['network']),
                ('Latitude',  f'{info["latitude"]:.3f}°'),
                ('Longitude', f'{info["longitude"]:.3f}°'),
                ('Elevation', f'{info["elevation_m"]:.0f} m'),
                ('Depth',     f'{info["depth_m"]:.2f} m'),
            ]
        ]
    )
    return info_panel, station_map(selected=station, color_field='country')


@app.callback(
    Output('demo-date-display', 'children'),
    Output('demo-verdict', 'children'),
    Output('demo-stat-row', 'children'),
    Output('demo-context-chart', 'figure'),
    Output('demo-comparison-row', 'children'),
    Input('demo-station-dropdown', 'value'),
    Input('demo-date-slider', 'value'),
)
def update_demo_forecast(station, slider_pct):
    sp = TEST_PREDS[TEST_PREDS['station'] == station].sort_values('datetime').reset_index(drop=True)
    if len(sp) < 10:
        empty_fig = go.Figure().update_layout(annotations=[
            dict(text='Not enough test data for this station',
                 xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                 font=dict(color=T.INK_SOFT))
        ])
        return 'No data', html.Div(), [], empty_fig, []

    idx = int(np.clip(int(len(sp) * slider_pct / 100), 0, len(sp) - 1))
    row = sp.iloc[idx]

    snap_date = pd.to_datetime(row['datetime'])
    future_date = snap_date + pd.Timedelta(hours=HORIZON_HOURS)
    sm_now = float(row['sm_now'])
    sm_pred = float(row['y_pred_xgb'])
    sm_actual = float(row['y_true'])
    sm_persist = float(row['y_pred_persist'])

    # Date display
    date_display = (
        f"Today: {snap_date.strftime('%a %d %b %Y, %H:%M')}   "
        f"→   Forecast: {future_date.strftime('%a %d %b %Y, %H:%M')}"
    )

    # Verdict
    level, headline, body = classify_recommendation(sm_now, sm_pred)
    verdict = verdict_panel(level, headline, body)

    # Stats row
    delta = sm_pred - sm_now
    margin = sm_pred - THRESHOLD
    stat_cards = [
        dbc.Col(stat('Today', f'{sm_now:.3f}',
                     f'{snap_date.strftime("%a %d %b")}', T.SIGNAL), md=3),
        dbc.Col(stat('Predicted +3 days', f'{sm_pred:.3f}',
                     f'{future_date.strftime("%a %d %b")}', T.ACCENT), md=3),
        dbc.Col(stat('Change', f'{delta:+.3f}',
                     'lower means drier soil',
                     T.ALERT if delta < -0.02 else T.OK), md=3),
        dbc.Col(stat('Margin to drought', f'{margin:+.3f}',
                     f'threshold is {THRESHOLD}',
                     T.ALERT if margin < 0 else T.OK), md=3),
    ]

    # Context chart
    win_before = pd.Timedelta(hours=24 * 14)
    win_after  = pd.Timedelta(hours=24 * 7)
    mask = (sp['datetime'] >= snap_date - win_before) & (sp['datetime'] <= snap_date + win_after)
    ctx = sp[mask].copy()
    ctx['datetime_forecast'] = ctx['datetime'] + pd.Timedelta(hours=HORIZON_HOURS)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ctx['datetime'], y=ctx['sm_now'],
        mode='lines', name='Actual (history)',
        line=dict(color=T.SIGNAL, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=ctx['datetime_forecast'], y=ctx['y_true'],
        mode='lines', name='Actual (future, truth)',
        line=dict(color=T.SIGNAL, width=2, dash='dot'),
    ))
    fig.add_trace(go.Scatter(
        x=ctx['datetime_forecast'], y=ctx['y_pred_xgb'],
        mode='lines', name='Model forecast',
        line=dict(color=T.ACCENT, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=ctx['datetime_forecast'], y=ctx['y_pred_persist'],
        mode='lines', name='Persistence baseline',
        line=dict(color=T.INK_FAINT, width=1.5, dash='dash'),
    ))
    fig.add_trace(go.Scatter(
        x=[future_date], y=[sm_pred], mode='markers',
        marker=dict(color=T.ACCENT, size=12,
                    line=dict(color=T.BG, width=2)),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[future_date], y=[sm_actual], mode='markers',
        marker=dict(color=T.SIGNAL, size=12,
                    line=dict(color=T.BG, width=2)),
        showlegend=False,
    ))
    fig.add_shape(type='line', x0=snap_date, x1=snap_date, y0=0, y1=1,
                  yref='paper', line=dict(color=T.INK, width=1.5))
    fig.add_annotation(x=snap_date, y=1.0, yref='paper', text='Today',
                       showarrow=False, yshift=10,
                       font=dict(color=T.INK, size=12, family=T.FONT))
    fig.add_hline(y=THRESHOLD, line_color=T.ALERT, line_width=1, line_dash='dot')
    fig.update_layout(
        xaxis_title=None, yaxis_title='Soil moisture (m³/m³)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    # Comparison row
    err_xgb = abs(sm_pred - sm_actual)
    err_persist = abs(sm_persist - sm_actual)
    diff = err_persist - err_xgb
    if diff > 0:
        winner_label, winner_value, winner_sub, winner_color = (
            'Model wins by', f'{diff:.4f}', 'fewer absolute errors', T.OK)
    elif diff < 0:
        winner_label, winner_value, winner_sub, winner_color = (
            'Baseline wins by', f'{-diff:.4f}', 'fewer absolute errors', T.ALERT)
    else:
        winner_label, winner_value, winner_sub, winner_color = (
            'Tie', '0.0000', 'equal error', T.INK_FAINT)

    comparison = [
        dbc.Col(stat('Model error', f'{err_xgb:.4f}',
                     f'predicted {sm_pred:.3f} vs actual {sm_actual:.3f}', T.ACCENT), md=4),
        dbc.Col(stat('Baseline error', f'{err_persist:.4f}',
                     f'guessed {sm_persist:.3f}', T.INK_FAINT), md=4),
        dbc.Col(stat(winner_label, winner_value, winner_sub, winner_color), md=4),
    ]

    return date_display, verdict, stat_cards, fig, comparison


# ── RESULTS callbacks ────────────────────────────────────────────────

@app.callback(
    Output('econ-result', 'children'),
    Input('econ-events-slider', 'value'),
    Input('econ-farm-slider', 'value'),
)
def econ_calc(events, farm_size):
    catch_gap = 0.575 - 0.455
    per_event_per_ha = 28.80
    value = events * catch_gap * per_event_per_ha * farm_size
    return html.Div(
        [
            section_label('Estimated yield protected'),
            html.Div(
                f'${value:.0f}',
                style={
                    'fontSize': '64px', 'fontWeight': 600, 'letterSpacing': '-0.03em',
                    'color': T.ACCENT, 'lineHeight': 1, 'marginBottom': '12px',
                },
            ),
            html.Div(
                f'per growing season on a {farm_size:.1f}-hectare farm experiencing '
                f'{events} onset events. Based on FAO maize yield-response methodology.',
                style={'fontSize': '14px', 'color': T.INK_SOFT, 'lineHeight': 1.55,
                       'maxWidth': '320px'},
            ),
        ],
        style={'paddingLeft': '32px', 'borderLeft': f'2px solid {T.LINE}'},
    )


@app.callback(
    Output('horizon-chart', 'figure'),
    Input('horizon-metric-radio', 'value'),
)
def horizon_chart(metric):
    if HORIZON_DF.empty:
        return go.Figure()

    fig = go.Figure()
    if metric == 'r2':
        fig.add_trace(go.Scatter(x=HORIZON_DF['horizon'], y=HORIZON_DF['persist_r2'],
                                 mode='lines+markers', name='Persistence baseline',
                                 line=dict(color=T.INK_FAINT, width=2, dash='dash'),
                                 marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=HORIZON_DF['horizon'], y=HORIZON_DF['xgb_nw_r2'],
                                 mode='lines+markers', name='XGBoost (no weather)',
                                 line=dict(color=T.SIGNAL, width=2),
                                 marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=HORIZON_DF['horizon'], y=HORIZON_DF['xgb_w_r2'],
                                 mode='lines+markers', name='XGBoost (with weather)',
                                 line=dict(color=T.ACCENT, width=2.5),
                                 marker=dict(size=10)))
        fig.update_layout(yaxis_title='R²')
    elif metric == 'mae':
        fig.add_trace(go.Scatter(x=HORIZON_DF['horizon'], y=HORIZON_DF['persist_mae'],
                                 mode='lines+markers', name='Persistence baseline',
                                 line=dict(color=T.INK_FAINT, width=2, dash='dash'),
                                 marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=HORIZON_DF['horizon'], y=HORIZON_DF['xgb_nw_mae'],
                                 mode='lines+markers', name='XGBoost (no weather)',
                                 line=dict(color=T.SIGNAL, width=2),
                                 marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=HORIZON_DF['horizon'], y=HORIZON_DF['xgb_w_mae'],
                                 mode='lines+markers', name='XGBoost (with weather)',
                                 line=dict(color=T.ACCENT, width=2.5),
                                 marker=dict(size=10)))
        fig.update_layout(yaxis_title='MAE (m³/m³)')
    else:
        fig.add_trace(go.Scatter(x=HORIZON_DF['horizon'], y=HORIZON_DF['xgb_nw_recall'],
                                 mode='lines+markers', name='XGBoost (no weather)',
                                 line=dict(color=T.SIGNAL, width=2),
                                 marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=HORIZON_DF['horizon'], y=HORIZON_DF['xgb_w_recall'],
                                 mode='lines+markers', name='XGBoost (with weather)',
                                 line=dict(color=T.ACCENT, width=2.5),
                                 marker=dict(size=10)))
        fig.update_layout(yaxis_title='Recall (drought events caught)')

    fig.update_layout(
        xaxis_title='Forecast horizon (hours)',
        xaxis=dict(tickvals=[24, 48, 72, 96, 120, 168]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


# ── CONCLUSIONS callbacks ────────────────────────────────────────────

@app.callback(
    Output('final-station-stats', 'children'),
    Input('final-station-dropdown', 'value'),
)
def final_station_stats(station):
    sp = TEST_PREDS[TEST_PREDS['station'] == station]
    if len(sp) == 0:
        return html.Div('No test data for this station.', style={'color': T.INK_SOFT})

    below = (sp['y_true'] < THRESHOLD).sum()
    total = len(sp)
    below_pct = below / total * 100
    mae_xgb = mean_absolute_error(sp['y_true'], sp['y_pred_xgb'])
    mae_persist = mean_absolute_error(sp['y_true'], sp['y_pred_persist'])
    ratio = mae_persist / mae_xgb if mae_xgb > 0 else 1.0

    return dbc.Row(
        [
            dbc.Col(stat('Test forecasts', f'{total:,}', 'made at this station'), md=3),
            dbc.Col(stat('Time below threshold', f'{below_pct:.1f}%',
                         f'{below:,} of {total:,} readings', T.ALERT), md=3),
            dbc.Col(stat('Model error', f'{mae_xgb:.4f}', 'mean absolute error', T.ACCENT), md=3),
            dbc.Col(stat('Baseline ratio', f'{ratio:.2f}×', 'baseline error vs model'), md=3),
        ],
        className='gx-3 gy-3',
    )


# ──────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8050)
