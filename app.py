"""
Drought Warning System — Plotly Dash demo.
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
    INK         = '#181613'   # primary text — warm near-black
    INK_SOFT    = '#5A554F'   # secondary text
    INK_FAINT   = '#8E867D'   # tertiary text / muted labels
    LINE        = '#E2DDD5'   # borders + grid
    LINE_SOFT   = '#EEEAE3'   # subtle dividers
    PANEL       = '#FBF9F5'   # card / content backgrounds (warm paper)
    PANEL_RAISED= '#FFFFFF'   # raised surface
    BG          = '#F2EEE7'   # page background (warm parchment)
    BG_DEEP     = '#E9E4DB'   # deeper recess for contrast wells

    ACCENT      = '#B8521F'   # primary accent (terracotta / soil)
    ACCENT_DEEP = '#8A3A12'   # accent on press / strong emphasis
    ACCENT_SOFT = '#F2DFCE'   # accent tint for fills
    ACCENT_GLOW = 'rgba(184, 82, 31, 0.18)'

    SIGNAL      = '#1E4856'   # secondary signal (deep teal)
    SIGNAL_SOFT = '#D6E2E7'

    OK          = '#3A6B3A'   # green for "healthy"
    OK_SOFT     = '#E5EFE2'
    WARN        = '#B07706'   # amber for "watch"
    WARN_SOFT   = '#F7ECD2'
    ALERT       = '#A23B3B'   # red for "drought"
    ALERT_SOFT  = '#F2DCDC'

    # Elevation
    SHADOW_SM   = '0 1px 2px rgba(40, 30, 20, 0.04), 0 1px 1px rgba(40, 30, 20, 0.03)'
    SHADOW_MD   = '0 4px 14px rgba(40, 30, 20, 0.06), 0 1px 3px rgba(40, 30, 20, 0.04)'
    SHADOW_LG   = '0 18px 40px rgba(40, 30, 20, 0.08), 0 4px 10px rgba(40, 30, 20, 0.05)'

    # Motion — Emil's strong custom curves (cubic-bezier)
    EASE_OUT    = 'cubic-bezier(0.23, 1, 0.32, 1)'           # entrances, hovers
    EASE_INOUT  = 'cubic-bezier(0.77, 0, 0.175, 1)'          # on-screen movement
    EASE_DRAWER = 'cubic-bezier(0.32, 0.72, 0, 1)'           # iOS-feel
    DUR_FAST    = '140ms'
    DUR_BASE    = '200ms'
    DUR_SLOW    = '420ms'

    # Type
    FONT = (
        '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", '
        'Roboto, Helvetica, Arial, sans-serif'
    )
    FONT_DISPLAY = (
        '"Fraunces", "Inter", -apple-system, BlinkMacSystemFont, Georgia, serif'
    )
    FONT_MONO = (
        '"JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace'
    )


# Plotly template
# Margins: left/bottom are generous so tick labels never clip; top minimal (titles live in HTML)
_MARGIN = dict(l=64, r=28, t=20, b=56)

pio.templates['min'] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=T.FONT, size=13, color=T.INK),
        colorway=[T.ACCENT, T.SIGNAL, T.OK, T.WARN, T.ALERT, T.INK_SOFT],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=_MARGIN,
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=T.LINE,
                   linewidth=1, ticks='outside', ticklen=4,
                   tickcolor=T.LINE, tickfont=dict(size=11, color=T.INK_FAINT,
                                                   family=T.FONT_MONO),
                   automargin=True),
        yaxis=dict(showgrid=True, gridcolor=T.LINE_SOFT, gridwidth=1,
                   zeroline=False, showline=False, linecolor=T.LINE, linewidth=1,
                   ticks='outside', ticklen=0, tickcolor=T.LINE,
                   tickfont=dict(size=11, color=T.INK_FAINT, family=T.FONT_MONO),
                   automargin=True),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0,
                    font=dict(size=12, color=T.INK_SOFT)),
        hoverlabel=dict(bgcolor=T.PANEL_RAISED, bordercolor=T.LINE,
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

# Event-level onset classifier metrics. Prefer the latest 7-day warning run,
# falling back to the earlier 72h experiment if those outputs are unavailable.
ONSET_EVENT_DF = _read_first([
    'test 14 - drought onset 168h 24h tolerance/drought_onset_event_metrics.csv',
])
if ONSET_EVENT_DF.empty:
    _ONSET_EVENT_MODELS = _read_first([
        'test 12 - drought onset tolerant events/drought_onset_event_metrics.csv',
    ])
    _ONSET_EVENT_BASELINES = _read_first([
        'test 13 - drought onset linear trend baseline/drought_onset_event_metrics.csv',
    ])
    if not _ONSET_EVENT_MODELS.empty and not _ONSET_EVENT_BASELINES.empty:
        ONSET_EVENT_DF = pd.concat([_ONSET_EVENT_BASELINES, _ONSET_EVENT_MODELS], ignore_index=True)
    elif not _ONSET_EVENT_MODELS.empty:
        ONSET_EVENT_DF = _ONSET_EVENT_MODELS.copy()
    else:
        ONSET_EVENT_DF = pd.DataFrame()

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

def section_label(text, accent=False):
    """Small uppercase label above a section title (eyebrow)."""
    cls = 'eyebrow eyebrow-accent' if accent else 'eyebrow'
    return html.Div(
        text.upper(),
        className=cls,
        style={'marginBottom': '14px'},
    )


def page_title(eyebrow, title, lede=None):
    """Standard page header — eyebrow label, big serif display title, optional lede."""
    children = [
        html.Div(className='reveal reveal-1', children=[section_label(eyebrow, accent=True)]),
        html.Div(
            className='reveal reveal-2',
            children=html.H1(
                title,
                className='display',
                style={
                    'fontSize': 'clamp(40px, 5.2vw, 60px)',
                    'color': T.INK,
                    'margin': '0 0 16px 0',
                    'maxWidth': '900px',
                },
            ),
        ),
    ]
    if lede:
        children.append(
            html.Div(
                className='reveal reveal-3',
                children=html.P(lede, style={
                    'fontSize': '18px', 'color': T.INK_SOFT, 'maxWidth': '720px',
                    'lineHeight': 1.55, 'margin': 0, 'fontWeight': 400,
                }),
            )
        )
    # decorative rule under headers
    children.append(html.Div(className='rule', style={'marginTop': '32px'}))
    return html.Div(children, style={'marginBottom': '56px'})


def section_title(title, sub=None):
    children = [html.H2(title, style={
        'fontSize': '26px', 'fontWeight': 600, 'letterSpacing': '-0.018em',
        'color': T.INK, 'margin': '0 0 10px 0', 'lineHeight': 1.15,
        'fontFamily': T.FONT_DISPLAY,
    })]
    if sub:
        children.append(html.P(sub, style={
            'fontSize': '15px', 'color': T.INK_SOFT, 'margin': '0 0 28px 0',
            'lineHeight': 1.55, 'maxWidth': '720px',
        }))
    else:
        children[-1].style['marginBottom'] = '28px'
    return html.Div(children, style={'marginBottom': '8px'})


def card(children, padding='28px', lift=True, raised=False):
    bg = T.PANEL_RAISED if raised else T.PANEL
    return html.Div(
        children,
        className='lift' if lift else '',
        style={
            'background': bg,
            'border': f'1px solid {T.LINE}',
            'borderRadius': '6px',
            'padding': padding,
            'height': '100%',
            'boxShadow': T.SHADOW_SM,
        },
    )


def stat(label, value, sub=None, value_color=None):
    return html.Div(
        [
            html.Div(label.upper(), className='eyebrow', style={'marginBottom': '18px'}),
            html.Div(value, className='stat-value', style={
                'fontSize': '42px',
                'fontWeight': 500,
                'color': value_color or T.INK,
                'lineHeight': 1,
                'marginBottom': '10px',
                'fontFamily': T.FONT_DISPLAY,
            }),
            html.Div(sub or '', style={
                'fontSize': '13px', 'color': T.INK_SOFT, 'lineHeight': 1.45,
            }),
        ],
        className='lift',
        style={
            'background': T.PANEL_RAISED,
            'border': f'1px solid {T.LINE}',
            'borderRadius': '6px',
            'padding': '30px',
            'height': '100%',
            'boxShadow': T.SHADOW_SM,
            'position': 'relative',
            'overflow': 'hidden',
        },
    )


def verdict_panel(level, headline, body):
    color = {'green': T.OK, 'amber': T.WARN, 'red': T.ALERT}[level]
    bg = {'green': T.OK_SOFT, 'amber': T.WARN_SOFT, 'red': T.ALERT_SOFT}[level]
    # `data-verdict` + a level-specific `key` make Dash re-mount the node
    # each time `level` changes, replaying the slide/blur entry animation.
    return html.Div(
        [
            html.Div(
                [
                    html.Span(className='pulse-dot', style={
                        'background': color, 'marginRight': '12px',
                        'verticalAlign': 'middle',
                    }),
                    html.Span(headline, style={
                        'fontSize': '22px', 'fontWeight': 600, 'color': color,
                        'letterSpacing': '-0.012em',
                        'fontFamily': T.FONT_DISPLAY,
                    }),
                ],
                style={'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center'},
            ),
            html.Div(body, style={
                'fontSize': '15px', 'color': T.INK, 'lineHeight': 1.6,
            }),
        ],
        key=f'verdict-{level}-{headline[:24]}',
        **{'data-verdict': level},
        style={
            'background': bg,
            'borderLeft': f'3px solid {color}',
            'borderRadius': '6px',
            'padding': '24px 28px',
        },
    )


def _comparison_card(label, value, value_sub, description, value_color):
    return html.Div(
        [
            html.Div(label.upper(), className='eyebrow', style={'marginBottom': '22px'}),
            html.Div(value, className='stat-value', style={
                'fontSize': '64px',
                'fontWeight': 500,
                'color': value_color,
                'lineHeight': 0.95,
                'marginBottom': '8px',
                'fontFamily': T.FONT_DISPLAY,
            }),
            html.Div(value_sub, className='eyebrow', style={'marginBottom': '24px'}),
            html.Div(className='rule', style={'marginBottom': '20px'}),
            html.Div(description, style={
                'fontSize': '14px', 'color': T.INK_SOFT, 'lineHeight': 1.6,
            }),
        ],
        className='lift',
        style={
            'background': T.PANEL_RAISED,
            'border': f'1px solid {T.LINE}',
            'borderRadius': '6px',
            'padding': '32px',
            'height': '100%',
            'boxShadow': T.SHADOW_SM,
        },
    )


def divider(margin='56px'):
    return html.Div(
        className='rule',
        style={'margin': f'{margin} 0'},
    )


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
    # Hero — display serif with italic accent, staggered reveal
    hero = html.Div(
        [
            html.Div(
                [
                    html.Span(className='pulse-dot', style={
                        'marginRight': '12px', 'verticalAlign': 'middle',
                    }),
                    html.Span('ENGG2112 · DATA FARMERS · 2026', className='eyebrow eyebrow-accent', style={
                        'verticalAlign': 'middle',
                    }),
                ],
                className='reveal reveal-1',
                style={'marginBottom': '28px', 'display': 'flex', 'alignItems': 'center'},
            ),
            html.Div(
                html.H1(
                    [
                        'A 7-day ',
                        html.Em('drought'),
                        html.Br(),
                        'warning system.',
                    ],
                    className='display',
                    style={
                        'fontSize': 'clamp(48px, 6.4vw, 76px)',
                        'color': T.INK,
                        'margin': '0 0 32px 0',
                        'maxWidth': '1000px',
                    },
                ),
                className='reveal reveal-2',
            ),
            html.Div(
                html.P(
                    [
                        'The core model predicts whether currently healthy soil will enter ',
                        html.Span('drought stress', style={
                            'color': T.INK, 'fontWeight': 500,
                            'borderBottom': f'1px solid {T.ACCENT}',
                            'paddingBottom': '1px',
                        }),
                        ' around one week from now. A second drought-state model then scans '
                        'later days to estimate whether the event is likely to be short-lived '
                        'or persistent.',
                    ],
                    style={
                        'fontSize': '20px', 'color': T.INK_SOFT, 'lineHeight': 1.55,
                        'maxWidth': '780px', 'margin': 0, 'fontWeight': 400,
                    },
                ),
                className='reveal reveal-3',
            ),
            html.Div(className='rule reveal reveal-4', style={'marginTop': '48px'}),
        ],
        style={'marginBottom': '72px'},
    )

    # Stat row — staggered reveal, reflecting the onset + state pipeline
    stats = html.Div(
        dbc.Row(
            [
                dbc.Col(html.Div(stat('Sensor stations', '48', 'Kenya, Uganda, Rwanda'),
                                 className='reveal reveal-2'), md=3),
                dbc.Col(html.Div(stat('Onset window', '168–192h',
                                      '7-day warning with 24h tolerance'),
                                 className='reveal reveal-3'), md=3),
                dbc.Col(html.Div(stat('State windows', '24h',
                                      'scanned forward for duration', T.SIGNAL),
                                 className='reveal reveal-4'), md=3),
                dbc.Col(html.Div(stat('Drought events caught', '98.7%',
                                      '297/301 · linear baseline catches 51.5%', T.ACCENT),
                                 className='reveal reveal-5'), md=3),
            ],
            className='gx-4 gy-4',
        ),
    )

    system_section = html.Div(
        [
            section_title(
                'What the system predicts',
                'The project is now centred on drought events, not just point soil-moisture accuracy.',
            ),
            html.Div(
                dbc.Row(
                    [
                        dbc.Col(
                            stat('1. Onset classifier', 'Will drought start?', 'XGBoost/RF classify drought entry 7 days ahead', T.ACCENT),
                            md=4,
                        ),
                        dbc.Col(
                            stat('2. State scanner', 'Will it persist?', 'RF/XGBoost check later 24h drought-state windows', T.SIGNAL),
                            md=4,
                        ),
                        dbc.Col(
                            stat('3. Moisture forecast', 'Supporting signal', 'Regression still explains the underlying soil-moisture path', T.INK_SOFT),
                            md=4,
                        ),
                    ],
                    className='gx-4 gy-4',
                ),
                className='reveal-on-scroll-stagger',
            ),
        ],
        className='reveal-on-scroll',
        style={'marginTop': '72px'},
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
                'The supporting soil-moisture forecast',
                f'Thirty days at one station ({STATION_LABELS[sample_key]}). The dotted line '
                'is the older point forecast, which now supports the drought warning story rather than leading it.',
            ),
            graph(fig, height=340),
        ],
        className='reveal-on-scroll',
        style={'marginTop': '72px'},
    )

    # Why this matters columns — numbered, editorial
    why_col = lambda label, title, body: html.Div(
        [
            html.Div(label, style={
                'fontFamily': T.FONT_MONO,
                'fontSize': '12px',
                'letterSpacing': '0.18em',
                'color': T.ACCENT,
                'fontWeight': 500,
                'marginBottom': '20px',
                'paddingBottom': '14px',
                'borderBottom': f'1px solid {T.LINE}',
            }),
            html.H3(title, style={
                'fontSize': '20px', 'fontWeight': 500, 'color': T.INK,
                'marginBottom': '14px', 'letterSpacing': '-0.012em',
                'fontFamily': T.FONT_DISPLAY, 'lineHeight': 1.25,
            }),
            html.P(body, style={
                'fontSize': '14.5px', 'color': T.INK_SOFT, 'lineHeight': 1.6, 'marginBottom': 0,
            }),
        ],
        style={'paddingRight': '32px'},
    )

    why = html.Div(
        [
            section_title('Why this matters'),
            html.Div(
                dbc.Row(
                    [
                        dbc.Col(why_col('01', 'The farmer\'s dilemma',
                            'Most smallholders irrigate reactively. By the time leaves wilt, the yield '
                            'damage is already done. A 7-day warning gives time to plan water before stress sets in.'), md=4),
                        dbc.Col(why_col('02', 'The data we have',
                            'ISMN soil sensors give hourly ground-truth moisture. NASA POWER satellites '
                            'give daily weather everywhere. Combined, they cover places no weather station does.'), md=4),
                        dbc.Col(why_col('03', 'What the model adds',
                            'A linear drying-trend baseline catches 51.5% of upcoming drought events while raising 483 false warnings. '
                            'XGBoost at the best-F1 threshold catches 297 of 301 events — 98.7% — at a 7-day horizon with a 24-hour onset window.'), md=4),
                    ],
                    className='gx-4 gy-4',
                ),
                className='reveal-on-scroll-stagger',
            ),
        ],
        className='reveal-on-scroll',
        style={'marginTop': '72px'},
    )

    # ── Model leaderboard — 5 models head-to-head from test 14 ─────────
    leaderboard = _home_leaderboard()

    # ── Top drivers — top 8 features from xgboost_feature_importance ───
    drivers = _home_drivers()

    # ── Horizon strip — degradation with lead time ─────────────────────
    horizon_strip = _home_horizon()

    # ── Station network preview ────────────────────────────────────────
    network = _home_network()

    # ── CTA strip ──────────────────────────────────────────────────────
    cta = _home_cta()

    return html.Div([
        hero,
        stats,
        chart_section,
        leaderboard,
        system_section,
        drivers,
        horizon_strip,
        network,
        why,
        cta,
    ])


def _home_leaderboard():
    """Five models, one chart, the question answered. Pulled from test 14."""
    if ONSET_EVENT_DF.empty:
        return html.Div()

    df = ONSET_EVENT_DF[ONSET_EVENT_DF['Threshold label'] == 'best_f1'].copy()
    order = ['Persistence', 'Linear Trend', 'LSTM', 'TFT', 'Random Forest', 'XGBoost']
    df = df[df['Model'].isin(order)].copy()
    df['Model'] = pd.Categorical(df['Model'], categories=order, ordered=True)
    df = df.sort_values('Model')

    recalls = (df['Event recall'].astype(float) * 100).round(1).tolist()
    precisions = (df['Event precision'].astype(float) * 100).round(1).tolist()
    models = df['Model'].astype(str).tolist()

    colors = {
        'Persistence':   T.INK_FAINT,
        'Linear Trend':  T.SIGNAL,
        'LSTM':          T.WARN,
        'TFT':           T.OK,
        'Random Forest': T.ACCENT_DEEP,
        'XGBoost':       T.ACCENT,
    }
    bar_colors = [colors[m] for m in models]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=('Event recall — % of 301 drought onsets caught',
                                        'Event precision — % of warnings that were real'))
    fig.add_trace(go.Bar(
        x=models, y=recalls, marker_color=bar_colors,
        text=[f'{v:.1f}%' for v in recalls], textposition='outside',
        textfont=dict(family=T.FONT_MONO, size=11, color=T.INK),
        cliponaxis=False, showlegend=False, hovertemplate='%{x}<br>recall %{y:.1f}%<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=models, y=precisions, marker_color=bar_colors,
        text=[f'{v:.1f}%' for v in precisions], textposition='outside',
        textfont=dict(family=T.FONT_MONO, size=11, color=T.INK),
        cliponaxis=False, showlegend=False, hovertemplate='%{x}<br>precision %{y:.1f}%<extra></extra>',
    ), row=1, col=2)
    fig.update_yaxes(range=[0, 109], showticklabels=False, showgrid=False)
    fig.update_xaxes(tickfont=dict(size=11, family=T.FONT, color=T.INK_SOFT))
    fig.update_layout(
        bargap=0.45,
        margin=dict(l=10, r=10, t=44, b=20),
        annotations=[
            dict(text=t['text'], x=t['x'], y=1.10, xref='paper', yref='paper',
                 showarrow=False, font=dict(size=12, color=T.INK_FAINT, family=T.FONT))
            for t in fig.layout.annotations
        ],
    )

    return html.Div(
        [
            section_title(
                'How five models stack up',
                'At a 7-day horizon with a 24-hour onset window. Best-F1 thresholds. '
                'Persistence assumes nothing changes; the linear baseline is a smart heuristic; the rest are trained models.',
            ),
            graph(fig, height=320),
            html.Div(
                [
                    html.Span('SOURCE  ', className='eyebrow'),
                    html.Span('test 14 · drought onset 168h 24h tolerance · 42,948 test samples · 301 onset events',
                              style={'fontFamily': T.FONT_MONO, 'fontSize': '11px',
                                     'color': T.INK_FAINT, 'letterSpacing': '0.04em'}),
                ],
                style={'marginTop': '12px', 'textAlign': 'right'},
            ),
        ],
        className='reveal-on-scroll',
        style={'marginTop': '88px'},
    )


def _home_drivers():
    """Top features the XGBoost classifier actually uses."""
    if FI.empty and len(FI) == 0:
        return html.Div()

    # Prefer the onset model FI from test 14 if available; else fall back to FI parquet
    fi_csv = IMAGES_DIR / 'test 14 - drought onset 168h 24h tolerance' / 'xgboost_feature_importance.csv'
    if fi_csv.exists():
        fi = pd.read_csv(fi_csv).head(8)
    else:
        fi = FI.head(8).copy()
        if 'group' not in fi.columns:
            fi['group'] = 'feature'

    label_map = {
        'sm_value':              ('Current soil moisture',        'soil'),
        'sm_rolling_mean_72h':   ('3-day rolling mean',           'soil'),
        'sm_rolling_mean_168h':  ('7-day rolling mean',           'soil'),
        'sm_rolling_std_168h':   ('7-day rolling volatility',     'soil'),
        'sm_rolling_std_72h':    ('3-day rolling volatility',     'soil'),
        'sm_lag_24h':            ('Soil moisture 24h ago',        'soil'),
        'PRECTOTCORR_t+48h':     ('Rainfall forecast · +48h',     'weather'),
        'PRECTOTCORR_t+72h':     ('Rainfall forecast · +72h',     'weather'),
        'PRECTOTCORR_t+96h':     ('Rainfall forecast · +96h',     'weather'),
        'PRECTOTCORR_t+120h':    ('Rainfall forecast · +120h',    'weather'),
        'PRECTOTCORR_t+144h':    ('Rainfall forecast · +144h',    'weather'),
        'ET0_t+48h':             ('Evapotranspiration · +48h',    'weather'),
        'ET0_t+96h':             ('Evapotranspiration · +96h',    'weather'),
        'ALLSKY_SFC_SW_DWN_t+144h': ('Solar radiation · +144h',   'weather'),
    }

    rows = []
    max_imp = float(fi['importance'].max())
    for _, r in fi.iterrows():
        feat = r['feature']
        label, group = label_map.get(feat, (feat, r.get('group', 'feature')))
        pct = (float(r['importance']) / max_imp) * 100
        color = T.ACCENT if group == 'soil' else T.SIGNAL
        rows.append(html.Div(
            [
                html.Div(label, style={
                    'flex': '0 0 240px', 'fontSize': '14px',
                    'color': T.INK, 'fontWeight': 500,
                }),
                html.Div(
                    html.Div(style={
                        'width': f'{pct:.1f}%', 'height': '100%',
                        'background': color, 'borderRadius': '2px',
                        'transition': f'width 900ms {T.EASE_OUT}',
                    }),
                    style={
                        'flex': '1 1 auto', 'height': '8px',
                        'background': T.LINE_SOFT, 'borderRadius': '2px',
                        'overflow': 'hidden', 'margin': '0 18px',
                    },
                ),
                html.Div(f'{float(r["importance"])*100:.1f}', style={
                    'flex': '0 0 56px', 'textAlign': 'right',
                    'fontFamily': T.FONT_MONO, 'fontSize': '12px',
                    'color': T.INK_FAINT, 'letterSpacing': '0.04em',
                }),
            ],
            style={'display': 'flex', 'alignItems': 'center', 'padding': '12px 0',
                   'borderBottom': f'1px solid {T.LINE_SOFT}'},
        ))

    legend = html.Div(
        [
            html.Span([
                html.Span(style={'display': 'inline-block', 'width': '10px', 'height': '10px',
                                 'background': T.ACCENT, 'borderRadius': '2px',
                                 'marginRight': '8px', 'verticalAlign': 'middle'}),
                'Soil moisture history',
            ], style={'marginRight': '28px', 'fontSize': '12px', 'color': T.INK_SOFT,
                      'fontFamily': T.FONT_MONO, 'letterSpacing': '0.04em'}),
            html.Span([
                html.Span(style={'display': 'inline-block', 'width': '10px', 'height': '10px',
                                 'background': T.SIGNAL, 'borderRadius': '2px',
                                 'marginRight': '8px', 'verticalAlign': 'middle'}),
                'Weather forecast',
            ], style={'fontSize': '12px', 'color': T.INK_SOFT,
                      'fontFamily': T.FONT_MONO, 'letterSpacing': '0.04em'}),
        ],
        style={'marginTop': '16px'},
    )

    return html.Div(
        [
            section_title(
                'What the model actually looks at',
                'The top eight features by XGBoost importance gain. The model leans on recent soil-moisture '
                'history first, then peers ahead at upcoming rainfall and evapotranspiration to anticipate the dry-down.',
            ),
            html.Div(rows, style={'borderTop': f'1px solid {T.LINE_SOFT}'}),
            legend,
        ],
        className='reveal-on-scroll',
        style={'marginTop': '88px'},
    )


def _home_horizon():
    """How the model degrades as we forecast farther ahead."""
    if HORIZON_DF.empty:
        return html.Div()

    # Find an R²/recall column we can chart
    df = HORIZON_DF.copy()
    if 'XGBoost_R2' in df.columns:
        ycol, label = 'XGBoost_R2', 'XGBoost R²'
    elif 'XGBoost (with wx)' in df.columns:
        ycol, label = 'XGBoost (with wx)', 'XGBoost R²'
    else:
        # try any numeric column other than horizon
        cands = [c for c in df.columns if c.lower() not in ('horizon', 'horizon_hours', 'hours')]
        ycol = next((c for c in cands if df[c].dtype.kind in 'fi'), None)
        if ycol is None:
            return html.Div()
        label = ycol

    xcol = next((c for c in df.columns if c.lower() in ('horizon', 'horizon_hours', 'hours')), df.columns[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[xcol], y=df[ycol],
        mode='lines+markers',
        line=dict(color=T.ACCENT, width=2.5, shape='spline'),
        marker=dict(size=8, color=T.ACCENT, line=dict(color=T.PANEL_RAISED, width=2)),
        hovertemplate='+%{x}h<br>' + label + ' %{y:.3f}<extra></extra>',
    ))
    fig.update_layout(
        xaxis_title='Lead time (hours)',
        yaxis_title=label,
        margin=dict(l=64, r=28, t=20, b=56),
    )

    return html.Div(
        [
            section_title(
                'Sharper close-in, softer far-out',
                'Forecast quality decays with lead time — accuracy is best at 24h and bottoms out near a week. '
                'The drought-onset classifier sits on top of this, converting the regression signal into a binary warning.',
            ),
            graph(fig, height=300),
        ],
        className='reveal-on-scroll',
        style={'marginTop': '88px'},
    )


def _home_network():
    """A compact preview of the 48-station network."""
    if META.empty:
        return html.Div()

    fig = px.scatter_map(
        META, lat='latitude', lon='longitude',
        color='country',
        color_discrete_map={'Kenya': T.ACCENT, 'Uganda': T.SIGNAL,
                            'Rwanda': T.OK, 'Other': T.INK_FAINT},
        hover_name='short_name',
        hover_data={'country': True, 'network': True, 'elevation_m': ':.0f',
                    'latitude': False, 'longitude': False},
        zoom=4.2, opacity=0.85,
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        map_style='carto-positron',
        map_center=dict(lat=0.5, lon=36),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=0.02, xanchor='left', x=0.02,
                    bgcolor='rgba(255,255,255,0.92)', font=dict(size=11, color=T.INK_SOFT),
                    itemsizing='constant'),
        legend_title_text='',
    )

    # Country breakdown
    counts = META['country'].value_counts()
    breakdown = []
    for country in ['Kenya', 'Uganda', 'Rwanda']:
        n = int(counts.get(country, 0))
        if not n:
            continue
        col = {'Kenya': T.ACCENT, 'Uganda': T.SIGNAL, 'Rwanda': T.OK}[country]
        breakdown.append(html.Div(
            [
                html.Div(country.upper(), className='eyebrow', style={'marginBottom': '8px'}),
                html.Div(str(n), className='stat-value', style={
                    'fontSize': '36px', 'fontFamily': T.FONT_DISPLAY,
                    'color': col, 'lineHeight': 1, 'marginBottom': '4px',
                }),
                html.Div('stations', style={'fontSize': '12px', 'color': T.INK_SOFT}),
            ],
            style={'padding': '20px 24px', 'borderLeft': f'2px solid {col}',
                   'marginBottom': '16px'},
        ))

    return html.Div(
        [
            section_title(
                'Forty-eight stations across East Africa',
                'ISMN soil-moisture sensors from the TAHMO and COSMOS networks, '
                'paired with NASA POWER satellite weather. Each dot is a real, calibrated sensor.',
            ),
            dbc.Row(
                [
                    dbc.Col(graph(fig, height=380), md=8),
                    dbc.Col(html.Div(breakdown), md=4),
                ],
                className='gx-4',
            ),
        ],
        className='reveal-on-scroll',
        style={'marginTop': '88px'},
    )


def _home_cta():
    """End-of-page nudge toward the live demo."""
    return html.Div(
        html.Div(
            [
                html.Div('TRY IT YOURSELF', className='eyebrow eyebrow-accent',
                         style={'marginBottom': '18px'}),
                html.H2([
                    'Pick a station. ',
                    html.Em('See the warning.'),
                ], className='display', style={
                    'fontSize': 'clamp(32px, 4vw, 48px)', 'color': T.INK,
                    'margin': '0 0 16px 0', 'maxWidth': '720px',
                }),
                html.P(
                    'The live demo runs the trained XGBoost classifier against held-out data for any '
                    'of the 48 stations, and surfaces the same recommendation a farmer would receive.',
                    style={
                        'fontSize': '17px', 'color': T.INK_SOFT, 'lineHeight': 1.6,
                        'maxWidth': '640px', 'margin': '0 0 32px 0',
                    },
                ),
                html.Div(
                    [
                        html.Span('Open the live demo', style={'verticalAlign': 'middle'}),
                        html.Span('  ↗', style={
                            'fontFamily': T.FONT_MONO, 'marginLeft': '6px',
                            'verticalAlign': 'middle',
                        }),
                    ],
                    id={'type': 'nav-link', 'page': 'demo'},
                    n_clicks=0,
                    className='press',
                    style={
                        'display': 'inline-block',
                        'padding': '14px 28px',
                        'background': T.INK,
                        'color': T.PANEL_RAISED,
                        'fontSize': '14px',
                        'fontWeight': 500,
                        'letterSpacing': '0.02em',
                        'cursor': 'pointer',
                        'borderRadius': '6px',
                        'boxShadow': T.SHADOW_MD,
                    },
                ),
            ],
            style={'padding': '64px 56px', 'maxWidth': '900px'},
        ),
        className='reveal-on-scroll',
        style={
            'marginTop': '120px',
            'background': T.PANEL_RAISED,
            'border': f'1px solid {T.LINE}',
            'borderRadius': '8px',
            'boxShadow': T.SHADOW_SM,
        },
    )


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
        page_title('Supporting model details', 'How the soil-moisture forecaster works',
                   'The original regression models remain useful for explaining the soil-moisture path, but the main product is now drought onset and duration warning.'),
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
            page_title('Moisture forecast', 'Pick a farm. Pick a day. Inspect the supporting forecast.',
                       'This page keeps the original 3-day soil-moisture regression demo. It is now a supporting diagnostic beside the newer drought-onset warning system.'),

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
    # Onset analysis section — uses event-level 7-day classifier metrics (test 14)
    onset_section = html.Div()
    _onset_src = ONSET_EVENT_DF if not ONSET_EVENT_DF.empty else pd.DataFrame()
    if not _onset_src.empty:
        _display_models = ['Linear Trend', 'Random Forest', 'XGBoost']
        _color_map = {
            'Linear Trend':   T.INK_FAINT,
            'Random Forest':  T.SIGNAL,
            'XGBoost':        T.ACCENT,
        }
        # Use the best-F1 operating point from the tolerant 168-192h warning experiment.
        _hi_catch = (
            _onset_src[
                (_onset_src['Model'].isin(_display_models)) &
                (_onset_src['Threshold label'] == 'best_f1')
            ]
            .copy()
            .set_index('Model')
            .reindex(_display_models)
        )

        stats_cards = dbc.Row(
            [
                dbc.Col(
                    stat(
                        model,
                        f'{row["Event recall"]*100:.1f}%',
                        f'{int(row["Caught events"]):,} / {int(row["Actual onset events"]):,} events caught  ·  '
                        f'{int(row["False alarm events"]):,} false alarm events',
                        value_color=_color_map.get(model, T.INK_SOFT),
                    ),
                    md=4,
                )
                for model, row in _hi_catch.iterrows() if pd.notna(row['Event recall'])
            ],
            className='gx-4 gy-4',
        )

        _models_list = _hi_catch.index.tolist()
        _colors_bar  = [_color_map.get(m, T.INK_SOFT) for m in _models_list]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Drought onset events caught (event recall)', 'False alarm events'),
            horizontal_spacing=0.18,
        )
        fig.add_trace(go.Bar(
            x=_models_list,
            y=_hi_catch['Event recall'] * 100,
            marker_color=_colors_bar, marker_line_width=0,
            text=[f'{v*100:.1f}%' for v in _hi_catch['Event recall']],
            textposition='outside', textfont=dict(size=13, color=T.INK_SOFT),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=_models_list,
            y=_hi_catch['False alarm events'],
            marker_color=_colors_bar, marker_line_width=0,
            text=[str(int(v)) if pd.notna(v) else '' for v in _hi_catch['False alarm events']],
            textposition='outside', textfont=dict(size=13, color=T.INK_SOFT),
            showlegend=False,
        ), row=1, col=2)
        fig.update_yaxes(title_text='% of drought events caught', row=1, col=1, range=[0, 110],
                         showgrid=True, gridcolor=T.LINE)
        fig.update_yaxes(title_text='False alarm events (count)', row=1, col=2,
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
                    'Seven-day drought warnings',
                    'A drought onset event is when soil is currently healthy but falls below the threshold '
                    'around 7 days from now. The model uses a 24-hour onset window, so warnings are useful '
                    'without pretending the exact hour of crossing is perfectly known.',
                ),
                stats_cards,
                html.Div(style={'height': '40px'}),
                graph(fig, height=440),
            ]
        )

    # Duration / drought-state extension — three-step pipeline
    duration_section = html.Div(
        [
            section_title(
                'From "will drought start?" to "how long will it last?"',
                'The latest implementation adds a drought-state model that can scan later 24-hour windows '
                'after a warning. This is the foundation for estimating whether a coming drought is a short '
                'dip or a sustained stress period.',
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(stat('Step 1', 'Onset',
                                          'warns for drought in the 168–192h window', T.ACCENT),
                                     className='reveal reveal-1'), md=4),
                    dbc.Col(html.Div(stat('Step 2', 'State scan',
                                          'checks later 24h windows with RF/XGBoost', T.SIGNAL),
                                     className='reveal reveal-2'), md=4),
                    dbc.Col(html.Div(stat('Step 3', 'Duration',
                                          'stops after sustained predicted recovery', T.OK),
                                     className='reveal reveal-3'), md=4),
                ],
                className='gx-4 gy-4',
            ),
            html.Div(
                'The state model predicts whether soil is below the drought threshold for at least 6 hours '
                'inside each future day. Running those predictions forward lets the app turn a warning into '
                'a practical duration bucket such as short, moderate, long, or extended.',
                style={
                    'fontSize': '14.5px', 'color': T.INK_SOFT, 'lineHeight': 1.65,
                    'marginTop': '24px', 'paddingLeft': '20px',
                    'borderLeft': f'2px solid {T.ACCENT}',
                    'maxWidth': '780px',
                },
            ),
        ],
        style={'marginTop': '88px'},
    )

    # Economic value / scale calculator
    economic_section = html.Div(
        [
            section_title(
                'But can\'t a farmer just guess?',
                'Three baselines, one honest answer.',
            ),
            # Comparison table — baselines vs model
            dbc.Row(
                [
                    dbc.Col(_comparison_card(
                        'Persistence', '0%',
                        'catch rate · 301 onset events',
                        'Assumes nothing changes. Blind to any transition by construction.',
                        T.INK_FAINT,
                    ), md=4),
                    dbc.Col(_comparison_card(
                        'Linear trend baseline', '51.5%',
                        'catch rate · 301 onset events',
                        'Flags when soil has been falling steadily for several days. A reasonable human heuristic — '
                        'but it also raises 483 false warnings to get there.',
                        T.SIGNAL,
                    ), md=4),
                    dbc.Col(_comparison_card(
                        'XGBoost (best-F1)', '98.7%',
                        'catch rate · 301 onset events',
                        'Trained on soil history, weather, station context. Catches 297 of 301 events at the best-F1 '
                        'threshold (PR-AUC 0.82, ROC-AUC 0.95) — nearly doubles the smart baseline.',
                        T.ACCENT,
                    ), md=4),
                ],
                className='gx-4 gy-4',
            ),
            html.Div(style={'height': '56px'}),
            section_title(
                'Scale it up',
                'One farm is $66/year. The real question is what happens at scale.',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                'Number of farms using the system',
                                style={'fontSize': '14px', 'color': T.INK_SOFT, 'marginBottom': '12px'},
                            ),
                            dcc.Slider(
                                id='econ-farms-slider',
                                min=0, max=5, value=2, step=1,
                                marks={
                                    0: '100',
                                    1: '1,000',
                                    2: '10,000',
                                    3: '100,000',
                                    4: '1M',
                                    5: '3M',
                                },
                                tooltip={'always_visible': False},
                            ),
                            html.Div(
                                'Based on 1.5 ha average farm, 1.6 drought onset events/year, '
                                'FAO maize yield-response factor (Ky = 1.25).',
                                style={'fontSize': '12px', 'color': T.INK_FAINT,
                                       'lineHeight': 1.5, 'marginTop': '20px'},
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(html.Div(id='econ-result'), md=7),
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
        page_title('Onset and duration', 'The core drought-warning results',
                   'Seven-day drought-onset detection, the new drought-state scanner, and the trade-offs behind false alarms.'),
        onset_section,
        duration_section,
        economic_section,
        horizon_section,
        wabl_section,
    ])


# ── CONCLUSIONS ──────────────────────────────────────────────────────

def page_conclusions():
    findings = [
        ('01', 'The warning horizon is practical', 'At a 7-day horizon with a 24-hour onset window, XGBoost catches 297 of 301 drought events and Random Forest catches 296.'),
        ('02', 'The baseline is not enough', 'A linear drying-trend baseline catches only 52% of events in the same setup, while generating more false warning events than either ML model.'),
        ('03', 'The system now estimates persistence', 'A new drought-state model can scan future 24-hour windows after an onset warning, forming the basis for short, moderate, long, or extended duration estimates.'),
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
                'letterSpacing': '-0.01em', 'textAlign': 'center', 'marginBottom': '24px',
            }),
            html.Div(
                [
                    html.Img(
                        src='/assets/datafarmers.png',
                        alt='The ENGG2112 Data Farmers',
                        style={'width': '100%', 'display': 'block', 'borderRadius': '4px'},
                    ),
                    # Zones sized to each person's actual position; height covers upper bodies only
                    html.Div([html.Div([
                        html.Span('James Robison', className='member-name'),
                        html.Span('Project Lead', className='member-role'),
                    ], className='member-card')], className='team-zone',
                        style={'left': '9%', 'width': '22%', 'height': '60%'}),
                    html.Div([html.Div([
                        html.Span('Oscar Everett', className='member-name'),
                        html.Span('ML Engineer', className='member-role'),
                    ], className='member-card')], className='team-zone',
                        style={'left': '30%', 'width': '21%', 'height': '60%'}),
                    html.Div([html.Div([
                        html.Span('Angus Wyles', className='member-name'),
                        html.Span('Domain Researcher', className='member-role'),
                    ], className='member-card')], className='team-zone',
                        style={'left': '46%', 'width': '25%', 'height': '60%'}),
                    html.Div([html.Div([
                        html.Span('Byron Evans', className='member-name'),
                        html.Span('Data Engineer', className='member-role'),
                    ], className='member-card')], className='team-zone',
                        style={'left': '63%', 'width': '32%', 'height': '60%'}),
                ],
                style={'position': 'relative', 'maxWidth': '700px', 'margin': '0 auto'},
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
    ('home',        'Drought Warning'),
    ('results',     'Onset + Duration'),
    ('demo',        'Moisture Forecast'),
    ('data',        'The Data'),
    ('models',      'Model Details'),
    ('conclusions', 'Conclusions'),
]


def sidebar():
    nav_links = []
    for idx, (page_id, nav_label) in enumerate(NAV_ITEMS):
        num = f'0{idx + 1}'
        nav_links.append(
            html.Div(
                [
                    html.Span(num, style={
                        'fontFamily': T.FONT_MONO, 'fontSize': '10px',
                        'color': T.INK_FAINT, 'marginRight': '14px',
                        'letterSpacing': '0.06em', 'fontWeight': 500,
                    }),
                    html.Span(nav_label, style={'fontFamily': T.FONT}),
                ],
                id={'type': 'nav-link', 'page': page_id},
                n_clicks=0,
                className='nav-link-item',
                style={
                    'padding': '13px 32px',
                    'fontSize': '14px',
                    'color': T.INK_SOFT,
                    'cursor': 'pointer',
                    'fontWeight': 500,
                    'display': 'flex',
                    'alignItems': 'center',
                },
            )
        )

    return html.Div(
        [
            # Wordmark
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(className='pulse-dot', style={
                                'background': T.OK, 'marginRight': '10px',
                                'verticalAlign': 'middle', 'width': '6px', 'height': '6px',
                            }),
                            html.Span('LIVE', style={
                                'fontFamily': T.FONT_MONO, 'fontSize': '10px',
                                'letterSpacing': '0.18em', 'color': T.OK, 'fontWeight': 600,
                                'verticalAlign': 'middle',
                            }),
                        ],
                        style={'marginBottom': '18px'},
                    ),
                    html.Div('Drought', className='display', style={
                        'fontSize': '28px', 'color': T.INK,
                        'lineHeight': 1, 'letterSpacing': '-0.025em',
                    }),
                    html.Div([
                        html.Span('warning', style={
                            'fontFamily': T.FONT_DISPLAY,
                            'fontStyle': 'italic',
                            'fontWeight': 400,
                            'color': T.ACCENT,
                            'fontSize': '28px',
                            'letterSpacing': '-0.02em',
                        }),
                    ], style={'lineHeight': 1, 'marginTop': '2px'}),
                    html.Div('System', style={
                        'fontFamily': T.FONT_MONO,
                        'fontSize': '11px',
                        'letterSpacing': '0.22em',
                        'color': T.INK_FAINT,
                        'marginTop': '14px',
                        'textTransform': 'uppercase',
                    }),
                ],
                style={'padding': '40px 32px 44px 32px', 'borderBottom': f'1px solid {T.LINE_SOFT}'},
            ),

            # Nav block
            html.Div(nav_links, style={'paddingTop': '24px'}),

            # Footer brand
            html.Div(
                [
                    html.Div(className='rule', style={'marginBottom': '20px'}),
                    html.Div('ENGG2112', style={
                        'fontFamily': T.FONT_MONO,
                        'fontSize': '10px',
                        'letterSpacing': '0.22em',
                        'color': T.INK_FAINT,
                        'fontWeight': 600,
                    }),
                    html.Div('Data Farmers', style={
                        'fontSize': '13px', 'color': T.INK_SOFT, 'marginTop': '6px',
                        'fontWeight': 500,
                    }),
                    html.Div('Sydney · MMXXVI', style={
                        'fontFamily': T.FONT_MONO,
                        'fontSize': '10px', 'color': T.INK_FAINT, 'marginTop': '3px',
                        'letterSpacing': '0.08em',
                    }),
                ],
                style={'position': 'absolute', 'bottom': '32px', 'left': '32px', 'right': '32px'},
            ),
        ],
        style={
            'position': 'fixed', 'left': 0, 'top': 0, 'bottom': 0,
            'width': '252px',
            'background': T.PANEL,
            'borderRight': f'1px solid {T.LINE}',
            'fontFamily': T.FONT,
            'boxShadow': '1px 0 0 rgba(40, 30, 20, 0.02)',
            'overflow': 'hidden',
        },
    )


app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=JetBrains+Mono:wght@400;500&display=swap',
    ],
    suppress_callback_exceptions=True,
    title='Drought Warning System',
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
            :root {
                --ink: ''' + T.INK + ''';
                --ink-soft: ''' + T.INK_SOFT + ''';
                --ink-faint: ''' + T.INK_FAINT + ''';
                --line: ''' + T.LINE + ''';
                --line-soft: ''' + T.LINE_SOFT + ''';
                --panel: ''' + T.PANEL + ''';
                --panel-raised: ''' + T.PANEL_RAISED + ''';
                --bg: ''' + T.BG + ''';
                --bg-deep: ''' + T.BG_DEEP + ''';
                --accent: ''' + T.ACCENT + ''';
                --accent-deep: ''' + T.ACCENT_DEEP + ''';
                --accent-soft: ''' + T.ACCENT_SOFT + ''';
                --accent-glow: ''' + T.ACCENT_GLOW + ''';
                --signal: ''' + T.SIGNAL + ''';
                --shadow-sm: ''' + T.SHADOW_SM + ''';
                --shadow-md: ''' + T.SHADOW_MD + ''';
                --shadow-lg: ''' + T.SHADOW_LG + ''';
                --ease-out: ''' + T.EASE_OUT + ''';
                --ease-inout: ''' + T.EASE_INOUT + ''';
                --ease-drawer: ''' + T.EASE_DRAWER + ''';
                --dur-fast: ''' + T.DUR_FAST + ''';
                --dur-base: ''' + T.DUR_BASE + ''';
                --dur-slow: ''' + T.DUR_SLOW + ''';
                --font: ''' + T.FONT + ''';
                --font-display: ''' + T.FONT_DISPLAY + ''';
                --font-mono: ''' + T.FONT_MONO + ''';
            }

            * { box-sizing: border-box; }
            html, body { margin: 0; padding: 0; background: var(--bg); }
            body {
                font-family: var(--font);
                color: var(--ink);
                line-height: 1.5;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                text-rendering: optimizeLegibility;
                font-feature-settings: "cv11", "ss01", "ss03";
            }

            /* The whole content surface fades up on first mount.
               `forwards` keeps the final state even if the animation never completes. */
            #page-content {
                animation: pageEnter 520ms var(--ease-out) forwards;
            }
            @keyframes pageEnter {
                0%   { opacity: 0; transform: translateY(8px); filter: blur(2px); }
                100% { opacity: 1; transform: translateY(0);   filter: blur(0); }
            }

            /* Reveal utility — children stagger in */
            .reveal { animation: reveal 600ms var(--ease-out) both; }
            .reveal-1 { animation-delay: 40ms; }
            .reveal-2 { animation-delay: 90ms; }
            .reveal-3 { animation-delay: 140ms; }
            .reveal-4 { animation-delay: 190ms; }
            .reveal-5 { animation-delay: 240ms; }
            .reveal-6 { animation-delay: 290ms; }
            @keyframes reveal {
                0%   { opacity: 0; transform: translateY(10px); }
                100% { opacity: 1; transform: translateY(0); }
            }

            /* Sidebar — refined hover with sliding indicator */
            .nav-link-item {
                position: relative;
                transition: color var(--dur-base) var(--ease-out),
                            background var(--dur-base) var(--ease-out),
                            padding-left var(--dur-base) var(--ease-out);
            }
            .nav-link-item::before {
                content: "";
                position: absolute;
                left: 0; top: 50%;
                width: 2px; height: 0;
                background: var(--accent);
                transform: translateY(-50%);
                transition: height var(--dur-base) var(--ease-out);
            }
            @media (hover: hover) and (pointer: fine) {
                .nav-link-item:hover {
                    color: var(--ink) !important;
                    background: var(--line-soft);
                    padding-left: 36px !important;
                }
                .nav-link-item:hover::before { height: 14px; }
            }
            .nav-link-item:active { transform: scale(0.99); }
            .nav-link-item.active {
                color: var(--accent) !important;
                font-weight: 600 !important;
                background: var(--bg) !important;
                padding-left: 36px !important;
            }
            .nav-link-item.active::before { height: 22px; }

            /* Cards — quiet lift on hover (only where opted-in via .lift) */
            .lift {
                transition: transform var(--dur-base) var(--ease-out),
                            box-shadow var(--dur-base) var(--ease-out),
                            border-color var(--dur-base) var(--ease-out);
                will-change: transform;
            }
            @media (hover: hover) and (pointer: fine) {
                .lift:hover {
                    transform: translateY(-2px);
                    box-shadow: var(--shadow-md);
                    border-color: #D6D0C5;
                }
            }

            /* Pressable feedback */
            .press {
                transition: transform var(--dur-fast) var(--ease-out);
            }
            .press:active { transform: scale(0.985); }

            /* Number ticker — looks alive without animating constantly */
            .stat-value {
                font-feature-settings: "tnum" 1, "ss01" 1;
                letter-spacing: -0.025em;
            }

            /* Display serif headings */
            .display {
                font-family: var(--font-display);
                font-weight: 500;
                font-optical-sizing: auto;
                font-variation-settings: "opsz" 96, "SOFT" 50;
                letter-spacing: -0.028em;
                line-height: 1.02;
            }
            .display em {
                font-style: italic;
                font-weight: 400;
                color: var(--accent);
                font-variation-settings: "opsz" 144, "SOFT" 100;
            }

            /* Eyebrow / mono tag */
            .eyebrow {
                font-family: var(--font-mono);
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                color: var(--ink-faint);
            }
            .eyebrow-accent { color: var(--accent); }

            /* Accent pulse — used sparingly on the hero status dot */
            .pulse-dot {
                display: inline-block;
                width: 8px; height: 8px;
                border-radius: 50%;
                background: var(--accent);
                position: relative;
                box-shadow: 0 0 0 0 var(--accent-glow);
                animation: pulseDot 2.4s var(--ease-inout) infinite;
            }
            @keyframes pulseDot {
                0%   { box-shadow: 0 0 0 0    var(--accent-glow); }
                70%  { box-shadow: 0 0 0 12px rgba(184, 82, 31, 0); }
                100% { box-shadow: 0 0 0 0    rgba(184, 82, 31, 0); }
            }

            /* Slider styling */
            .rc-slider-track { background: var(--accent) !important; transition: background var(--dur-base) var(--ease-out); }
            .rc-slider-rail  { background: var(--line) !important; }
            .rc-slider-handle {
                border-color: var(--accent) !important;
                background: var(--panel-raised) !important;
                box-shadow: var(--shadow-sm) !important;
                transition: box-shadow var(--dur-base) var(--ease-out),
                            transform var(--dur-base) var(--ease-out) !important;
            }
            .rc-slider-handle:hover {
                box-shadow: 0 0 0 6px var(--accent-glow), var(--shadow-sm) !important;
            }
            .rc-slider-handle:active,
            .rc-slider-handle:focus {
                box-shadow: 0 0 0 8px var(--accent-glow), var(--shadow-md) !important;
                transform: scale(1.08) translate(-50%, -50%) !important;
            }
            .rc-slider-dot-active { border-color: var(--accent) !important; }
            .rc-slider-mark-text {
                color: var(--ink-faint) !important;
                font-size: 11px !important;
                font-family: var(--font-mono) !important;
                letter-spacing: 0.04em;
            }

            /* Dropdown styling */
            .Select-control, .dash-dropdown .Select-control {
                border: 1px solid var(--line) !important;
                border-radius: 4px !important;
                background: var(--panel-raised) !important;
                min-height: 42px !important;
                transition: border-color var(--dur-base) var(--ease-out),
                            box-shadow var(--dur-base) var(--ease-out);
            }
            .Select-control:hover { border-color: var(--ink-faint) !important; }
            .is-focused:not(.is-open) > .Select-control {
                border-color: var(--accent) !important;
                box-shadow: 0 0 0 4px var(--accent-glow) !important;
            }
            .Select-menu-outer {
                border-radius: 4px !important;
                border-color: var(--line) !important;
                box-shadow: var(--shadow-md) !important;
                margin-top: 4px !important;
                animation: dropdown 200ms var(--ease-out) both;
                transform-origin: top center;
            }
            @keyframes dropdown {
                0%   { opacity: 0; transform: scale(0.97) translateY(-4px); }
                100% { opacity: 1; transform: scale(1)    translateY(0); }
            }
            .Select-option { transition: background var(--dur-fast) var(--ease-out); }
            .VirtualizedSelectFocusedOption { background: var(--bg) !important; color: var(--ink) !important; }

            /* Buttons */
            button, .btn {
                transition: transform var(--dur-fast) var(--ease-out),
                            background var(--dur-base) var(--ease-out),
                            box-shadow var(--dur-base) var(--ease-out) !important;
            }
            button:active, .btn:active { transform: scale(0.975); }
            button:focus-visible, .btn:focus-visible {
                outline: none;
                box-shadow: 0 0 0 4px var(--accent-glow) !important;
            }

            /* Selection */
            ::selection { background: var(--accent-soft); color: var(--ink); }

            /* Hairline underline used in section headers */
            .rule {
                height: 1px;
                background: linear-gradient(to right, var(--line) 0%, var(--line) 70%, transparent 100%);
            }

            /* Plotly graph chrome — drop the modebar shadow on hover */
            .js-plotly-plot .plotly { transition: opacity var(--dur-base) var(--ease-out); }

            /* Scroll-triggered reveal — content is ALWAYS fully visible.
               JS only triggers a subtle translate when it enters view. No opacity dimming. */
            .reveal-on-scroll {
                transform: translateY(0);
                transition: transform 700ms var(--ease-out);
            }
            html.js-ready .reveal-on-scroll:not(.revealed) {
                transform: translateY(12px);
            }
            .reveal-on-scroll-stagger > * {
                transform: translateY(0);
                transition: transform 600ms var(--ease-out);
            }
            html.js-ready .reveal-on-scroll-stagger:not(.revealed) > * {
                transform: translateY(10px);
            }
            .reveal-on-scroll-stagger.revealed > *:nth-child(1) { transition-delay: 0ms; }
            .reveal-on-scroll-stagger.revealed > *:nth-child(2) { transition-delay: 70ms; }
            .reveal-on-scroll-stagger.revealed > *:nth-child(3) { transition-delay: 140ms; }
            .reveal-on-scroll-stagger.revealed > *:nth-child(4) { transition-delay: 210ms; }
            .reveal-on-scroll-stagger.revealed > *:nth-child(5) { transition-delay: 280ms; }
            .reveal-on-scroll-stagger.revealed > *:nth-child(6) { transition-delay: 350ms; }

            /* Cursor spotlight — radial highlight that tracks the mouse over .lift cards */
            .lift {
                position: relative;
                isolation: isolate;
            }
            .lift::after {
                content: "";
                position: absolute;
                inset: 0;
                border-radius: inherit;
                pointer-events: none;
                opacity: 0;
                background: radial-gradient(
                    260px circle at var(--spot-x, 50%) var(--spot-y, 50%),
                    rgba(184, 82, 31, 0.10),
                    transparent 60%
                );
                transition: opacity var(--dur-base) var(--ease-out);
                z-index: 0;
            }
            @media (hover: hover) and (pointer: fine) {
                .lift:hover::after { opacity: 1; }
            }
            .lift > * { position: relative; z-index: 1; }

            /* Nav links get a faster transform transition so the magnetic pull feels snappy */
            .nav-link-item { transition-property: color, background, padding-left, transform; }

            /* Verdict panels — slide + blur lift on appear */
            [data-verdict] {
                animation: verdictIn 540ms var(--ease-drawer) both;
            }
            @keyframes verdictIn {
                0%   { opacity: 0; transform: translateY(10px) scale(0.985); filter: blur(3px); }
                60%  { filter: blur(0); }
                100% { opacity: 1; transform: translateY(0)    scale(1); filter: blur(0); }
            }

            /* Plotly path draw-on — gives line charts a 'being plotted' feel */
            .js-plotly-plot .scatterlayer .trace .js-line {
                stroke-linecap: round;
            }

            /* Scrollbar — subtle, on-brand */
            ::-webkit-scrollbar { width: 10px; height: 10px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb {
                background: var(--line);
                border-radius: 10px;
                border: 2px solid var(--bg);
            }
            ::-webkit-scrollbar-thumb:hover { background: var(--ink-faint); }

            /* prefers-reduced-motion — keep fades, drop movement */
            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    animation-duration: 0.001ms !important;
                    transition-duration: 0.001ms !important;
                }
                #page-content { animation: none !important; }
            }
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
                    'padding': '80px 88px',
                    'minHeight': '100vh',
                    'borderRadius': '8px',
                    'boxShadow': T.SHADOW_LG,
                    'border': f'1px solid {T.LINE_SOFT}',
                },
            ),
            style={
                'marginLeft': '252px',
                'padding': '28px 36px 56px 36px',
                'maxWidth': '1320px',
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
    Input('econ-farms-slider', 'value'),
)
def econ_calc(slider_pos):
    # Slider positions map to farm counts
    farm_counts = [100, 1_000, 10_000, 100_000, 1_000_000, 3_000_000]
    n_farms = farm_counts[int(slider_pos)]

    # Constants (FAO-based)
    avg_farm_ha        = 1.5      # average East Africa smallholder farm
    events_per_year    = 1.6      # onset events per station per year (from test 10 data)
    per_event_per_ha   = 28.80    # FAO maize: Ky=1.25, 8% yield loss @ $360/ha revenue
    model_catch        = 0.97     # from test 12 dedicated classifier
    baseline_catch     = 0.50     # linear trend baseline (test 13)
    revenue_per_ha     = 360      # $/ha/season (maize, East Africa)

    value_per_farm     = events_per_year * model_catch * per_event_per_ha * avg_farm_ha
    baseline_per_farm  = events_per_year * baseline_catch * per_event_per_ha * avg_farm_ha
    season_revenue     = revenue_per_ha * avg_farm_ha
    pct_revenue        = value_per_farm / season_revenue * 100

    total_model        = value_per_farm * n_farms
    total_baseline     = baseline_per_farm * n_farms
    extra_vs_baseline  = (value_per_farm - baseline_per_farm) * n_farms

    def _fmt(v):
        if v >= 1_000_000: return f'${v/1_000_000:.1f}M'
        if v >= 1_000:     return f'${v/1_000:.0f}K'
        return f'${v:.0f}'

    farm_label = f'{n_farms:,}'

    return dbc.Row(
        [
            dbc.Col(html.Div(
                [
                    html.Div('Per farm, per year', style={
                        'fontSize': '11px', 'letterSpacing': '0.12em', 'fontWeight': 600,
                        'color': T.INK_FAINT, 'marginBottom': '14px',
                    }),
                    html.Div(f'${value_per_farm:.0f}', style={
                        'fontSize': '48px', 'fontWeight': 700, 'letterSpacing': '-0.03em',
                        'color': T.ACCENT, 'lineHeight': 1, 'marginBottom': '6px',
                    }),
                    html.Div(f'{pct_revenue:.0f}% of seasonal revenue', style={
                        'fontSize': '14px', 'color': T.INK_SOFT,
                    }),
                ],
                style={'borderLeft': f'3px solid {T.ACCENT}', 'paddingLeft': '20px'},
            ), md=4),
            dbc.Col(html.Div(
                [
                    html.Div(f'Across {farm_label} farms', style={
                        'fontSize': '11px', 'letterSpacing': '0.12em', 'fontWeight': 600,
                        'color': T.INK_FAINT, 'marginBottom': '14px',
                    }),
                    html.Div(_fmt(total_model), style={
                        'fontSize': '48px', 'fontWeight': 700, 'letterSpacing': '-0.03em',
                        'color': T.ACCENT, 'lineHeight': 1, 'marginBottom': '6px',
                    }),
                    html.Div('in yield protected per year', style={
                        'fontSize': '14px', 'color': T.INK_SOFT,
                    }),
                ],
                style={'borderLeft': f'3px solid {T.ACCENT}', 'paddingLeft': '20px'},
            ), md=4),
            dbc.Col(html.Div(
                [
                    html.Div('Vs smart baseline', style={
                        'fontSize': '11px', 'letterSpacing': '0.12em', 'fontWeight': 600,
                        'color': T.INK_FAINT, 'marginBottom': '14px',
                    }),
                    html.Div(_fmt(extra_vs_baseline), style={
                        'fontSize': '48px', 'fontWeight': 700, 'letterSpacing': '-0.03em',
                        'color': T.SIGNAL, 'lineHeight': 1, 'marginBottom': '6px',
                    }),
                    html.Div('extra protected vs linear trend baseline', style={
                        'fontSize': '14px', 'color': T.INK_SOFT,
                    }),
                ],
                style={'borderLeft': f'3px solid {T.SIGNAL}', 'paddingLeft': '20px'},
            ), md=4),
        ],
        className='gx-4',
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
