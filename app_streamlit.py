"""
Soil Moisture Forecaster — Streamlit demo app.
ENGG2112 Data Farmers.

Launch:
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


HERE = Path(__file__).parent
APP_ASSETS = HERE / 'app_assets'
IMAGES_DIR = HERE / 'images'

THRESHOLD = 0.30
HORIZON_HOURS = 72

CFG_PALETTE = {
    'green':   '#2E7D32',
    'amber':   '#F9A825',
    'red':     '#C62828',
    'blue':    '#1565C0',
    'orange':  '#E65100',
    'grey':    '#757575',
    'light':   '#F1F8E9',
    'panel':   '#F7F9FC',
    'persist': '#9E9E9E',
    'xgb':     '#E65100',
    'actual':  '#1565C0',
}


# ────────────────────────────────────────────────────────────────────────────
# Page config + global CSS
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Soil Moisture Forecaster',
    page_icon='🌱',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    /* Tighten the main container */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px; }
    /* Headings */
    h1 { font-weight: 700; color: #1B5E20; letter-spacing: -0.5px; }
    h2 { font-weight: 600; color: #2E7D32; margin-top: 0.5rem; }
    h3 { color: #424242; font-weight: 600; }
    /* Cards */
    .stat-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #E8EAED;
        height: 100%;
    }
    .stat-card .label { color: #5F6368; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-card .value { color: #1B5E20; font-size: 2.2rem; font-weight: 700; line-height: 1.1; margin-top: 0.3rem;}
    .stat-card .sub   { color: #757575; font-size: 0.9rem; margin-top: 0.3rem; }
    /* Hero box */
    .hero {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 60%, #43A047 100%);
        color: white; border-radius: 14px; padding: 2.5rem 2.5rem; margin-bottom: 1rem;
    }
    .hero h1 { color: white !important; margin-bottom: 0.6rem; font-size: 2.4rem; }
    .hero p  { color: rgba(255,255,255,0.95); font-size: 1.05rem; margin: 0; max-width: 850px; }
    /* Verdict box (live demo) */
    .verdict {
        border-radius: 14px; padding: 1.6rem; margin: 0.6rem 0;
        font-size: 1.15rem; line-height: 1.4; font-weight: 500;
    }
    .verdict.green { background: #E8F5E9; border-left: 6px solid #2E7D32; color: #1B5E20; }
    .verdict.amber { background: #FFF8E1; border-left: 6px solid #F9A825; color: #E65100; }
    .verdict.red   { background: #FFEBEE; border-left: 6px solid #C62828; color: #B71C1C; }
    .verdict .big  { font-size: 1.6rem; font-weight: 700; }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F1F8E9 0%, #ffffff 100%);
    }
    /* Subtle pill labels */
    .pill {
        display: inline-block; padding: 0.2rem 0.7rem; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.3px;
    }
    .pill.green { background: #C8E6C9; color: #1B5E20; }
    .pill.blue  { background: #BBDEFB; color: #0D47A1; }
    .pill.grey  { background: #ECEFF1; color: #455A64; }
    /* Metric layout: kill stmetric border noise */
    .stMetric { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Data loaders (cached)
# ────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load(APP_ASSETS / 'xgboost_t72h.joblib')


@st.cache_data
def load_feature_cols():
    with open(APP_ASSETS / 'feature_cols.json') as f:
        return json.load(f)


@st.cache_data
def load_test_predictions():
    df = pd.read_parquet(APP_ASSETS / 'test_predictions.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


@st.cache_data
def load_sample_features():
    df = pd.read_parquet(APP_ASSETS / 'sample_features.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


@st.cache_data
def load_station_meta():
    df = pd.read_parquet(APP_ASSETS / 'station_meta.parquet')
    df['country'] = df.apply(_country_from_latlon, axis=1)
    df['short_name'] = (
        df['station'].str.split('/', n=1).str[1]
        .str.replace('_', ' ', regex=False)
    )
    return df


@st.cache_data
def load_feature_importance():
    return pd.read_parquet(APP_ASSETS / 'feature_importance.parquet')


@st.cache_data
def load_raw_timeseries():
    df = pd.read_parquet(APP_ASSETS / 'raw_timeseries.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


@st.cache_data
def load_weather_correlations():
    return pd.read_parquet(APP_ASSETS / 'weather_correlations.parquet')


@st.cache_data
def load_horizon_metrics():
    """Read horizon_comparison.csv from the latest test folder available."""
    for candidate in ['test 10 - current models rerun',
                      'test 9 - TFT fix + threshold opt',
                      'test 7 - ET0 dropout tuned XGB']:
        path = IMAGES_DIR / candidate / 'horizon_comparison.csv'
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_onset_analysis():
    for candidate in ['test 10 - current models rerun',
                      'test 9 - TFT fix + threshold opt']:
        path = IMAGES_DIR / candidate / 'onset_analysis.csv'
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_model_metrics():
    for candidate in ['test 10 - current models rerun',
                      'test 9 - TFT fix + threshold opt',
                      'test 7 - ET0 dropout tuned XGB']:
        path = IMAGES_DIR / candidate / 'metrics.csv'
        if path.exists():
            df = pd.read_csv(path)
            # Normalise column name R²/R2
            df = df.rename(columns={'R²': 'R2'})
            return df
    return pd.DataFrame()


@st.cache_data
def load_weather_ablation():
    path = IMAGES_DIR / 'test 11 - weather vs no wx t+120h' / 'weather_ablation_four_models.csv'
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _country_from_latlon(row):
    """Rough country attribution from coordinates (covers Kenya/Uganda/Rwanda)."""
    lat, lon = row['latitude'], row['longitude']
    # Rwanda: lat -3 to -1, lon 28.8 to 31
    if -3.0 <= lat <= -1.0 and 28.5 <= lon <= 31.0:
        return 'Rwanda'
    # Uganda: lat -1.5 to 4.5, lon 29.5 to 35
    if -1.5 < lat <= 4.5 and 29.5 <= lon < 35.5:
        return 'Uganda'
    # Kenya: rest of East Africa lat range
    if -5.0 <= lat <= 5.5 and 33.0 <= lon <= 42.5:
        return 'Kenya'
    return 'Other'


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def stat_card(label, value, sub=None, value_color=None):
    color = value_color or CFG_PALETTE['blue']
    sub_html = f'<div class="sub">{sub}</div>' if sub else ''
    st.markdown(
        f'<div class="stat-card">'
        f'  <div class="label">{label}</div>'
        f'  <div class="value" style="color:{color}">{value}</div>'
        f'  {sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def verdict_box(level, headline, body):
    st.markdown(
        f'<div class="verdict {level}">'
        f'  <div class="big">{headline}</div>'
        f'  <div style="margin-top:0.5rem">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def make_station_map(stations_df, selected_station=None, color_field='country'):
    df = stations_df.copy()
    df['marker_size'] = np.where(df['station'] == selected_station, 22, 10)
    df['is_selected'] = df['station'] == selected_station

    hover = {
        'short_name': True, 'country': True,
        'network': True, 'elevation_m': ':.0f',
        'n_readings': ':,',
        'latitude': ':.3f', 'longitude': ':.3f',
        'marker_size': False, 'is_selected': False,
    }

    fig = px.scatter_mapbox(
        df,
        lat='latitude', lon='longitude',
        color=color_field,
        size='marker_size', size_max=22,
        hover_name='short_name',
        hover_data=hover,
        color_discrete_map={
            'Kenya':   '#E65100',
            'Uganda':  '#1565C0',
            'Rwanda':  '#2E7D32',
            'COSMOS':  '#7B1FA2',
            'TAHMO':   '#00838F',
            'Other':   '#757575',
        },
        zoom=4.6, height=520,
    )
    fig.update_layout(
        mapbox_style='carto-positron',
        mapbox_center=dict(lat=0.5, lon=36),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=0.02, xanchor='center', x=0.5,
                    bgcolor='rgba(255,255,255,0.85)'),
    )
    return fig


def classify_recommendation(sm_now, sm_pred, threshold=THRESHOLD):
    """Return (level, headline, body) for the traffic light."""
    if sm_pred < threshold and sm_now < threshold:
        return ('red',
                '🔴 Drought stress now & next 3 days',
                f'Soil moisture is currently {sm_now:.3f} m³/m³ and predicted to '
                f'remain at {sm_pred:.3f}. The drought threshold is {threshold}. '
                'Crops are already water-stressed — irrigate immediately if water is available.')
    if sm_pred < threshold and sm_now >= threshold:
        gap = sm_now - sm_pred
        return ('red',
                '🔴 Drought stress incoming within 3 days',
                f'Soil moisture is healthy now ({sm_now:.3f} m³/m³) but is forecast to '
                f'drop to {sm_pred:.3f} — below the {threshold} drought threshold. '
                f'Plan to irrigate in the next 24–48 hours. This is the most valuable '
                'warning the model provides.')
    if sm_pred < threshold + 0.04 and sm_now >= threshold:
        return ('amber',
                '🟡 Borderline — watch closely',
                f'Soil moisture is healthy now ({sm_now:.3f} m³/m³) but predicted to '
                f'fall toward {sm_pred:.3f}. Close to the {threshold} drought threshold. '
                'Check the soil tomorrow and have irrigation ready.')
    if sm_pred < threshold + 0.04:
        return ('amber',
                '🟡 Currently dry, but no further fall',
                f'Soil moisture is {sm_now:.3f} now and predicted {sm_pred:.3f}. '
                'Already near the drought threshold — irrigation likely needed this week.')
    return ('green',
            '🟢 Soil moisture healthy',
            f'Currently {sm_now:.3f} m³/m³, predicted {sm_pred:.3f} m³/m³ in 3 days. '
            f'Well above the {threshold} drought threshold. No irrigation needed.')


# ────────────────────────────────────────────────────────────────────────────
# Sidebar nav
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('### 🌱 Soil Moisture')
    st.markdown('#### **Forecaster**')
    st.markdown('<span class="pill green">ENGG2112</span> <span class="pill blue">Data Farmers</span>', unsafe_allow_html=True)
    st.markdown('---')

    page = st.radio(
        'Navigate',
        options=[
            '🏠 Home',
            '📊 The Data',
            '🤖 The Models',
            '🎯 Live Demo',
            '🏆 Results',
            '💡 Conclusions',
        ],
        label_visibility='collapsed',
    )

    st.markdown('---')
    st.caption('48 stations · 3 countries · 1.1M soil readings · 4 models trained')


# ────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ────────────────────────────────────────────────────────────────────────────

if page == '🏠 Home':
    st.markdown(
        '<div class="hero">'
        '<h1>Will the soil be dry on Thursday?</h1>'
        '<p>A machine-learning system that warns smallholder farmers in East Africa '
        'three days before their soil dries below the drought stress threshold — so '
        'they irrigate <em>in advance</em>, not after their crops have already wilted.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write('')
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card('Sensor stations', '48', 'across Kenya, Uganda, Rwanda', CFG_PALETTE['green'])
    with c2: stat_card('Hourly readings', '1.1M', 'quality-flagged "Good" only', CFG_PALETTE['blue'])
    with c3: stat_card('Forecast horizon', 'up to 7 days', 'best results at 1–3 days', CFG_PALETTE['orange'])
    with c4: stat_card('Drought events caught', '57%', 'vs persistence baseline at 27%', CFG_PALETTE['red'])

    st.write('')
    st.markdown('### The problem in one picture')

    # Pick an example station with onset event to highlight
    preds = load_test_predictions()
    meta = load_station_meta()

    # find a station with a nice mix of above/below threshold for the demo plot
    s_sample = preds[preds['station'] == 'TAHMO/Kibanda_Hydromet']
    if s_sample.empty:
        s_sample = preds[preds['station'] == preds['station'].iloc[0]]
    s_sample = s_sample.sort_values('datetime').head(720)  # ~1 month

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s_sample['datetime'], y=s_sample['y_true'],
        mode='lines', name='Actual soil moisture',
        line=dict(color=CFG_PALETTE['actual'], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=s_sample['datetime'], y=s_sample['y_pred_xgb'],
        mode='lines', name='Model forecast (t+72h)',
        line=dict(color=CFG_PALETTE['xgb'], width=2.5, dash='dot'),
    ))
    fig.add_hline(
        y=THRESHOLD, line_dash='dash', line_color=CFG_PALETTE['red'],
        annotation_text=f'Drought threshold  ({THRESHOLD} m³/m³)',
        annotation_position='top right',
        annotation_font_color=CFG_PALETTE['red'],
    )
    fig.add_hrect(y0=0, y1=THRESHOLD, fillcolor='red', opacity=0.06, line_width=0)
    fig.update_layout(
        title='Example: 30 days at one station — model predicts moisture 3 days ahead',
        xaxis_title=None, yaxis_title='Soil moisture (m³/m³)',
        height=440, margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='right', x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=13),
    )
    fig.update_yaxes(gridcolor='#EEEEEE', zeroline=False)
    fig.update_xaxes(gridcolor='#EEEEEE')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Why this matters')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '#### 🌾 The farmer\'s dilemma\n'
            'Most smallholder farmers in East Africa irrigate **reactively** — once they see '
            'wilting leaves, the yield damage is already done. A 3-day warning means '
            'water can be applied *before* stress sets in.'
        )
    with col2:
        st.markdown(
            '#### 📡 The data we have\n'
            '**ISMN** soil sensors give hourly ground-truth moisture readings. '
            '**NASA POWER** satellites give daily weather everywhere on Earth. '
            'Combined, they cover places no traditional weather station does.'
        )
    with col3:
        st.markdown(
            '#### 🤖 What the model adds\n'
            'A naive "tomorrow looks like today" forecast catches only **27%** '
            'of upcoming drought events. Our XGBoost model catches **57%** — '
            'twice as many, with fewer false alarms.'
        )

    st.info(
        '**Use the sidebar →** start with "The Data" to see the sensor network, '
        '"The Models" for the architecture, or jump straight to **🎯 Live Demo** '
        'to make a forecast yourself.'
    )


# ────────────────────────────────────────────────────────────────────────────
# PAGE: THE DATA
# ────────────────────────────────────────────────────────────────────────────

elif page == '📊 The Data':
    st.title('The Data')
    st.caption('Two complementary sources: ISMN soil sensors (hourly, ground-truth) and NASA POWER weather (daily, satellite-derived).')

    meta = load_station_meta()
    raw  = load_raw_timeseries()

    tab1, tab2, tab3 = st.tabs(['🗺️ Sensor network', '📈 Station deep-dive', '🌤️ Weather variables'])

    # --- Tab 1: map ---
    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('#### Where are the sensors?')
            st.markdown(
                'Each pin is a soil moisture station. **TAHMO** (46 stations) covers '
                'most of Kenya, Uganda and Rwanda; **COSMOS** (2 stations) uses '
                'cosmic-ray probes for deeper soil readings.'
            )
            color_by = st.radio(
                'Colour by',
                ['country', 'network'],
                horizontal=True,
            )
            st.markdown('---')
            st.markdown('**Network summary**')
            summary = (meta.groupby(color_by)
                       .agg(stations=('station', 'count'),
                            readings=('n_readings', 'sum'))
                       .reset_index())
            summary['readings'] = summary['readings'].map('{:,}'.format)
            st.dataframe(summary, hide_index=True, use_container_width=True)

        with c2:
            fig = make_station_map(meta, color_field=color_by)
            st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: station deep-dive ---
    with tab2:
        st.markdown('#### Inspect any station')
        col_a, col_b = st.columns([1, 2])

        with col_a:
            station_choice = st.selectbox(
                'Choose a station',
                options=meta['station'].tolist(),
                index=meta['station'].tolist().index('TAHMO/Kibanda_Hydromet')
                    if 'TAHMO/Kibanda_Hydromet' in meta['station'].tolist() else 0,
                format_func=lambda s: s.split('/', 1)[1].replace('_', ' '),
            )
            info = meta[meta['station'] == station_choice].iloc[0]

            st.markdown(
                f"**Network:** `{info['network']}`  \n"
                f"**Country:** {info['country']}  \n"
                f"**Location:** {info['latitude']:.3f}°, {info['longitude']:.3f}°  \n"
                f"**Elevation:** {info['elevation_m']:.0f} m  \n"
                f"**Sensor depth:** {info['depth_m']:.2f} m  \n"
                f"**Readings:** {info['n_readings']:,}  \n"
                f"**Period:** {pd.to_datetime(info['start']).date()} → {pd.to_datetime(info['end']).date()}"
            )

        with col_b:
            station_ts = raw[raw['station'] == station_choice].sort_values('datetime')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=station_ts['datetime'], y=station_ts['sm_value'],
                mode='lines', name='Soil moisture',
                line=dict(color=CFG_PALETTE['actual'], width=1.2),
                fill='tozeroy',
                fillcolor='rgba(21, 101, 192, 0.10)',
            ))
            fig.add_hline(
                y=THRESHOLD, line_dash='dash', line_color=CFG_PALETTE['red'],
                annotation_text='Drought threshold',
                annotation_position='top right',
            )
            fig.update_layout(
                title=f"Full history — {info['short_name']}",
                xaxis_title=None, yaxis_title='Soil moisture (m³/m³)',
                height=420, margin=dict(l=10, r=10, t=60, b=10),
                plot_bgcolor='white', paper_bgcolor='white',
            )
            fig.update_yaxes(gridcolor='#EEEEEE')
            st.plotly_chart(fig, use_container_width=True)

        # Below: histogram of readings
        col_c, col_d = st.columns(2)
        with col_c:
            hist = go.Figure()
            hist.add_trace(go.Histogram(
                x=station_ts['sm_value'], nbinsx=50,
                marker=dict(color=CFG_PALETTE['blue'], line=dict(color='white', width=1)),
            ))
            hist.add_vline(x=THRESHOLD, line_dash='dash', line_color=CFG_PALETTE['red'],
                           annotation_text='Threshold', annotation_position='top')
            hist.update_layout(
                title='Distribution of moisture readings',
                xaxis_title='Soil moisture (m³/m³)', yaxis_title='Count',
                height=340, margin=dict(l=10, r=10, t=50, b=10),
                plot_bgcolor='white', showlegend=False,
            )
            hist.update_yaxes(gridcolor='#EEEEEE')
            st.plotly_chart(hist, use_container_width=True)

        with col_d:
            # Monthly average
            station_ts2 = station_ts.copy()
            station_ts2['month'] = station_ts2['datetime'].dt.month
            monthly = station_ts2.groupby('month')['sm_value'].agg(['mean', 'std']).reset_index()
            monthly['month_name'] = monthly['month'].apply(lambda m: pd.Timestamp(2024, m, 1).strftime('%b'))
            seas = go.Figure()
            seas.add_trace(go.Scatter(
                x=monthly['month_name'], y=monthly['mean'],
                mode='lines+markers', line=dict(color=CFG_PALETTE['green'], width=3),
                error_y=dict(type='data', array=monthly['std'], visible=True, color='#A5D6A7'),
                name='Monthly mean ± std',
            ))
            seas.add_hline(y=THRESHOLD, line_dash='dash', line_color=CFG_PALETTE['red'])
            seas.update_layout(
                title='Seasonal pattern',
                xaxis_title=None, yaxis_title='Soil moisture (m³/m³)',
                height=340, margin=dict(l=10, r=10, t=50, b=10),
                plot_bgcolor='white', showlegend=False,
            )
            seas.update_yaxes(gridcolor='#EEEEEE')
            st.plotly_chart(seas, use_container_width=True)

    # --- Tab 3: weather correlations ---
    with tab3:
        st.markdown('#### How much do weather variables actually relate to soil moisture?')
        st.markdown(
            'The model uses 8 weather features: temperature (mean/max/min), humidity, '
            'precipitation, wind, solar radiation, and **ET₀** (reference evapotranspiration). '
            'Below: how each correlates with soil moisture in the raw data.'
        )

        corr = load_weather_correlations()
        if not corr.empty:
            corr_sorted = corr.sort_values('pearson_r', key=lambda s: s.abs(), ascending=True)
            corr_sorted['label'] = corr_sorted['weather_var'].map({
                'T2M': 'Mean temperature',
                'T2M_MAX': 'Max temperature',
                'T2M_MIN': 'Min temperature',
                'RH2M': 'Relative humidity',
                'PRECTOTCORR': 'Precipitation',
                'WS2M': 'Wind speed',
                'ALLSKY_SFC_SW_DWN': 'Solar radiation',
                'ET0': 'Evapotranspiration (ET₀)',
            })
            colors = ['#C62828' if r < 0 else '#2E7D32' for r in corr_sorted['pearson_r']]
            fig = go.Figure(go.Bar(
                x=corr_sorted['pearson_r'], y=corr_sorted['label'],
                orientation='h', marker=dict(color=colors, line=dict(color='white', width=1)),
                text=[f'{v:+.3f}' for v in corr_sorted['pearson_r']],
                textposition='outside',
            ))
            fig.add_vline(x=0, line_color='black', line_width=1)
            fig.update_layout(
                title='Pearson correlation: weather variable ↔ soil moisture',
                xaxis_title='Correlation coefficient (r)',
                yaxis_title=None,
                height=420, margin=dict(l=10, r=30, t=60, b=10),
                plot_bgcolor='white',
            )
            fig.update_xaxes(gridcolor='#EEEEEE', range=[-0.35, 0.25])
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            '**Insight:** No single weather variable has a strong correlation with soil moisture — '
            'the highest is humidity (~0.18). That\'s why a model that combines many weak signals '
            '*and* recent soil moisture history outperforms one that uses any single source alone.'
        )


# ────────────────────────────────────────────────────────────────────────────
# PAGE: THE MODELS
# ────────────────────────────────────────────────────────────────────────────

elif page == '🤖 The Models':
    st.title('The Models')
    st.caption('Four models were trained side-by-side. XGBoost won. Here\'s how each one sees the world.')

    tab1, tab2, tab3 = st.tabs(['🧠 Architecture', '⚙️ Features', '📐 How XGBoost decides'])

    metrics_df = load_model_metrics()

    with tab1:
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown('#### Four architectures, same problem')

            st.markdown(
                '''
**Persistence baseline**
A naive "tomorrow looks like today." Predicts soil moisture in 3 days = soil moisture 24 hours ago.
No machine learning — a sanity check every real model must beat.

**Random Forest**
200 decision trees voting on the answer.
Sees hand-crafted features (lagged moisture, time of day, weather).

**XGBoost** ⭐
Gradient-boosted trees. Each tree corrects the errors of the previous one.
Same features as Random Forest. Best overall performance.

**LSTM**
Recurrent neural network reading the last 5 days of moisture + weather hour by hour.
Has to *learn* the temporal patterns Random Forest and XGBoost get pre-computed.

**TFT (Temporal Fusion Transformer)**
LSTM + multi-head attention + per-station static features (lat/lon/elevation/depth).
The most complex architecture — competitive but not better than XGBoost.
'''
            )

        with col_b:
            st.markdown('#### Performance at t+72h (3 days ahead)')
            if not metrics_df.empty:
                show = metrics_df[['Model', 'MAE', 'R2', 'Recall', 'ROC AUC']].copy()
                show['R2'] = show['R2'].astype(float).round(4)
                show['MAE'] = show['MAE'].astype(float).round(4)
                st.dataframe(
                    show.set_index('Model').style.format({
                        'MAE': '{:.4f}', 'R2': '{:.4f}', 'Recall': '{:.3f}', 'ROC AUC': '{:.3f}',
                    }).background_gradient(subset=['R2', 'Recall', 'ROC AUC'], cmap='Greens')
                    .background_gradient(subset=['MAE'], cmap='Reds_r'),
                    use_container_width=True,
                    height=240,
                )

                # Bar chart: R² across models
                fig = go.Figure()
                colors_map = {
                    'Random Forest': CFG_PALETTE['grey'],
                    'XGBoost':       CFG_PALETTE['orange'],
                    'LSTM':          CFG_PALETTE['blue'],
                    'TFT':           CFG_PALETTE['green'],
                }
                fig.add_trace(go.Bar(
                    x=show['Model'], y=show['R2'].astype(float),
                    marker_color=[colors_map.get(m, '#999') for m in show['Model']],
                    text=[f'{v:.4f}' for v in show['R2'].astype(float)],
                    textposition='outside',
                ))
                fig.update_layout(
                    title='R² at t+72h — higher is better',
                    yaxis_title='R²', xaxis_title=None,
                    height=340, margin=dict(l=10, r=10, t=60, b=10),
                    plot_bgcolor='white', showlegend=False,
                )
                fig.update_yaxes(range=[0.85, 1.0], gridcolor='#EEEEEE')
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('#### What XGBoost actually looks at')
        st.markdown('Feature importance from the trained model — the higher the bar, the more it relies on that feature.')

        fi = load_feature_importance()
        top = fi.head(20).iloc[::-1]

        # Categorise features by type
        def categorise(f):
            if f == 'sm_value': return 'Soil moisture (now)'
            if f.startswith('sm_lag'): return 'Soil moisture (past)'
            if f.startswith('sm_rolling'): return 'Rolling average'
            if f in ('hour', 'month', 'dayofyear'): return 'Time'
            if f in ('latitude','longitude','elevation_m','depth_m'): return 'Station info'
            if 't+' in f: return 'Forecast weather'
            return 'Current weather'
        top['category'] = top['feature'].apply(categorise)
        cat_colors = {
            'Soil moisture (now)':  '#1565C0',
            'Soil moisture (past)': '#42A5F5',
            'Rolling average':      '#90CAF9',
            'Time':                 '#FFB300',
            'Station info':         '#8E24AA',
            'Current weather':      '#43A047',
            'Forecast weather':     '#7CB342',
        }
        top['color'] = top['category'].map(cat_colors)

        fig = go.Figure(go.Bar(
            x=top['importance'], y=top['feature'],
            orientation='h',
            marker=dict(color=top['color']),
            customdata=top[['category']].values,
            hovertemplate='%{y}<br>Importance: %{x:.4f}<br>Category: %{customdata[0]}<extra></extra>',
        ))
        fig.update_layout(
            title='Top 20 most important features',
            xaxis_title='Importance', yaxis_title=None,
            height=550, margin=dict(l=10, r=10, t=60, b=10),
            plot_bgcolor='white',
        )
        fig.update_xaxes(gridcolor='#EEEEEE')
        st.plotly_chart(fig, use_container_width=True)

        # Category summary pie
        col_x, col_y = st.columns([1, 1])
        with col_x:
            fi['category'] = fi['feature'].apply(categorise)
            cat_summary = fi.groupby('category')['importance'].sum().reset_index().sort_values('importance', ascending=False)
            pie = go.Figure(go.Pie(
                labels=cat_summary['category'], values=cat_summary['importance'],
                marker=dict(colors=[cat_colors.get(c, '#999') for c in cat_summary['category']]),
                textinfo='label+percent', hole=0.45,
            ))
            pie.update_layout(
                title='Importance by category',
                height=400, margin=dict(l=10, r=10, t=60, b=10),
                showlegend=False,
            )
            st.plotly_chart(pie, use_container_width=True)

        with col_y:
            st.markdown('#### Reading the chart')
            st.markdown(
                '''
- **Most signal comes from the recent past.** The current reading (`sm_value`) and the
  reading 24h ago (`sm_lag_24h`) dominate. Soil moisture is a slow-moving variable —
  knowing where it *is* tells you most of what you need to know about where it\'s *going*.
- **Weather matters most at longer horizons.** At t+72h forecast weather adds a small
  but reliable lift. At t+24h, soil inertia dwarfs everything else.
- **Station identity matters a little.** Latitude/longitude help the model learn
  station-specific drying rates.
                '''
            )

    with tab3:
        st.markdown('#### A worked example: how XGBoost actually makes a prediction')

        samples = load_sample_features()
        sample = samples.sample(1, random_state=42).iloc[0]
        feature_cols = load_feature_cols()

        st.markdown(
            f'A single test point. Station **{sample["station"].split("/")[-1].replace("_", " ")}**, '
            f'date **{pd.to_datetime(sample["datetime"]).strftime("%Y-%m-%d %H:%M")}**.'
        )

        c1, c2, c3 = st.columns(3)
        with c1: stat_card('Soil moisture now', f'{sample["sm_value"]:.3f}', 'm³/m³', CFG_PALETTE['actual'])
        with c2: stat_card('24h ago', f'{sample["sm_lag_24h"]:.3f}', 'm³/m³', CFG_PALETTE['grey'])
        with c3: stat_card('120h ago (5 days)', f'{sample["sm_lag_120h"]:.3f}', 'm³/m³', CFG_PALETTE['grey'])

        st.markdown('---')

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('#### Weather context (current day)')
            wx = {
                'Temperature (°C)':     sample.get('T2M', None),
                'Max temp (°C)':        sample.get('T2M_MAX', None),
                'Humidity (%)':         sample.get('RH2M', None),
                'Precipitation (mm)':   sample.get('PRECTOTCORR', None),
                'Wind speed (m/s)':     sample.get('WS2M', None),
                'Solar (MJ/m²/day)':    sample.get('ALLSKY_SFC_SW_DWN', None),
                'ET₀ (mm/day)':         sample.get('ET0', None),
            }
            wx_df = pd.DataFrame({'Variable': list(wx.keys()), 'Value': [f'{v:.2f}' if v is not None else 'n/a' for v in wx.values()]})
            st.dataframe(wx_df, hide_index=True, use_container_width=True)

        with c2:
            st.markdown('#### The forecast')
            xgb = load_model()
            X_row = sample[feature_cols].values.reshape(1, -1).astype(float)
            pred = float(xgb.predict(X_row)[0])
            actual = float(sample['target'])
            err = pred - actual

            verdict_box(
                'green' if abs(err) < 0.02 else ('amber' if abs(err) < 0.05 else 'red'),
                f'Predicted: {pred:.3f} m³/m³',
                f'Actual was {actual:.3f} m³/m³ — error of {err:+.4f}. '
                f'Threshold {THRESHOLD}: '
                f'{"will stay above" if pred >= THRESHOLD else "predicted to drop below"}.'
            )

        st.markdown('---')
        st.markdown(
            '> **Behind the scenes:** the model received '
            f'{len(feature_cols)} features for this single prediction — soil moisture at '
            '5 past timesteps, rolling 3-day and 7-day means/stds, current weather, '
            '*forecast* weather at +24h/+48h/+72h, time of day, time of year, and '
            'static station info. It compares this row to thousands of similar rows '
            'in the training data, then averages the answers from 300 decision trees.'
        )


# ────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE DEMO
# ────────────────────────────────────────────────────────────────────────────

elif page == '🎯 Live Demo':
    st.title('Live forecast — pick a farm, pick a day')
    st.caption('Choose any sensor station as your "farm". The model gives you a 3-day forecast and an irrigation recommendation.')

    preds = load_test_predictions()
    meta = load_station_meta()
    samples = load_sample_features()
    feature_cols = load_feature_cols()
    xgb = load_model()

    # Init session state
    if 'demo_station' not in st.session_state:
        # Default to a station with interesting variation
        candidates = preds.groupby('station').agg(
            n=('y_true', 'count'),
            std=('y_true', 'std'),
        )
        candidates = candidates[candidates['n'] > 1000].sort_values('std', ascending=False)
        st.session_state.demo_station = candidates.index[0] if len(candidates) else preds['station'].iloc[0]

    col_map, col_picker = st.columns([2, 1])

    with col_picker:
        st.markdown('### 1️⃣ Pick your station')
        station_options = sorted(meta['station'].tolist())
        idx = station_options.index(st.session_state.demo_station) if st.session_state.demo_station in station_options else 0
        chosen = st.selectbox(
            'Choose station',
            options=station_options,
            index=idx,
            format_func=lambda s: s.split('/', 1)[1].replace('_', ' '),
            key='station_selectbox',
            label_visibility='collapsed',
        )
        st.session_state.demo_station = chosen

        info = meta[meta['station'] == chosen].iloc[0]
        st.markdown(
            f'<span class="pill blue">{info["country"]}</span> '
            f'<span class="pill grey">{info["network"]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**{info["short_name"]}**  \n'
            f'📍 {info["latitude"]:.3f}°, {info["longitude"]:.3f}°  \n'
            f'⛰ Elevation {info["elevation_m"]:.0f} m  \n'
            f'🪱 Sensor depth {info["depth_m"]:.2f} m'
        )

    with col_map:
        st.markdown('### 🗺️ The sensor network')
        fig = make_station_map(meta, selected_station=chosen, color_field='country')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    # === Forecast section ===
    station_preds = preds[preds['station'] == chosen].sort_values('datetime').reset_index(drop=True)
    station_samples = samples[samples['station'] == chosen].sort_values('datetime').reset_index(drop=True)

    if len(station_preds) < 10:
        st.warning('Not enough test-set data for this station to run the live demo. Pick another.')
        st.stop()

    # === Date slider ===
    st.markdown('### 2️⃣ Pick a date — the model will forecast 3 days ahead')

    min_dt = station_preds['datetime'].min().to_pydatetime()
    max_dt = station_preds['datetime'].max().to_pydatetime()

    date_choice = st.slider(
        'Today\'s date',
        min_value=min_dt, max_value=max_dt,
        value=min_dt + (max_dt - min_dt) * 0.4,
        format='YYYY-MM-DD HH:mm',
        label_visibility='collapsed',
    )

    # Snap to nearest available test datetime
    idx = (station_preds['datetime'] - date_choice).abs().idxmin()
    row = station_preds.loc[idx]

    sm_now = float(row['sm_now'])
    sm_pred = float(row['y_pred_xgb'])
    sm_actual = float(row['y_true'])
    sm_persist = float(row['y_pred_persist'])
    snap_date = pd.to_datetime(row['datetime'])
    future_date = snap_date + pd.Timedelta(hours=HORIZON_HOURS)

    # === Verdict ===
    st.markdown('### 3️⃣ Recommendation')
    level, headline, body = classify_recommendation(sm_now, sm_pred)
    verdict_box(level, headline, body)

    # === Key numbers ===
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card('Today', f'{sm_now:.3f}', f'on {snap_date.strftime("%a %d %b")}', CFG_PALETTE['actual'])
    with c2: stat_card('Predicted in 3 days', f'{sm_pred:.3f}', f'on {future_date.strftime("%a %d %b")}', CFG_PALETTE['orange'])
    with c3:
        delta = sm_pred - sm_now
        stat_card('Change forecast', f'{delta:+.3f}', 'lower = drier', CFG_PALETTE['red'] if delta < -0.02 else CFG_PALETTE['green'])
    with c4:
        margin = sm_pred - THRESHOLD
        stat_card('Margin to drought', f'{margin:+.3f}', f'threshold = {THRESHOLD}', CFG_PALETTE['red'] if margin < 0 else CFG_PALETTE['green'])

    st.markdown('---')

    # === Comparison chart ===
    st.markdown('### 4️⃣ The forecast in context')

    # Window around chosen date
    window_before = 24 * 14    # 14 days of history
    window_after  = 24 * 7     # 7 days into the future

    mask = (
        (station_preds['datetime'] >= snap_date - pd.Timedelta(hours=window_before)) &
        (station_preds['datetime'] <= snap_date + pd.Timedelta(hours=window_after))
    )
    ctx = station_preds[mask].copy()
    ctx['datetime_forecast'] = ctx['datetime'] + pd.Timedelta(hours=HORIZON_HOURS)

    fig = go.Figure()

    # Actual (historical + future where we have it)
    fig.add_trace(go.Scatter(
        x=ctx['datetime'], y=ctx['sm_now'],
        mode='lines', name='Actual (history)',
        line=dict(color=CFG_PALETTE['actual'], width=2.5),
    ))
    # Future actual (target)
    fig.add_trace(go.Scatter(
        x=ctx['datetime_forecast'], y=ctx['y_true'],
        mode='lines', name='Actual (future, ground truth)',
        line=dict(color=CFG_PALETTE['actual'], width=2.5, dash='dot'),
    ))
    # Model forecast
    fig.add_trace(go.Scatter(
        x=ctx['datetime_forecast'], y=ctx['y_pred_xgb'],
        mode='lines', name='Model forecast (XGBoost, t+72h)',
        line=dict(color=CFG_PALETTE['orange'], width=2.5),
    ))
    # Persistence forecast
    fig.add_trace(go.Scatter(
        x=ctx['datetime_forecast'], y=ctx['y_pred_persist'],
        mode='lines', name='Persistence baseline',
        line=dict(color=CFG_PALETTE['persist'], width=1.5, dash='dash'),
    ))
    # Today line — plotly's add_vline annotation positioning chokes on Timestamps,
    # so we add the line as a shape and the label as a separate annotation.
    fig.add_shape(
        type='line', xref='x', yref='paper',
        x0=snap_date, x1=snap_date, y0=0, y1=1,
        line=dict(color='black', width=2),
    )
    fig.add_annotation(
        x=snap_date, y=1.0, yref='paper',
        text='Today', showarrow=False, yshift=10,
        font=dict(color='black', size=12, family='Arial Black'),
    )
    # Forecast point
    fig.add_trace(go.Scatter(
        x=[future_date], y=[sm_pred],
        mode='markers', name='Forecast point',
        marker=dict(color=CFG_PALETTE['orange'], size=14, line=dict(color='white', width=2)),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[future_date], y=[sm_actual],
        mode='markers', name='Actual point',
        marker=dict(color=CFG_PALETTE['actual'], size=14, line=dict(color='white', width=2)),
        showlegend=False,
    ))
    # Threshold
    fig.add_hline(
        y=THRESHOLD, line_dash='dash', line_color=CFG_PALETTE['red'],
        annotation_text=f'Drought ({THRESHOLD})',
        annotation_position='top right',
    )
    fig.add_hrect(y0=0, y1=THRESHOLD, fillcolor='red', opacity=0.05, line_width=0)

    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
                    bgcolor='rgba(255,255,255,0.85)'),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis_title=None, yaxis_title='Soil moisture (m³/m³)',
    )
    fig.update_xaxes(gridcolor='#EEEEEE')
    fig.update_yaxes(gridcolor='#EEEEEE')
    st.plotly_chart(fig, use_container_width=True)

    # === Compare model vs persistence ===
    st.markdown('### 5️⃣ Why not just guess "tomorrow looks like today"?')
    error_xgb = abs(sm_pred - sm_actual)
    error_persist = abs(sm_persist - sm_actual)

    cA, cB, cC = st.columns(3)
    with cA: stat_card('Model error', f'{error_xgb:.4f}', f'predicted {sm_pred:.3f} vs actual {sm_actual:.3f}', CFG_PALETTE['orange'])
    with cB: stat_card('Persistence error', f'{error_persist:.4f}', f'guessed {sm_persist:.3f}', CFG_PALETTE['grey'])
    with cC:
        if error_xgb < error_persist:
            stat_card('Model wins by', f'{(error_persist - error_xgb):.4f}', 'fewer mistakes', CFG_PALETTE['green'])
        elif error_xgb > error_persist:
            stat_card('Persistence wins by', f'{(error_xgb - error_persist):.4f}', 'fewer mistakes', CFG_PALETTE['red'])
        else:
            stat_card('Tie', '0.0000', 'equal error', CFG_PALETTE['grey'])


# ────────────────────────────────────────────────────────────────────────────
# PAGE: RESULTS
# ────────────────────────────────────────────────────────────────────────────

elif page == '🏆 Results':
    st.title('Results')
    st.caption('Before vs after, how far ahead can we forecast, and what really matters: catching upcoming drought events.')

    tab1, tab2, tab3 = st.tabs(['🎯 Drought event detection', '⏳ Horizon trade-off', '🌤️ Does weather help?'])

    # --- Tab 1: drought event detection ---
    with tab1:
        st.markdown('#### The number that matters most: catching upcoming drought events')
        st.markdown(
            '**Drought onset event** = soil is healthy right now, but will fall below the '
            f'{THRESHOLD} threshold within 3 days. Missing one means crops go thirsty. '
            'Catching one means the farmer can irrigate in time.'
        )

        onset = load_onset_analysis()
        if not onset.empty:
            # Filter to interesting models
            show_onset = onset[onset['Model'].isin(['Persistence', 'Random Forest', 'XGBoost'])].copy()
            show_onset = show_onset.set_index('Model')
            show_onset = show_onset.reindex(['Persistence', 'Random Forest', 'XGBoost'])

            c1, c2, c3 = st.columns(3)
            for col, model in zip([c1, c2, c3], ['Persistence', 'Random Forest', 'XGBoost']):
                if model not in show_onset.index:
                    continue
                row = show_onset.loc[model]
                catch_pct = row['Catch rate'] * 100
                if model == 'Persistence':
                    color = CFG_PALETTE['grey']
                elif model == 'XGBoost':
                    color = CFG_PALETTE['orange']
                else:
                    color = CFG_PALETTE['blue']
                with col:
                    stat_card(
                        model,
                        f'{catch_pct:.1f}%',
                        f'{int(row["Caught"]):,} / {int(row["Onsets in test"]):,} caught · {int(row["False alarms"]):,} false alarms',
                        color,
                    )

            # Bar chart of catch rate vs false alarm rate
            fig = make_subplots(rows=1, cols=2, subplot_titles=['Drought onsets caught', 'False alarms'])
            fig.add_trace(go.Bar(
                x=show_onset.index, y=show_onset['Catch rate'] * 100,
                marker_color=[CFG_PALETTE['grey'], CFG_PALETTE['blue'], CFG_PALETTE['orange']],
                text=[f'{v*100:.1f}%' for v in show_onset['Catch rate']],
                textposition='outside', name='Catch rate',
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=show_onset.index, y=show_onset['False alarm rate'] * 100,
                marker_color=[CFG_PALETTE['grey'], CFG_PALETTE['blue'], CFG_PALETTE['orange']],
                text=[f'{v*100:.1f}%' for v in show_onset['False alarm rate']],
                textposition='outside', name='False alarm rate',
            ), row=1, col=2)
            fig.update_layout(
                showlegend=False, height=400,
                margin=dict(l=10, r=10, t=60, b=10),
                plot_bgcolor='white',
            )
            fig.update_yaxes(title_text='% of upcoming droughts', row=1, col=1, range=[0, 80], gridcolor='#EEEEEE')
            fig.update_yaxes(title_text='% of dry-stays-dry alarms', row=1, col=2, range=[0, 12], gridcolor='#EEEEEE')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('---')
        st.markdown('#### What this means in £')
        col_a, col_b = st.columns(2)
        with col_a:
            est_events_per_season = st.slider('Drought onset events per growing season', 1, 6, 3)
            farm_ha = st.slider('Farm size (hectares)', 0.5, 5.0, 1.0, 0.5)
        with col_b:
            avoided_loss_per_event_per_ha = 28.80  # $ — from PROJECT_ANALYSIS.md
            extra_catches_per_event = (0.575 - 0.455)   # XGB - persistence catch rate
            value_per_season = est_events_per_season * extra_catches_per_event * avoided_loss_per_event_per_ha * farm_ha

            stat_card(
                'Extra yield protected vs no-model baseline',
                f'${value_per_season:.0f}',
                f'per season on a {farm_ha:.1f} ha farm, '
                f'{est_events_per_season} onset events, '
                f'based on FAO maize yield-loss estimates',
                CFG_PALETTE['green'],
            )

    # --- Tab 2: horizon ---
    with tab2:
        st.markdown('#### How far ahead can we forecast?')
        st.markdown(
            'Every horizon is harder than the one before. Below: model accuracy as a function '
            'of how many hours into the future we ask it to predict.'
        )

        hz = load_horizon_metrics()
        if not hz.empty:
            metric_choice = st.radio(
                'Metric',
                ['R² (higher = better)', 'MAE (lower = better)', 'Recall (catching drought)'],
                horizontal=True,
            )

            fig = go.Figure()
            if metric_choice.startswith('R²'):
                fig.add_trace(go.Scatter(x=hz['horizon'], y=hz['persist_r2'], mode='lines+markers',
                                         name='Persistence baseline',
                                         line=dict(color=CFG_PALETTE['grey'], width=2.5, dash='dash')))
                fig.add_trace(go.Scatter(x=hz['horizon'], y=hz['xgb_nw_r2'], mode='lines+markers',
                                         name='XGBoost (no weather)',
                                         line=dict(color=CFG_PALETTE['blue'], width=2.5)))
                fig.add_trace(go.Scatter(x=hz['horizon'], y=hz['xgb_w_r2'], mode='lines+markers',
                                         name='XGBoost (with weather)',
                                         line=dict(color=CFG_PALETTE['orange'], width=3)))
                fig.update_yaxes(title_text='R²', gridcolor='#EEEEEE')
            elif metric_choice.startswith('MAE'):
                fig.add_trace(go.Scatter(x=hz['horizon'], y=hz['persist_mae'], mode='lines+markers',
                                         name='Persistence baseline',
                                         line=dict(color=CFG_PALETTE['grey'], width=2.5, dash='dash')))
                fig.add_trace(go.Scatter(x=hz['horizon'], y=hz['xgb_nw_mae'], mode='lines+markers',
                                         name='XGBoost (no weather)',
                                         line=dict(color=CFG_PALETTE['blue'], width=2.5)))
                fig.add_trace(go.Scatter(x=hz['horizon'], y=hz['xgb_w_mae'], mode='lines+markers',
                                         name='XGBoost (with weather)',
                                         line=dict(color=CFG_PALETTE['orange'], width=3)))
                fig.update_yaxes(title_text='MAE (m³/m³)', gridcolor='#EEEEEE')
            else:
                fig.add_trace(go.Scatter(x=hz['horizon'], y=hz['xgb_nw_recall'], mode='lines+markers',
                                         name='XGBoost (no weather)',
                                         line=dict(color=CFG_PALETTE['blue'], width=2.5)))
                fig.add_trace(go.Scatter(x=hz['horizon'], y=hz['xgb_w_recall'], mode='lines+markers',
                                         name='XGBoost (with weather)',
                                         line=dict(color=CFG_PALETTE['orange'], width=3)))
                fig.update_yaxes(title_text='Recall (drought events caught)', gridcolor='#EEEEEE')

            fig.update_layout(
                height=460, margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title='Forecast horizon (hours ahead)',
                plot_bgcolor='white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            )
            fig.update_xaxes(gridcolor='#EEEEEE', tickvals=[24, 48, 72, 96, 120, 168])
            st.plotly_chart(fig, use_container_width=True)

            # Performance gap table
            st.markdown('#### Performance gap over the naive baseline')
            gap = hz[['horizon', 'persist_r2', 'xgb_w_r2', 'persist_mae', 'xgb_w_mae']].copy()
            gap['R² gain'] = gap['xgb_w_r2'] - gap['persist_r2']
            gap['MAE reduction'] = gap['persist_mae'] - gap['xgb_w_mae']
            gap['MAE reduction %'] = gap['MAE reduction'] / gap['persist_mae'] * 100
            display_gap = gap[['horizon', 'persist_r2', 'xgb_w_r2', 'R² gain', 'persist_mae', 'xgb_w_mae', 'MAE reduction %']].copy()
            display_gap.columns = ['Horizon (h)', 'Persistence R²', 'Model R²', 'R² gain', 'Persistence MAE', 'Model MAE', 'MAE reduction %']
            st.dataframe(
                display_gap.style.format({
                    'Persistence R²': '{:.4f}', 'Model R²': '{:.4f}', 'R² gain': '{:+.4f}',
                    'Persistence MAE': '{:.4f}', 'Model MAE': '{:.4f}', 'MAE reduction %': '{:.1f}%',
                }).background_gradient(subset=['R² gain', 'MAE reduction %'], cmap='Greens'),
                use_container_width=True, hide_index=True,
            )

    # --- Tab 3: weather impact ---
    with tab3:
        st.markdown('#### Does adding weather features help?')

        wabl = load_weather_ablation()
        hz = load_horizon_metrics()

        if not wabl.empty:
            st.markdown('At t+120h (5 days), here\'s how each model performs **with and without weather features**:')

            fig = make_subplots(rows=1, cols=2, subplot_titles=['R² (higher = better)', 'MAE (lower = better)'])

            wabl_w  = wabl[wabl['Weather'] == 'yes' ].set_index('Model')
            wabl_nw = wabl[wabl['Weather'] == 'no'  ].set_index('Model')

            for i, model in enumerate(['Random Forest', 'XGBoost', 'LSTM', 'TFT']):
                if model not in wabl_w.index: continue
                colors = ['#90A4AE', CFG_PALETTE['orange']]
                fig.add_trace(go.Bar(
                    x=[model], y=[wabl_nw.loc[model, 'R2']],
                    marker_color=colors[0], name='No weather', showlegend=(i == 0),
                    legendgroup='nw', text=f'{wabl_nw.loc[model, "R2"]:.3f}', textposition='outside',
                ), row=1, col=1)
                fig.add_trace(go.Bar(
                    x=[model], y=[wabl_w.loc[model, 'R2']],
                    marker_color=colors[1], name='With weather', showlegend=(i == 0),
                    legendgroup='w', text=f'{wabl_w.loc[model, "R2"]:.3f}', textposition='outside',
                ), row=1, col=1)
                fig.add_trace(go.Bar(
                    x=[model], y=[wabl_nw.loc[model, 'MAE']],
                    marker_color=colors[0], showlegend=False, legendgroup='nw',
                    text=f'{wabl_nw.loc[model, "MAE"]:.4f}', textposition='outside',
                ), row=1, col=2)
                fig.add_trace(go.Bar(
                    x=[model], y=[wabl_w.loc[model, 'MAE']],
                    marker_color=colors[1], showlegend=False, legendgroup='w',
                    text=f'{wabl_w.loc[model, "MAE"]:.4f}', textposition='outside',
                ), row=1, col=2)

            fig.update_layout(
                barmode='group', height=460,
                margin=dict(l=10, r=10, t=80, b=10),
                plot_bgcolor='white',
                legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5),
            )
            fig.update_yaxes(gridcolor='#EEEEEE')
            fig.update_yaxes(range=[0.83, 0.93], row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)

            st.success(
                '**Verdict:** All four models gain from weather, but the benefit is *small at short horizons* '
                'and grows as we forecast further out. At t+120h, weather features add ~3 R² points across '
                'the board. At t+24h (1 day) the effect is under 0.5 R² points — soil inertia dominates.'
            )

        if not hz.empty:
            st.markdown('#### % improvement from weather features, by horizon')

            hz2 = hz.copy()
            hz2['r2_pct_gain'] = (hz2['xgb_w_r2'] - hz2['xgb_nw_r2']) / hz2['xgb_nw_r2'] * 100
            hz2['mae_pct_gain'] = (hz2['xgb_nw_mae'] - hz2['xgb_w_mae']) / hz2['xgb_nw_mae'] * 100

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f't+{h}h' for h in hz2['horizon']], y=hz2['r2_pct_gain'],
                name='R² gain (%)', marker_color=CFG_PALETTE['orange'],
                text=[f'+{v:.2f}%' for v in hz2['r2_pct_gain']], textposition='outside',
            ))
            fig.add_trace(go.Bar(
                x=[f't+{h}h' for h in hz2['horizon']], y=hz2['mae_pct_gain'],
                name='MAE reduction (%)', marker_color=CFG_PALETTE['green'],
                text=[f'-{v:.2f}%' for v in hz2['mae_pct_gain']], textposition='outside',
            ))
            fig.update_layout(
                barmode='group', height=380,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title='Forecast horizon',
                yaxis_title='% improvement when adding weather features',
                plot_bgcolor='white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            )
            fig.update_yaxes(gridcolor='#EEEEEE')
            st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# PAGE: CONCLUSIONS
# ────────────────────────────────────────────────────────────────────────────

elif page == '💡 Conclusions':
    st.title('Conclusions')
    st.caption('What we built, what we found, and what comes next.')

    st.markdown('### Headline findings')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '#### 🎯 The model works\n\n'
            '**R² = 0.93** at t+72h. The model predicts soil moisture 3 days in advance with average error of '
            '**0.015 m³/m³** — equivalent to ±5% of the drought threshold.'
        )
    with col2:
        st.markdown(
            '#### ⚠️ It catches what matters\n\n'
            'When the soil will fall *from healthy to dry* in the next 3 days, the model catches **57% of those events**. '
            'The naive baseline catches **27%**. The model halves the miss rate.'
        )
    with col3:
        st.markdown(
            '#### 🤖 XGBoost beat the neural networks\n\n'
            'Tree-based gradient boosting outperformed both vanilla LSTM and TFT across most experiments. '
            'Hand-crafted lag features are still hard to beat for slow-moving physical variables.'
        )

    st.markdown('---')
    st.markdown('### The journey — 10 iterations')

    journey = pd.DataFrame([
        {'Iteration': 'Test 1', 'Change': 'Baseline: no weather, t+24h',                'XGB R²': 0.97},
        {'Iteration': 'Test 2', 'Change': 'Weather features added, t+72h',              'XGB R²': 0.90},
        {'Iteration': 'Test 3', 'Change': 'Extended lags to 5 days',                    'XGB R²': 0.898},
        {'Iteration': 'Test 4', 'Change': 'TFT model introduced',                       'XGB R²': 0.898},
        {'Iteration': 'Test 5', 'Change': 'Weather added to LSTM/TFT sequences',        'XGB R²': 0.924},
        {'Iteration': 'Test 6', 'Change': 'Rolling-window features added',              'XGB R²': 0.923},
        {'Iteration': 'Test 7', 'Change': 'ET₀ + tuned XGB + LSTM dropout + norm fix',  'XGB R²': 0.9305},
        {'Iteration': 'Test 8', 'Change': 'Oracle future weather for LSTM/TFT',         'XGB R²': 0.9302},
        {'Iteration': 'Test 9', 'Change': 'TFT fix + threshold optimisation',           'XGB R²': 0.9302},
        {'Iteration': 'Test 10', 'Change': 'Final unified run',                          'XGB R²': 0.9302},
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=journey['Iteration'], y=journey['XGB R²'],
        mode='lines+markers+text',
        text=[f'{v:.3f}' for v in journey['XGB R²']], textposition='top center',
        line=dict(color=CFG_PALETTE['orange'], width=3),
        marker=dict(size=14, color=CFG_PALETTE['orange'], line=dict(color='white', width=2)),
        hovertemplate='<b>%{x}</b><br>R²: %{y:.4f}<br>%{customdata}<extra></extra>',
        customdata=journey['Change'],
    ))
    fig.update_layout(
        title='XGBoost R² across 10 iterations',
        height=420, margin=dict(l=10, r=10, t=60, b=80),
        plot_bgcolor='white', yaxis_title='R² (t+72h with weather)',
    )
    fig.update_yaxes(range=[0.88, 0.96], gridcolor='#EEEEEE')
    fig.update_xaxes(tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    # ── What's next ──
    st.markdown('### What we\'d do next')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '''
**🌍 Couple with real weather forecasts**
Right now the model uses *observed* future weather as a stand-in for a perfect forecast.
The next version would use real numerical weather predictions (NWP), with their own errors.

**🌐 Cross-station generalisation**
Train on 40 stations, evaluate on 8 *unseen* stations. Quantifies how well the model
deploys to a new farm.

**🏔 Soil texture features**
Clay vs sand vs loam fundamentally changes how soil holds water. SoilGrids provides this
globally — currently missing.
            '''
        )
    with c2:
        st.markdown(
            '''
**🌿 Vegetation index (NDVI)**
Plant transpiration is the main driver of moisture loss between rains.
Monthly MODIS/Sentinel-2 NDVI would add this signal.

**📊 Uncertainty quantification**
Quantile XGBoost can produce prediction ranges ("between 0.24 and 0.28") instead of
a single number. Farmers need ranges, not point estimates.

**🌧 Seasonal performance breakdown**
Disaggregate metrics by wet vs dry season — performance likely varies by both.
            '''
        )

    st.markdown('---')

    # ── Try a fresh scenario ──
    st.markdown('### One last scenario — pick a station and see one final summary')
    meta = load_station_meta()
    preds = load_test_predictions()

    final_station = st.selectbox(
        'Pick any station for a final summary',
        options=sorted(meta['station'].tolist()),
        format_func=lambda s: s.split('/', 1)[1].replace('_', ' '),
        key='final_summary_station',
    )

    station_preds = preds[preds['station'] == final_station]
    if len(station_preds) > 0:
        below = (station_preds['y_true'] < THRESHOLD).sum()
        total = len(station_preds)
        below_pct = below / total * 100

        below_pred_xgb = (station_preds['y_pred_xgb'] < THRESHOLD).sum()
        below_pred_pers = (station_preds['y_pred_persist'] < THRESHOLD).sum()

        from sklearn.metrics import mean_absolute_error
        mae_xgb = mean_absolute_error(station_preds['y_true'], station_preds['y_pred_xgb'])
        mae_persist = mean_absolute_error(station_preds['y_true'], station_preds['y_pred_persist'])

        c1, c2, c3, c4 = st.columns(4)
        with c1: stat_card('Test forecasts made', f'{total:,}', 'on this station', CFG_PALETTE['blue'])
        with c2: stat_card('Below threshold (truth)', f'{below_pct:.1f}%', f'{below:,} dry forecasts', CFG_PALETTE['red'])
        with c3: stat_card('Model error (MAE)', f'{mae_xgb:.4f}', 'mean abs deviation', CFG_PALETTE['orange'])
        with c4:
            ratio = (mae_persist / mae_xgb) if mae_xgb > 0 else 1
            stat_card('Baseline error', f'{mae_persist:.4f}', f'{ratio:.2f}× worse than model', CFG_PALETTE['grey'])

    st.markdown('---')
    st.markdown(
        '<div style="text-align:center;padding:1.5rem;background:#F1F8E9;border-radius:12px;">'
        '<h3 style="margin-top:0;">Built by the ENGG2112 Data Farmers</h3>'
        '<p style="margin-bottom:0;font-size:1.05rem;color:#555;">'
        '<strong>Angus</strong> — Domain Researcher · '
        '<strong>James</strong> — Project Lead · '
        '<strong>Oscar</strong> — ML Engineer · '
        '<strong>Byron</strong> — Data Engineer'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )
