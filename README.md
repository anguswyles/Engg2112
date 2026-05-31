# ENGG2112 — Data Farmers

<p align="center">
  <img src="images/datafarmers.png" alt="Data Farmers team photo" width="680"/>
</p>

<p align="center">
  <strong>Machine learning for soil moisture forecasting and irrigation decision support</strong><br/>
  ENGG2112 · University of Queensland · 2025
</p>

<p align="center">
  <a href="#https://engg2112.onrender.com/">🌐 Project Website</a> &nbsp;·&nbsp;
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#running-an-experiment">Quick Start</a> &nbsp;·&nbsp;
  <a href="#team">Team</a>
</p>

---

## Overview

This project builds a multi-model forecasting system that predicts soil moisture up to **7 days ahead** at sensor stations across East Africa, and flags when moisture is likely to fall below the critical irrigation threshold (0.30 m³/m³). The goal is to give smallholder farmers actionable advance warning before drought stress sets in — moving from reactive to proactive irrigation.

Four model architectures are trained on real-world sensor data from **48 ISMN stations** (TAHMO and COSMOS networks) across Kenya, Uganda, and Rwanda. Weather features from the **NASA POWER API** are merged per station and evaluated for their marginal impact at each forecast horizon.

---

## Project Structure

```
data/
  soil_data/
    ismn_data/          # ISMN station CSVs (COSMOS, TAHMO networks)
    clean_ismn.py       # Processes raw ISMN zip files into per-station CSVs
  weather_data/
    stations/           # NASA POWER weather CSVs per station
    fetch_weather.py    # Fetches weather data from the NASA POWER API

models/
  config.py             # Shared constants and hyperparameter grids
  prepare.py            # Data loading and feature engineering pipeline
  train_models.py       # RF, XGBoost, LSTM, and TFT training functions
  diagnostics.py        # Leakage checks, feature ablation, horizon sweeps

scripts/
  run_experiment.py               # Full experiment runner — all models, all plots
  run_station_holdout.py          # Cross-station generalisation evaluation
  run_drought_onset_classification.py
  weather_ablation_four_models.py

app.py                  # Interactive Plotly Dash demo app
images/                 # Saved outputs from each numbered test run
```

---

## Models

| Model | Approach | Cross-validation |
|---|---|---|
| **Random Forest** | Ensemble of decision trees with bagging | GroupKFold (by station) |
| **XGBoost** | Gradient boosted trees with regularisation | GroupKFold (by station) |
| **LSTM** | Recurrent network on 120-hour moisture windows | Temporal 80/20 split |
| **TFT** | Temporal Fusion Transformer with static station encoding and multi-head attention | Temporal 80/20 split |

Features include soil moisture lags (24–120 h), time-of-day and calendar variables, station metadata (latitude, longitude, elevation, depth), and 7 NASA POWER weather variables including derived ET₀ (reference evapotranspiration, Hargreaves-Samani method).

---

## Results

Results below are from the station-holdout evaluation — models trained on 40 stations and tested on 8 **unseen** stations — at the t+24 h horizon. This measures cross-station generalisation rather than in-distribution fit.

| Model | Weather | MAE | R² | Recall | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | No | 0.0158 | 0.938 | 0.974 | 0.984 |
| XGBoost | No | 0.0165 | 0.932 | 0.970 | 0.985 |
| LSTM | No | 0.0139 | 0.942 | 0.988 | 0.987 |
| Random Forest | Weather | 0.0154 | 0.943 | 0.963 | 0.987 |
| **XGBoost** | **Weather** | **0.0145** | **0.946** | **0.964** | **0.987** |
| LSTM | Weather | 0.0215 | 0.912 | 0.948 | 0.982 |
| TFT | No | 0.0545 | 0.467 | 0.839 | 0.973 |
| TFT | Weather | 0.0264 | 0.872 | 0.915 | 0.968 |

Weather features improve XGBoost and RF most noticeably at horizons beyond 48 h. The benefit is largest at t+72–120 h where single-day persistence is no longer a reliable baseline.

---

## Running an Experiment

```bash
# Train all 4 models and save outputs to a new numbered test folder
python scripts/run_experiment.py "test N - description"

# Auto-names the next test number if no argument is given
python scripts/run_experiment.py

# Train individual models without generating plots
python models/train_models.py

# Run diagnostics (leakage check, feature ablation, horizon sweep)
python models/diagnostics.py

# Fetch weather data from NASA POWER API
python data/weather_data/fetch_weather.py

# Process raw ISMN zip files into per-station CSVs
python data/soil_data/clean_ismn.py
```

Each run saves to `images/test N - description/` and produces ROC and precision-recall curves, confusion matrices, predicted vs actual scatter, residual distributions, feature importance, horizon comparison across t+24–168 h, and a weather impact analysis. A progression table comparing all previous runs is printed at the end.

---

## Data Sources

| Source | Description |
|---|---|
| [ISMN](https://ismn.earth) | In-situ hourly soil moisture — TAHMO and COSMOS networks, East Africa |
| [NASA POWER](https://power.larc.nasa.gov) | Satellite-derived daily weather reanalysis (temperature, humidity, precipitation, solar radiation, wind speed) |

---

## Demo App

An interactive Plotly Dash app (`app.py`) visualises model predictions, drought onset warnings, and station-level metrics.

```bash
python app.py
```

---

## Team

| Name | Role |
|---|---|
| Angus Wyles | Domain Researcher |
| James | Project Lead |
| Oscar Everett | ML Engineer |
| Byron | Data Engineer |
