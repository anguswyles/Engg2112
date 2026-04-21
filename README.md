# ENGG2112 — Data Farmers

<p align="center">
  <img src="images/datafarmers.png" alt="Team photo" width="700"/>
</p>

A machine learning system for soil moisture forecasting to support smarter irrigation decisions. Given historical soil moisture readings and weather data, the system predicts future soil moisture levels up to 7 days ahead and flags when moisture is likely to fall below critical irrigation thresholds.

---

## Overview

The system trains three models — XGBoost, Random Forest, and LSTM — on real-world sensor data from the ISMN (International Soil Moisture Network) spanning 48 stations across East Africa. Weather features (temperature, humidity, precipitation, solar radiation, wind speed) are sourced from the NASA POWER API and merged per station.

Models are evaluated on regression metrics (MAE, R²) and classification metrics (Recall, ROC AUC) for detecting below-threshold drought stress events. Performance is tracked across prediction horizons from t+24h to t+168h, with and without weather features, to quantify the added value of weather data at each horizon.

---

## Project Structure

```
data/
  soil_data/
    ismn_data/          # ISMN station CSVs (COSMOS, TAHMO networks)
    clean_ismn.py       # Processes raw ISMN zip files into station CSVs
  weather_data/
    stations/           # NASA POWER weather CSVs per station (COSMOS, TAHMO)
    fetch_weather.py    # Fetches weather data from NASA POWER API

models/
  config.py             # All shared constants and hyperparameter grids
  prepare.py            # Data loading and feature engineering
  train_models.py       # XGBoost, Random Forest, LSTM training functions
  diagnostics.py        # Feature ablation, leakage checks, horizon comparison

scripts/
  run_experiment.py     # Full experiment runner — trains all models and saves plots

images/
  test 1 - baseline no weather t+24h/
  test 2 - weather features t+72h/
  test 3 - extended lags 5 day/
```

---

## Running an Experiment

```bash
python scripts/run_experiment.py "test 4 - description of change"
```

Each run saves to a new folder under `images/` containing ROC curves, precision-recall curves, confusion matrices, predicted vs actual plots, residual distributions, feature importance, horizon comparison, and weather impact plots. A progression table comparing all previous runs is printed at the end.

---

## Models

| Model | Approach |
|---|---|
| XGBoost | Gradient boosted trees with GroupKFold cross-validation, grid search tuning |
| Random Forest | Ensemble with GroupKFold cross-validation, grid search tuning |
| LSTM | Sequence model on raw hourly soil moisture windows |

Features include soil moisture lags (24h–120h), time of day, month, day of year, station latitude/longitude/elevation/depth, and 7 NASA POWER weather variables.

---

## Key Results

Weather features consistently improve performance, with the largest benefit at t+72h–t+120h horizons. XGBoost achieves R²=0.96 at t+24h, degrading to R²=0.79 at t+168h. The persistence baseline (predicting no change from 24h ago) is competitive at very long horizons, confirming the fundamental difficulty of multi-day forecasting.

---

## Team

| Name | Role |
|---|---|
| Angus | Domain Researcher |
| James | Project Lead |
| Oscar | ML Engineer |
| Byron | Data Engineer |
