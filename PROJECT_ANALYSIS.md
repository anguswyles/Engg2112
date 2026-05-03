# Soil Moisture Forecasting — Full Project Analysis
### ENGG2112 — Data Farmers

---

## 1. Project Overview

This project builds a machine learning system to forecast soil moisture up to 7 days ahead at 48 sensor stations across East Africa (Kenya, Uganda, Rwanda). The primary application is **smart irrigation scheduling for smallholder farmers** — predicting when soil moisture will drop below a critical threshold so farmers can irrigate in advance rather than reactively.

The system frames the problem as two overlapping tasks:
1. **Regression**: predict the exact soil moisture value (m³/m³) at a future time horizon
2. **Classification**: predict whether soil moisture will fall below 0.30 m³/m³ (the drought stress threshold for most crops), treated as a binary drought detection problem

Four model types were built and compared: **Random Forest**, **XGBoost**, **LSTM**, and a **Temporal Fusion Transformer (TFT)**. Results are benchmarked against a **persistence baseline** (naive assumption that tomorrow's soil moisture equals today's).

---

## 2. Dataset

### 2.1 Soil Moisture Data — ISMN

**Source**: International Soil Moisture Network (ISMN) — the global reference database for in-situ soil moisture observations.

**Networks used**:
- **TAHMO** (Trans-African Hydro-Meteorological Observatory): 46 stations
- **COSMOS** (COsmic-ray Soil Moisture Observing System): 2 stations (KLEE, Mpala North)

**Geographic coverage**: East Africa — predominantly Kenya (≈30 stations), Uganda (≈10 stations), Rwanda (≈8 stations). Stations span a wide range of environments: coastal lowlands (Likoni, Base Titanium Dam), highland forest (Kitabi College, IPRC Musanze), savanna (Maasai Mara stations), and urban/peri-urban (Kampala KCCA school cluster).

**Total stations**: 48  
**Total quality-flagged good readings**: 1,103,461  
**Temporal coverage**: 2016–2026 (varies by station)  
**Measurement frequency**: hourly  
**Sensor depths**: varies by station (mostly 0.2m)  
**Units**: volumetric water content (m³/m³)

**Raw data format**: ISMN `.stm` files — plain text, one header line followed by timestamped records:
```
yyyy/mm/dd HH:MM  value  ismn_flag  provider_flag
```

### 2.2 Data Cleaning Pipeline (`clean_ismn.py`)

The raw ISMN data was processed through a bespoke cleaning script:

1. **Quality flag filtering**: Only readings flagged `G` (Good) were retained. Readings flagged `U` (Unknown) or `C`/`D` codes (instrument drift, out-of-range, calibration issues) were discarded entirely.
2. **Sentinel value removal**: Values of `-9999` and `-999` (ISMN no-data codes) were detected and converted to `NaN` rather than passed to models.
3. **Malformed row handling**: Rows with unexpected column counts were flagged and excluded.
4. **Duplicate timestamp removal**: Where a station had duplicate timestamps, the first occurrence was kept.
5. **Hourly resampling**: After filtering, data was resampled to a uniform 1-hour grid using linear time interpolation, capped at a maximum gap of 6 hours. Gaps longer than 6 hours remained as `NaN` and were dropped at dataset build time.

### 2.3 Weather Data — NASA POWER

**Source**: NASA POWER (Prediction Of Worldwide Energy Resources) API — a satellite-derived reanalysis product providing complete global daily coverage with no missing values.

**Fetching method**: A bespoke script (`fetch_weather.py`) queried the NASA POWER REST API once per station. Station coordinates were read directly from the ISMN CSV files (each row contains latitude and longitude of the sensor). No spatial matching or nearest-neighbour lookup was needed — coordinates from the ISMN data were passed directly to the API, which returns data for that exact geographic point.

**Variables fetched**:
| Parameter | Description | Units |
|---|---|---|
| T2M | Mean air temperature at 2m | °C |
| T2M_MAX | Daily maximum temperature | °C |
| T2M_MIN | Daily minimum temperature | °C |
| RH2M | Relative humidity at 2m | % |
| PRECTOTCORR | Corrected precipitation | mm/day |
| WS2M | Wind speed at 2m | m/s |
| ALLSKY_SFC_SW_DWN | All-sky surface shortwave solar radiation | MJ/m²/day |

**Derived variable — ET0 (Reference Evapotranspiration)**: Computed from the above using the Hargreaves-Samani equation, which is an FAO-recognised method requiring only temperature and solar radiation:

```
ET0 = 0.0023 × (Ra / 0.408) × (T2M + 17.8) × √(T2M_MAX − T2M_MIN)
```

where Ra is solar radiation (ALLSKY_SFC_SW_DWN) converted from MJ/m²/day to equivalent mm/day by dividing by 0.408. ET0 directly quantifies how much water the atmosphere is drawing from the soil surface — the primary mechanism of soil moisture loss between rain events.

**Weather data quality**: NASA POWER is a reanalysis product (model-based, not raw sensor), so it has no missing values or quality flag issues. Post-fetch coverage checks verified that weather date ranges covered each station's soil moisture period.

**Temporal resolution mismatch**: Weather data is daily; soil moisture is hourly. When merging, daily weather values were broadcast across all 24 hours of each day (left-merge on date, same value for each hour of that day). This is a known approximation — discussed in limitations.

---

## 3. Feature Engineering

### 3.1 Tabular Features (for XGBoost and Random Forest)

Each row represents one timestep at one station. Features:

**Soil moisture history (6 features)**:
- `sm_value`: current soil moisture reading (m³/m³)
- `sm_lag_24h`, `sm_lag_48h`, `sm_lag_72h`, `sm_lag_96h`, `sm_lag_120h`: past readings at 1, 2, 3, 4, and 5 days ago

**Rolling statistics (4 features)** — added in Test 6:
- `sm_rolling_mean_72h`: 3-day rolling mean (shifted by 1h to prevent lookahead)
- `sm_rolling_std_72h`: 3-day rolling standard deviation
- `sm_rolling_mean_168h`: 7-day rolling mean
- `sm_rolling_std_168h`: 7-day rolling standard deviation

Rolling features encode trend (is soil moisture rising or falling?) and volatility (are conditions stable or changing rapidly?), analogous to moving averages in financial time series.

**Temporal features (3 features)**:
- `hour`, `month`, `dayofyear`: capture diurnal and seasonal cycles

**Station metadata (4 features)**:
- `latitude`, `longitude`, `elevation_m`, `depth_m`: capture geographic and physical context

**Current weather (8 features, when `with_weather=True`)**:
- The 7 NASA POWER variables + ET0

**Forward weather windows** — a key design choice: to simulate having a weather forecast, the model also receives the actual observed weather at each 24-hour step from t+24h up to the prediction horizon. For t+72h this adds 3 × 8 = 24 extra weather features (weather at t+24h, t+48h, t+72h). This is explicitly acknowledged as an upper bound — real deployment would use a numerical weather prediction forecast, which introduces additional error.

**Total features at t+72h**: 49

### 3.2 Sequence Features (for LSTM and TFT)

A sliding window of the 120 most recent hourly readings is passed as a sequence. In Test 7 (with weather), this is shape `(120, 9)` — 120 timesteps, 9 channels (sm_value + 7 weather variables + ET0). In earlier tests without weather channels, shape was `(120, 1)`.

**Normalisation**: Each channel normalised independently using mean and standard deviation computed **on the training split only** — a correction made in Test 7 after identifying test-set statistics were leaking into normalisation in earlier runs.

---

## 4. Model Architecture

### 4.1 Random Forest
- 200 trees, max_depth=10, max_features='sqrt', random_state=0
- Trained on tabular feature matrix
- No explicit hyperparameter tuning in experiment runs (fixed parameters)

### 4.2 XGBoost
- Gradient boosted trees with histogram-based split finding
- Parameters (as of Test 7): `n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`
- Note: earlier tests (2–6) used default `max_depth=3, n_estimators=100` — these were hardcoded defaults from `config.py` and were not the result of grid search. The `train_models.py` script contains a `GridSearchCV` routine but its results were never written back to the config used by experiments. This was identified and fixed in Test 7.

### 4.3 LSTM
- Architecture: `LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)`
- Input: normalised (120, n_features) sequence
- Dropout added in Test 7 to address lack of regularisation in earlier tests
- Trained with Adam optimiser, MAE loss, batch size 512, up to 20 epochs with early stopping (patience=3, restore best weights)

**Why LSTM underperforms XGBoost**: LSTM processes a raw time series and must implicitly learn the relationships between lag values. XGBoost is explicitly given those lag values as tabular features, along with weather and station metadata. The lag feature engineering essentially does the temporal work for XGBoost, making LSTM's recurrent advantage redundant. A fair comparison would use LSTM with identical feature access.

### 4.4 TFT (Temporal Fusion Transformer — inspired)
- Custom Keras implementation (not pytorch-forecasting)
- Architecture: Static feature encoder (Dense(16, elu) → Dense(16)) → broadcast to sequence → Concatenate with sequence input → LSTM(64, return_sequences=True) → MultiHeadAttention(4 heads, key_dim=16) → Gated residual connection (sigmoid gate + Add + LayerNorm) → last timestep → Dense(16, relu) → Dense(1)
- Key additions over plain LSTM: station metadata (lat, lon, elevation, depth) injected at every timestep; multi-head self-attention allows the model to explicitly focus on specific past timesteps
- Trained with same settings as LSTM

**TFT stability history**: In Test 5, TFT achieved R²=−0.49 (worse than predicting the mean). Root cause was normalisation leakage — test set statistics were used to compute normalisation parameters for both train and test. Once fixed in Test 7, TFT recovered to R²=0.866, matching LSTM.

---

## 5. Training and Evaluation Methodology

### 5.1 Train/Test Split

**Per-station chronological split**: Each station's data is split independently at 80% by time. The earliest 80% of each station's readings go to training; the most recent 20% form the test set. This is implemented in `station_temporal_split()` in `prepare.py`.

This approach prevents two forms of leakage:
1. **Temporal leakage**: the model never trains on data from the future of any station
2. **Station cross-contamination**: GroupKFold in hyperparameter search (in `train_models.py`) ensures the same station doesn't appear in both train and validation folds

### 5.2 Sequence model sampling

LSTM and TFT datasets contain tens of millions of possible windows. These are randomly subsampled to 100,000 examples for memory and training time constraints. Different random seeds are used for LSTM and TFT subsamples.

### 5.3 Metrics

For regression:
- **MAE** (Mean Absolute Error): average prediction error in m³/m³ — the most interpretable metric, directly in the units of soil moisture
- **RMSE** (Root Mean Squared Error): penalises large errors more than MAE
- **R²** (coefficient of determination): proportion of variance in soil moisture captured by the model (1.0 = perfect, 0.0 = no better than predicting the mean, negative = worse than the mean)

For classification (drought detection, threshold = 0.30 m³/m³):
- **Recall**: proportion of actual drought events correctly flagged — prioritised over precision because missing a drought (false negative) is worse than a false alarm for farmers
- **ROC AUC**: area under the receiver operating characteristic curve
- **PR AUC**: area under the precision-recall curve

### 5.4 Persistence Baseline

The persistence model predicts that soil moisture at time t+h equals soil moisture 24 hours ago (sm_lag_24h). This is the simplest possible forecast — "yesterday's conditions will continue." It serves as a sanity check: any model that cannot beat persistence has no practical value.

---

## 6. Experiment Progression

All experiments evaluated at **t+72h (72-hour / 3-day ahead prediction)** unless otherwise noted.

### Test 2 — Weather features added
First integration of NASA POWER weather data into the tabular feature set.

| Model | Features | MAE | R² | Recall | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | weather+lag | 0.0173 | 0.8990 | 0.969 | 0.960 |
| XGBoost | weather+lag | 0.0165 | 0.9005 | 0.970 | 0.960 |
| LSTM | sequence | 0.0182 | 0.8604 | 0.954 | 0.955 |

### Test 3 — Extended lags (5-day history)
Lag features extended to cover the full 5-day window (up to sm_lag_120h).

| Model | Features | MAE | R² | Recall | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | weather+lag | 0.0175 | 0.8955 | 0.970 | 0.958 |
| XGBoost | weather+lag | 0.0166 | 0.8984 | 0.971 | 0.959 |
| LSTM | sequence | 0.0187 | 0.8537 | 0.951 | 0.953 |

### Test 4 — TFT introduced
First appearance of the TFT model. LSTM and TFT used raw SM sequence only (no weather channels in sequence).

| Model | Features | MAE | R² | Recall | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | weather+lag | 0.0175 | 0.8955 | 0.970 | 0.958 |
| XGBoost | weather+lag | 0.0166 | 0.8984 | 0.971 | 0.959 |
| LSTM | sequence | 0.0189 | 0.8517 | 0.946 | 0.953 |
| TFT | sequence+static | 0.0275 | 0.7908 | 0.939 | 0.947 |

### Test 5 — Weather channels in sequences
Weather added as additional channels in LSTM/TFT sequences (daily values broadcast to hourly). Tree model improvement due to upstream addition of forward weather windows (t+24h, t+48h, t+72h weather features) and per-station temporal split. TFT achieved R²=−0.486 due to normalisation leakage bug (see Test 7 fix).

| Model | Features | MAE | R² | Recall | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | weather+lag | 0.0163 | 0.9216 | 0.977 | 0.970 |
| XGBoost | weather+lag | 0.0156 | 0.9244 | 0.976 | 0.971 |
| LSTM | weather+sequence | 0.0246 | 0.8504 | 0.961 | 0.958 |
| TFT | weather+sequence+static | 0.0837 | −0.486 | 0.922 | 0.901 |

### Test 6 — Rolling features
72h and 168h rolling mean and standard deviation added to tabular features. Rolling features did not improve tree model performance (XGBoost R² 0.9244 → 0.9232), suggesting the existing lag features already encode sufficient temporal trend information. XGBoost can implicitly derive trend from the set of lag values without needing explicit rolling statistics.

| Model | Features | MAE | R² | Recall | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | weather+lag | 0.0161 | 0.9205 | 0.978 | 0.970 |
| XGBoost | weather+lag | 0.0155 | 0.9232 | 0.978 | 0.971 |
| LSTM | weather+sequence | 0.0261 | 0.8408 | 0.942 | 0.957 |
| TFT | weather+sequence+static | 0.0416 | 0.6726 | 0.938 | 0.929 |

### Test 7 — ET0, LSTM dropout, tuned XGBoost, normalisation fix (FINAL)
Four changes applied simultaneously:
1. ET0 added as a derived feature (Hargreaves-Samani from existing NASA POWER variables)
2. XGBoost parameters updated: max_depth 3→5, n_estimators 100→300, subsample/colsample_bytree 0.8
3. Dropout(0.2) added after each LSTM layer
4. Normalisation statistics computed on training split only (fixes leakage in Tests 5–6)

| Model | Features | MAE | R² | Recall | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | weather+lag | 0.0160 | 0.9206 | 0.978 | 0.970 |
| **XGBoost** | **weather+lag** | **0.0149** | **0.9305** | **0.975** | **0.974** |
| LSTM | weather+sequence | 0.0225 | 0.8671 | 0.948 | 0.959 |
| TFT | weather+sequence+static | 0.0221 | 0.8661 | 0.955 | 0.962 |

XGBoost is the best-performing model across all metrics. LSTM and TFT converged to nearly identical performance (R² 0.867 vs 0.866) after the normalisation fix.

---

## 7. Horizon Analysis

Performance across all prediction horizons (Test 7, XGBoost with weather vs persistence baseline):

| Horizon | Persist MAE | Persist R² | XGB MAE | XGB R² | MAE Improvement | R² Gap |
|---|---|---|---|---|---|---|
| t+24h (1 day) | 0.0121 | 0.9363 | 0.0080 | 0.9690 | 33.9% | +0.033 |
| t+48h (2 days) | 0.0157 | 0.9083 | 0.0119 | 0.9478 | 24.2% | +0.040 |
| t+72h (3 days) | 0.0187 | 0.8822 | 0.0149 | 0.9305 | 20.3% | +0.048 |
| t+96h (4 days) | 0.0212 | 0.8566 | 0.0173 | 0.9162 | 18.4% | +0.060 |
| t+120h (5 days) | 0.0235 | 0.8313 | 0.0192 | 0.9038 | 18.3% | +0.073 |
| t+168h (7 days) | 0.0273 | 0.7864 | 0.0223 | 0.8815 | 18.3% | +0.095 |

**Key observation**: The absolute R² values look high across all horizons but this partly reflects the inherent autocorrelation of soil moisture — it is a slow-moving physical variable that changes gradually. Persistence already achieves R²=0.936 at 1 day and R²=0.786 at 7 days purely by assuming nothing changes. The model's **incremental value** (R² gap over persistence) actually **increases** with horizon — from +0.033 at 1 day to +0.095 at 7 days. This is the correct interpretation: the model adds proportionally more value as the forecast horizon extends and naive persistence becomes less reliable.

---

## 8. Why High Accuracy Is Not Suspicious

The high reported R² values (0.93 at 72h) should be interpreted carefully:

1. **Soil moisture is highly autocorrelated**: the variable changes slowly and predictably between rain events. This makes it inherently easier to predict than, say, stock prices or weather itself. Persistence achieves R²=0.88 at 72h with zero modelling effort.

2. **Forward weather windows use observed data**: the model receives actual observed future weather (T2M_t+72h, PRECTOTCORR_t+72h etc.) rather than a forecast. In real deployment, these would be weather forecast values with their own prediction errors. The reported metrics therefore represent an upper bound achievable with a perfect weather forecast.

3. **Per-station training**: the model has seen the seasonal patterns of every specific station it is evaluated on. A harder generalisation test — training on 40 stations and predicting at 8 never-seen stations — would yield lower metrics.

The appropriate frame: R²=0.93 is not a breakthrough discovery, it is a well-engineered system exploiting a predictable physical variable. The scientific contribution is the **incremental improvement over persistence** and the **agricultural applicability**.

---

## 9. Drought Onset Event Analysis and Economic Impact

### 9.1 The Distinction That Matters

Overall recall (model: 97.5%, persistence: 95.5% at t+72h) is misleadingly similar because both approaches correctly identify stations that are **already in drought**. Persistence says "SM stays the same" — correct for stations already below 0.30.

The agriculturally important case is **drought onset events**: soil moisture is currently adequate (≥ 0.30 m³/m³) but will drop below the critical threshold within 72 hours. For these events, persistence almost always fails because it predicts "SM stays adequate."

### 9.2 Empirical Results

Computed on the test set at t+72h:

| | Events caught | % caught |
|---|---|---|
| **Persistence** | 1,750 / 6,435 | **27.2%** |
| **XGBoost** | 3,602 / 6,435 | **56.0%** |

**The XGBoost model detects twice as many upcoming drought onset events as the persistence baseline.**

### 9.3 Economic Estimate

Assumptions (FAO-based, conservative):
- Primary crop: maize (dominant smallholder crop in East Africa)
- Typical smallholder yield: 2.0 tonnes/ha
- Local market price: $180/tonne → $360/ha seasonal revenue
- FAO yield response factor for maize (Ky = 1.25)
- Yield loss from one missed drought onset event (3-day water stress during sensitive period): 8% (conservative, FAO-56 methodology)
- Value of one correctly caught onset event: $360 × 0.08 = **$28.80/ha**

For a farmer experiencing 2–3 drought onset events per growing season, advance warning from the model versus relying on persistence represents approximately **$58–87/ha per season** in avoided yield losses.

On a $360/ha revenue base, this represents a **16–24% improvement in net yield**. For subsistence farmers operating on thin margins, this is a meaningful difference.

**Important caveat**: this estimate assumes the farmer has water access and acts on the warning. It does not account for irrigation infrastructure costs or the cost of false alarms. It should be presented as an illustrative order-of-magnitude estimate, not a precise economic forecast.

---

## 10. Model Comparison Summary

### 10.1 Why XGBoost Outperforms LSTM and TFT

The key reason is **feature access asymmetry**:
- XGBoost receives explicit lag features (sm_lag_24h through sm_lag_120h), weather at current time and all future windows, rolling statistics, and station metadata — all as flat columns in a table
- LSTM receives the raw 120-hour soil moisture sequence as its primary signal, with weather broadcast to hourly resolution
- The lag feature engineering essentially pre-computes the temporal structure that LSTM must learn implicitly from the raw sequence

This means the comparison is not entirely fair to LSTM. A proper comparison would give LSTM the same explicit features as XGBoost, structured as a sequence.

### 10.2 Why LSTM ≈ TFT After Fixing Normalisation

After correcting the normalisation leakage (Test 7), LSTM (R²=0.867) and TFT (R²=0.866) converged to nearly identical performance. This suggests the multi-head attention and static enrichment in TFT are not adding meaningful signal given the current data representation. Both architectures are working from the same (120, 9) input sequences and converging to the same answer.

### 10.3 Random Forest vs XGBoost

RF (R²=0.921) and XGBoost (R²=0.931) are consistently close. XGBoost's slight advantage comes from its sequential boosting (each tree corrects the errors of the previous), versus RF's independent averaging. The gap is small enough that both could be considered equivalent for most purposes.

---

## 11. Limitations

1. **Forward weather as perfect forecast**: All results use observed future weather, not a forecast. Real-world deployment requires coupling with a numerical weather prediction model, which would reduce performance — especially at longer horizons (t+96h–168h) where forecast uncertainty is high.

2. **Same-station evaluation**: Models are trained and tested on the same 48 stations. Generalisation to new, unseen stations has not been evaluated. Given the geographic diversity of the dataset (coastal, highland, urban, savanna environments), a cross-station generalisation test would be a meaningful additional evaluation.

3. **Daily weather resolution**: Weather is available daily; soil moisture is hourly. Broadcasting daily values across 24 hours means each LSTM/TFT timestep within a day carries identical weather values. This limits the ability of sequence models to exploit short-term weather dynamics.

4. **Drought threshold**: The 0.30 m³/m³ threshold used for classification is a general guideline, not calibrated to specific crops or soil types at each station. Different crops and soils have different field capacity and wilting point values.

5. **No soil type data**: The model knows station depth and elevation but not soil texture (clay, sandy loam, etc.). Soil texture fundamentally determines how soil moisture behaves and is a significant missing feature. SoilGrids provides this globally.

6. **Single-depth predictions**: Most sensors are at 0.2m depth. Root zone soil moisture (0–1m) is more relevant to crop yield decisions.

---

## 12. Future Work

In priority order:

1. **Couple with weather forecast model**: Replace observed future weather with actual NWP (numerical weather prediction) forecasts to produce operationally realistic results.

2. **Cross-station generalisation test**: Train on 40 stations, evaluate on 8 held-out stations never seen during training. This would quantify how well the model can be deployed at a new location.

3. **Add soil type features**: Fetch clay/sand/silt fraction from SoilGrids API (free, global) and include as static station features.

4. **NDVI vegetation index**: Monthly MODIS or Sentinel-2 NDVI as an additional time-varying feature — plant transpiration is the primary driver of soil moisture loss between rain events and is not currently captured.

5. **Seasonal performance analysis**: Disaggregate metrics by season (long rains: March–May, dry: June–August, short rains: October–December). Performance likely varies significantly between wet and dry seasons, with different implications for operational use.

6. **Encoder-decoder LSTM**: Give the sequence models the same weather information as XGBoost — past SM as encoder input, future weather forecasts as decoder input. This would close the fairness gap in the model comparison.

7. **Uncertainty quantification**: XGBoost with quantile regression can produce prediction intervals (e.g., "SM will be 0.24–0.28 m³/m³"). Telling a farmer a range is more useful than a single number.

---

## 13. Key Numbers for Report

| Metric | Value |
|---|---|
| Number of stations | 48 (46 TAHMO, 2 COSMOS) |
| Countries | Kenya, Uganda, Rwanda |
| Total quality-flagged readings | 1,103,461 |
| Training/test split | 80/20 chronological per station |
| Default prediction horizon | 72 hours (3 days) |
| Drought stress threshold | 0.30 m³/m³ |
| Best model | XGBoost (Test 7) |
| Best MAE | 0.0149 m³/m³ |
| Best R² | 0.9305 (at t+72h) |
| Best recall (drought detection) | 0.975 |
| Best ROC AUC | 0.974 |
| Persistence R² at t+72h | 0.882 |
| Model improvement over persistence (MAE) | 20.3% at t+72h |
| Drought onset detection — persistence | 27.2% of events caught |
| Drought onset detection — XGBoost | 56.0% of events caught |
| Estimated economic value vs persistence | $58–87/ha/season (maize, FAO-based) |
| Total features (XGBoost, t+72h) | 49 |
| Sequence length (LSTM/TFT) | 120 hours |
| Sequence channels (with weather) | 9 (SM + 7 weather + ET0) |

---

## 14. File Structure

```
ENGG2112/
├── data/
│   ├── soil_data/
│   │   ├── clean_ismn.py          # ISMN .stm → CSV cleaning pipeline
│   │   └── ismn_data/
│   │       ├── TAHMO/             # 46 station CSVs
│   │       └── COSMOS/            # 2 station CSVs
│   └── weather_data/
│       ├── fetch_weather.py       # NASA POWER API fetcher
│       └── stations/
│           ├── TAHMO/             # Per-station weather CSVs
│           └── COSMOS/
├── models/
│   ├── config.py                  # Shared constants, model hyperparameters
│   ├── prepare.py                 # Data loading, feature engineering, dataset builders
│   ├── train_models.py            # Standalone training with GridSearchCV
│   └── diagnostics.py             # Leakage checks, feature ablation
├── scripts/
│   └── run_experiment.py          # Main experiment runner — trains all 4 models,
│                                  # saves metrics + plots to images/test N/
└── images/
    ├── test 2 - weather features t+72h/
    ├── test 3 - extended lags 5 day/
    ├── test 4 - TFT/
    ├── test 5 - weather sequences/
    ├── test 6 - rolling features/
    └── test 7 - ET0 dropout tuned XGB/   ← current best
        ├── metrics.csv
        ├── horizon_comparison.csv
        ├── roc_curves.png
        ├── precision_recall.png
        ├── confusion_matrices.png
        ├── predicted_vs_actual.png
        ├── residuals.png
        ├── feature_importance.png
        ├── horizon_comparison.png
        ├── weather_impact.png
        └── recall_vs_horizon.png
```

---

*Analysis prepared from codebase and experimental results — ENGG2112 Data Farmers*
