import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(0)
n_samples = 2000

# --- constants ---
WATER_THRESHOLD  = 0.30   # water the plant when soil drops below this
OPTIMAL_MOISTURE = 0.65   # target moisture level after watering
SOIL_VOLUME_L    = 10     # assumed soil volume in litres (e.g. a medium garden bed)
MAX_HOURS        = 72     # cap prediction at 72 hours ahead

# --- simulate sensor + weather inputs ---
current_moisture     = np.random.uniform(0.15, 0.90, n_samples)
moisture_trend       = np.random.uniform(-0.015, 0.005, n_samples)  # moisture change per hour (mostly drying)
temperature_forecast = np.random.uniform(10, 42, n_samples)         # degrees C, next 24h average
humidity_forecast    = np.random.uniform(20, 95, n_samples)         # %, next 24h average
rainfall_forecast_mm = np.random.exponential(scale=4, size=n_samples)    # mm expected in next 24h
sunshine_hours       = np.random.uniform(0, 12, n_samples)          # forecast sunshine hours

data = pd.DataFrame({
    "current_moisture":     current_moisture,
    "moisture_trend":       moisture_trend,
    "temperature_forecast": temperature_forecast,
    "humidity_forecast":    humidity_forecast,
    "rainfall_forecast_mm": rainfall_forecast_mm,
    "sunshine_hours":       sunshine_hours
})

# --- calculate target: hours until soil hits the critical threshold ---
# evaporation rate (fraction per hour) rises with heat and sun, drops with humidity
evap_rate = (0.0018 * temperature_forecast + 0.0025 * (sunshine_hours / 12)) * (1 - humidity_forecast / 100)

# rainfall slows drying (spread over 24h)
net_drying_rate = np.maximum(evap_rate - (rainfall_forecast_mm / 24) * 0.003, 0.0001)

# combine with actual sensor trend for a better estimate
combined_rate = 0.6 * net_drying_rate + 0.4 * np.abs(moisture_trend + 1e-6)

# how long until moisture hits the threshold
deficit = current_moisture - WATER_THRESHOLD
hours_until_critical = deficit / combined_rate
hours_until_critical = np.clip(hours_until_critical, 0, MAX_HOURS)

# add a bit of noise to make it realistic
hours_until_critical += np.random.normal(0, 1.5, n_samples)
hours_until_critical  = np.clip(hours_until_critical, 0, MAX_HOURS)

data["hours_until_critical"] = hours_until_critical

# --- train/test split ---
feature_columns = [
    "current_moisture", "moisture_trend", "temperature_forecast",
    "humidity_forecast", "rainfall_forecast_mm", "sunshine_hours"
]
X = data[feature_columns]
y = data["hours_until_critical"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# --- train model ---
model = RandomForestRegressor(n_estimators=150, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# --- model accuracy ---
print("=" * 45)
print("  SOIL MOISTURE PREDICTION MODEL")
print("=" * 45)
print(f"\nModel Performance (test set, n={len(y_test)})")
print(f"  MAE: {mean_absolute_error(y_test, y_pred):.2f} hours")
print(f"  R²:  {r2_score(y_test, y_pred):.4f}")

# check how often the model correctly identifies urgent cases (< 12 hours)
urgent_actual    = (y_test < 12).astype(int)
urgent_predicted = (y_pred < 12).astype(int)
urgent_accuracy  = (urgent_actual == urgent_predicted).mean()
print(f"\n  Urgent watering alert accuracy (<12h): {urgent_accuracy:.1%}")

# --- feature importance ---
print("\nFeature Importance:")
importance = pd.Series(model.feature_importances_, index=feature_columns)
for feat, val in importance.sort_values(ascending=False).items():
    print(f"  {feat.replace('_', ' ').title():<28} {val:.3f}")

# --- helper: turn a prediction into an actionable recommendation ---
def watering_recommendation(hours_predicted, current_moist):
    water_volume_ml = max(0, (OPTIMAL_MOISTURE - current_moist) * SOIL_VOLUME_L * 1000)

    if hours_predicted <= 0:
        return f"WATER NOW  — already below threshold  ({water_volume_ml:.0f} ml recommended)"
    elif hours_predicted < 12:
        return f"Water soon — critical in ~{hours_predicted:.1f}h        ({water_volume_ml:.0f} ml recommended)"
    elif hours_predicted < 24:
        return f"Water today — critical in ~{hours_predicted:.1f}h       ({water_volume_ml:.0f} ml recommended)"
    else:
        return f"No action  — OK for ~{hours_predicted:.1f}h"

# --- run some realistic sensor examples through the model ---
print("\n" + "=" * 45)
print("  SAMPLE SENSOR READINGS")
print("=" * 45)

examples = pd.DataFrame([
    # dry soil, hot sunny day forecast, no rain
    {"current_moisture": 0.32, "moisture_trend": -0.010, "temperature_forecast": 36,
     "humidity_forecast": 30, "rainfall_forecast_mm": 0.0, "sunshine_hours": 11},
    # already critical, been drying fast
    {"current_moisture": 0.22, "moisture_trend": -0.012, "temperature_forecast": 30,
     "humidity_forecast": 45, "rainfall_forecast_mm": 0.0, "sunshine_hours": 9},
    # moderate moisture, mild day
    {"current_moisture": 0.55, "moisture_trend": -0.005, "temperature_forecast": 22,
     "humidity_forecast": 65, "rainfall_forecast_mm": 2.0, "sunshine_hours": 6},
    # well watered, rain forecast
    {"current_moisture": 0.80, "moisture_trend":  0.001, "temperature_forecast": 16,
     "humidity_forecast": 85, "rainfall_forecast_mm": 15.0, "sunshine_hours": 2},
])

preds = model.predict(examples)

for i, (hours, (_, row)) in enumerate(zip(preds, examples.iterrows()), start=1):
    rec = watering_recommendation(hours, row["current_moisture"])
    print(f"\n  Example {i}:")
    print(f"    Moisture: {row['current_moisture']:.0%}  |  "
          f"Temp: {row['temperature_forecast']:.0f}°C  |  "
          f"Rain: {row['rainfall_forecast_mm']:.0f}mm  |  "
          f"Sun: {row['sunshine_hours']:.0f}h")
    print(f"    → {rec}")

print()
