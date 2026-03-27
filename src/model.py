import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(0)
n_samples = 2000

WATER_THRESHOLD  = 0.30
OPTIMAL_MOISTURE = 0.65
SOIL_VOLUME_L    = 10
MAX_HOURS        = 72

current_moisture     = np.random.uniform(0.15, 0.90, n_samples)
moisture_trend       = np.random.uniform(-0.015, 0.005, n_samples)
temperature_forecast = np.random.uniform(10, 42, n_samples)
humidity_forecast    = np.random.uniform(20, 95, n_samples)
rainfall_forecast_mm = np.random.exponential(scale=4, size=n_samples)
sunshine_hours       = np.random.uniform(0, 12, n_samples)

data = pd.DataFrame({
    "current_moisture":     current_moisture,
    "moisture_trend":       moisture_trend,
    "temperature_forecast": temperature_forecast,
    "humidity_forecast":    humidity_forecast,
    "rainfall_forecast_mm": rainfall_forecast_mm,
    "sunshine_hours":       sunshine_hours
})

evap_rate = (0.0018 * temperature_forecast + 0.0025 * (sunshine_hours / 12)) * (1 - humidity_forecast / 100)
net_drying_rate = np.maximum(evap_rate - (rainfall_forecast_mm / 24) * 0.003, 0.0001)
combined_rate = 0.6 * net_drying_rate + 0.4 * np.abs(moisture_trend + 1e-6)

deficit = current_moisture - WATER_THRESHOLD
hours_until_critical = np.clip(deficit / combined_rate, 0, MAX_HOURS)
hours_until_critical += np.random.normal(0, 1.5, n_samples)
hours_until_critical  = np.clip(hours_until_critical, 0, MAX_HOURS)

data["hours_until_critical"] = hours_until_critical

feature_columns = [
    "current_moisture", "moisture_trend", "temperature_forecast",
    "humidity_forecast", "rainfall_forecast_mm", "sunshine_hours"
]
X = data[feature_columns]
y = data["hours_until_critical"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestRegressor(n_estimators=150, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} hours")
print(f"R2:  {r2_score(y_test, y_pred):.4f}")

urgent_actual    = (y_test < 12).astype(int)
urgent_predicted = (y_pred < 12).astype(int)
print(f"Urgent alert accuracy (<12h): {(urgent_actual == urgent_predicted).mean():.1%}")

print("\nFeature Importance:")
importance = pd.Series(model.feature_importances_, index=feature_columns)
for feat, val in importance.sort_values(ascending=False).items():
    print(f"  {feat.replace('_', ' '):<28} {val:.3f}")


def watering_recommendation(hours_predicted, current_moist):
    water_volume_ml = max(0, (OPTIMAL_MOISTURE - current_moist) * SOIL_VOLUME_L * 1000)
    if hours_predicted <= 0:
        return f"water now — already below threshold ({water_volume_ml:.0f} ml)"
    elif hours_predicted < 12:
        return f"water soon — critical in ~{hours_predicted:.1f}h ({water_volume_ml:.0f} ml)"
    elif hours_predicted < 24:
        return f"water today — critical in ~{hours_predicted:.1f}h ({water_volume_ml:.0f} ml)"
    else:
        return f"no action needed — ok for ~{hours_predicted:.1f}h"


examples = pd.DataFrame([
    {"current_moisture": 0.32, "moisture_trend": -0.010, "temperature_forecast": 36,
     "humidity_forecast": 30, "rainfall_forecast_mm": 0.0, "sunshine_hours": 11},
    {"current_moisture": 0.22, "moisture_trend": -0.012, "temperature_forecast": 30,
     "humidity_forecast": 45, "rainfall_forecast_mm": 0.0, "sunshine_hours": 9},
    {"current_moisture": 0.55, "moisture_trend": -0.005, "temperature_forecast": 22,
     "humidity_forecast": 65, "rainfall_forecast_mm": 2.0, "sunshine_hours": 6},
    {"current_moisture": 0.80, "moisture_trend":  0.001, "temperature_forecast": 16,
     "humidity_forecast": 85, "rainfall_forecast_mm": 15.0, "sunshine_hours": 2},
])

preds = model.predict(examples)

print()
for i, (hours, (_, row)) in enumerate(zip(preds, examples.iterrows()), start=1):
    print(f"example {i}: moisture={row['current_moisture']:.0%}  temp={row['temperature_forecast']:.0f}C  "
          f"rain={row['rainfall_forecast_mm']:.0f}mm  sun={row['sunshine_hours']:.0f}h")
    print(f"  -> {watering_recommendation(hours, row['current_moisture'])}")
