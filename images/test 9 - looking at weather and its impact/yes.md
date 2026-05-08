Fig 1 – R² line plots: Weather and no-weather lines are nearly identical across every test and horizon. The gap never meaningfully opens up.
Fig 2 – Impact heatmap: Confirms it numerically — ΔR² and ΔMAE are near-zero across the board, regardless of test or horizon.
Fig 3 – % improvement bars: Even as a percentage, the best-case gain is ~5.5% at t+168h. At practical short horizons (t+24h, t+48h) it's under 1%.
Fig 4 – Model comparison: XGBoost and RF are completely flat as weather is added. LSTM and TFT are volatile, and TFT sometimes gets worse with weather.
Fig 5 – Correlations: The root cause — weather variables are weakly correlated with soil moisture to begin with. Precipitation is only r = +0.10, wind speed r = −0.03.
Fig 6 – Summary table: Hard numbers confirming all of the above in one place.
