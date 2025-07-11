# Solar-GHI-Forecasting
Forecasting next-day Global Horizontal Irradiance (GHI) using historical weather data. The project uses XGBoost to model solar patterns from time-series data, handling noisy, skewed values and accounting for day-night variations in solar exposure.

## Project Overview 

This project focuses on forecasting day-ahead Global Horizontal Irradiance (GHI) using past environmental and weather data. The aim is to accurately predict solar irradiance values for the next day, particularly accounting for periods like sunrise, sunset, and night when GHI naturally drops to zero. The final model is built using the XGBoost Regressor and evaluated with a customised Mean Absolute Percentage Error (MAPE) metric that handles zero-GHI edge cases.


### Dataset Description 

The dataset consists of hourly records of solar and weather parameters. Some of the key features include:
1. ghi (target): Global Horizontal Irradiance
2. irradiance_global_reference: Measured irradiance from a reference station
3. wind_speed, wind_direction, temperature, humidity, etc.
4. timestamp: Hourly datetime stamps

The data showed cyclic solar patterns: high GHI during the day and nearly zero during the night. However, it also contained missing values, abrupt spikes, and noisy measurements.

### Data Cleaning & Preprocessing

1. Dropped Irrelevant Columns: Removed (Unnamed: 0) and (timestamp), as time features were extracted separately.
2. Missing Values: Dropped columns with ≥30% missing data; remaining missing values filled with KNN imputation (5 neighbours).
3. Outlier Handling: Removed top 1% of GHI values to eliminate unrealistic spikes.
4. Wind Direction Encoding: Transformed using cos(θ) + sin(θ) to preserve its cyclical nature.
5. Exponential Moving Average (EMA): Applied with (span=24) to smooth all numeric features (except hour).
6. Log1p Transformation: Applied to GHI to reduce skewness and stabilise variance.
7. Hour Feature: Extracted from (datetime) to capture daily solar cycles.

### Exploratory Data Analysis (EDA)

1. Time Series Plot of GHI: Confirmed strong periodicity with flat zero values at night.
2. Distribution of GHI: Right-skewed distribution, further validating the need for log transformation.
3. Missing Value Heatmaps: Used to visually inspect column-wise missingness.

### Model Selection - XGBoost

- Excellent performance on tabular data.
- Robustness to outliers and non-linear relationships.
- Built-in regularisation and efficient training.
- Ability to handle missing data natively.

#### Hyperparamters:
```
n_estimators=100
learning_rate=0.1
max_depth=5
random_state=42
```

### Post-processing Logic

After predicting log-transformed GHI and applying np.expm1, predictions were further refined:
1. Set Predicted GHI = 0 where irradiance_global_reference < 5.
This mimics physical reality during night/low-sunlight hours and prevents inflated MAPE.

### Custom MAPE Evaluation

Sklearn's default MAPE does not handle actual=0 well. 

So I implemented a custom logic:

```
If actual = 0 and predicted = 0: APE = 0
If actual = 0 and predicted ≠ 0: APE = 100
Else: APE = abs(actual - predicted)/actual * 100
```

This approach ensured fairness in evaluation during zero-radiation periods.

### Future Improvements

- Incorporate additional temporal features like cloud cover lag, solar zenith angle, etc.

- Try deep learning models like LSTMs or CNNs for better sequence learning.

- Use ensemble of multiple models to capture both trend and variance.

- Deploy the model as a REST API or integrate into a live forecasting dashboard.


