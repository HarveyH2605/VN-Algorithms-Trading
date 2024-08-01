from vnstock3 import Vnstock
import os 
if "ACCEPT_TC" not in os.environ:
    os.environ["ACCEPT_TC"] = "tôi đồng ý"

from vnstock3 import Vnstock
stock = Vnstock().stock(symbol='HPG', source='VCI')

df = stock.quote.history(start='2022-01-01', end='2024-12-31')
df

from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Convert to datetime
df['time'] = pd.to_datetime(df['time'])
df.sort_values(by='time', inplace=True)

# Daily returns calculation
df['return'] = df['close'].pct_change().dropna()
df['return'] = np.log(1 + df['return']).dropna()

# Drop NaNs
df = df.dropna()

# Stationary testing by ADF
adf_result = adfuller(df['return'])
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
if adf_result[1] < 0.05:
    print("The return series is stationary.")
else:
    print("The return series is not stationary. Differencing might be required.")

# Plot returns
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['return'])
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Log Returns')
plt.grid(True)
plt.show()

# Fit the TGARCH model
model = arch_model(df['return'], vol='GARCH', p=1, o=1, q=1, dist='normal')
result = model.fit(disp='off')

# Print the summary
print(result.summary())
if len(df['time'].iloc[1:]) != len(result.conditional_volatility):
    print("Adjusting the lengths to match.")
    
    # Ensure the time series matches the conditional volatility
    time_series = df['time'].iloc[1:].reset_index(drop=True)  # Adjust indices to match
    volatility_series = result.conditional_volatility

    # Adjust to the smaller length
    min_length = min(len(time_series), len(volatility_series))
    time_series = time_series[:min_length]
    volatility_series = volatility_series[:min_length]
else:
    time_series = df['time'].iloc[1:]
    volatility_series = result.conditional_volatility

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(time_series, volatility_series)
plt.title('Conditional Volatility from TGARCH Model')
plt.xlabel('Date')
plt.ylabel('Conditional Volatility')
plt.show()

# Forecast the volatility for the next 5 days
forecast = result.forecast(horizon=5)
print("Forecasted variance for the next 5 days:")
print(forecast.variance[-1:])

# Plot forecasted volatility
plt.figure(figsize=(12, 6))
plt.plot(forecast.variance.values[-1, :], label='Forecasted Volatility')
plt.title('Forecasted Volatility for Next 5 Days')
plt.xlabel('Days Ahead')
plt.ylabel('Forecasted Volatility')
plt.grid(True)
plt.legend()
plt.show()

# Calculate historical volatility
window = 20  # Example window size (adjust as needed)
historical_volatility = df['close'].pct_change().rolling(window).std() * np.sqrt(window)

# Drop NaN values (due to rolling calculation)
historical_volatility = historical_volatility.dropna()

# Print first few rows of historical volatility to verify
print(historical_volatility.head())

# Plot historical volatility
plt.figure(figsize=(12, 6))
plt.plot(historical_volatility, label=f'Historical Volatility (window={window})', color='blue')
plt.title('Historical Volatility from Price Data')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.grid(True)
plt.legend()
plt.show()

# Create a GJR-GARCH model instance
model = arch_model(df['return'], vol='Garch', p=1, o=1, q=1)

# Fit the model
model_fit = model.fit(disp='off')  # Set disp='off' to suppress convergence output

# Print model summary
print(model_fit.summary())

# Forecast volatility until the end of DataFrame
forecast_horizon = len(df) - 1  # Forecast until the last observation in df
forecast = model_fit.forecast(start=df.index[0], horizon=forecast_horizon)

# Print forecasted variance
print("Forecasted variance from the beginning to the end of DataFrame:")
print(forecast.variance)
