
# Import df
from vnstock3 import Vnstock
import os 
if "ACCEPT_TC" not in os.environ:
    os.environ["ACCEPT_TC"] = "tôi đồng ý"

from vnstock3 import Vnstock
stock = Vnstock().stock(symbol='HPG', source='VCI')

df = stock.quote.history(start='2022-01-01', end='2024-12-31')
df

# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

# Set timestamp columns as index
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Log returns calculations
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df.dropna(inplace=True)

# Log returns distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['log_return'], bins=50, kde=True)
plt.title('Log Returns Distribution')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.show()

# Define MCMC model
with pm.Model() as model:
    # Define priors for unknown model parameters
    mu = pm.Normal('mu', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Sampling distribution of observations
    returns_obs = pm.Normal('returns_obs', mu=mu, sigma=sigma, observed=df['log_return'])
    
    # Inference
    trace = pm.sample(2000, return_inferencedata=True, tune=1000)

# Summary
az.plot_trace(trace)
plt.show()

# Posterior summary
az.summary(trace, hdi_prob=0.95)

# Monte Carlo simulation to forecast stock prices
n_simulations = 1000
n_days = 30
last_price = df['close'].iloc[-1]

# Generate forecast paths respecting the bounds
forecast_paths = np.zeros((n_simulations, n_days))
forecast_paths[:, 0] = last_price

# Define constants
price_fluctuation_limit = 0.07  # 7% fluctuation limit for Vietnam stock

# Upper and lower bounds
upper_bound = df['close'] * (1 + price_fluctuation_limit)
lower_bound = df['close'] * (1 - price_fluctuation_limit)

# Sampling mu and sigma from the posterior
mu_samples = trace.posterior['mu'].values.flatten()
sigma_samples = trace.posterior['sigma'].values.flatten()

# Generate forecast paths respecting the bounds
for i in range(n_simulations):
    for t in range(1, n_days):
        mu_t = np.random.choice(mu_samples)
        sigma_t = np.random.choice(sigma_samples)
        
        # Ensure the simulated price respects the bounds
        price_today = forecast_paths[i, t-1] * np.exp(np.random.normal(mu_t, sigma_t))
        
        # Apply the upper and lower bounds
        if price_today > upper_bound.iloc[-1]:
            price_today = upper_bound.iloc[-1]
        elif price_today < lower_bound.iloc[-1]:
            price_today = lower_bound.iloc[-1]
        
        forecast_paths[i, t] = price_today

# Plotting the forecast paths
plt.figure(figsize=(12, 6))
plt.plot(range(n_days), forecast_paths.T, color='blue', alpha=0.1)
plt.title('Monte Carlo Simulations of Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

# Analyze the forecast
final_prices = forecast_paths[:, -1]
sns.histplot(final_prices, bins=50, kde=True)
plt.title('Distribution of Forecasted Stock Prices in 30 Days')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Print statistics
print(f'Expected price in 30 days: {np.mean(final_prices)}')
print(f'95% Confidence Interval: {np.percentile(final_prices, [2.5, 97.5])}')

# Forecasted price for the next day
forecast_price_next_day = np.mean(forecast_paths[:, -1])

# Current closing price
current_price = df['close'].iloc[-1]

# Simple trading signal
if forecast_price_next_day > current_price:
    trading_signal = 'Buy'
else:
    trading_signal = 'Sell'

print(f'Today\'s Closing Price: {current_price}')
print(f'Forecasted Price for Next Day: {forecast_price_next_day}')
print(f'Trading Signal: {trading_signal}')

# Convert time index
df.index = pd.to_datetime(df.index)

# Define the mean (mu) and standard deviation (sigma) from the MCMC model
mu = -0.0000  # Mean return
sigma = 0.024  # Standard deviation of returns

# Function to generate extended forecast prices based on MCMC model
def generate_extended_forecast(df, mu, sigma):
    start_date = df.index[0]
    end_date = df.index[-1]
    forecast_index = pd.date_range(start_date, end_date, freq='B')  # Business days only
    
    initial_price = df['close'].iloc[0]
    
    forecast_prices = [initial_price]
    for _ in range(len(forecast_index) - 1):
        next_price = forecast_prices[-1] * np.exp(np.random.normal(mu, sigma))
        forecast_prices.append(next_price)
    
    forecast_df = pd.DataFrame({'Forecast': forecast_prices}, index=forecast_index)
    return forecast_df

# Function to plot historical prices with forecast
def plot_historical_and_forecast(df, forecast_df):
    plt.figure(figsize=(14, 7))
    
    # Plot historical prices
    plt.plot(df.index, df['close'], label='Historical Prices', color='blue')
    
    # Plot forecast
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast Prices', color='orange', linestyle='--')
    
    # Titles and labels
    plt.title('Historical Prices with Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate extended forecast from the start to the end of the historical data
extended_forecast_df = generate_extended_forecast(df, mu, sigma)

# Plotting the historical prices and extended forecast
plot_historical_and_forecast(df, extended_forecast_df)