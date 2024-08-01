from vnstock3 import Vnstock
import os 
if "ACCEPT_TC" not in os.environ:
    os.environ["ACCEPT_TC"] = "tôi đồng ý"

from vnstock3 import Vnstock
stock = Vnstock().stock(symbol='HPG', source='VCI')


df = stock.quote.history(start='2022-01-01', end='2024-12-31')
df

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Log returns
r_t = np.log(df['close'] / df['close'].shift(1)).values[1:]
mean = np.nanmean(r_t)
r_t[0]=mean
r_t[:5]

# Plot return rate
plt.figure(figsize=(16, 4))
plt.plot(np.arange(r_t.shape[0]), r_t, '-o')
plt.axhline(y=mean, label='mean return', c='red')
plt.title('Return rate according to date')
plt.xlabel('Date Order')
plt.ylabel('Return Rate')
plt.legend()
plt.show()

# Plot Return lag vs Lag order
plt.figure(figsize=(8, 8))
plt.scatter(x=r_t[1:], y=r_t[:-1])
plt.title('Return rate vs Lag order 1 according to date')
plt.xlabel('r(t-1)')
plt.ylabel('r(t)')
plt.show()

# Plot distribution return
plt.figure(figsize = (8, 6))
sns.distplot(r_t, bins = 20)
plt.axvline(x=mean, label='mean return', c='red')
plt.title('Distribution return of VND')
plt.legend()
plt.xlabel('return rate')
plt.ylabel('frequency')

sm.qqplot(r_t)
plt.show()
tq = stats.probplot(r_t)
plt.scatter(x=tq[0][0], y = tq[0][1])
plt.show()

# ADF test
result = adfuller(r_t)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

# Plot autocorrelation and partial correlation
plt.figure(figsize = (10, 8))
ax1 = plot_acf(r_t)
plt.figure(figsize = (8, 6))
ax2 = plot_pacf(r_t)

# ARIMA summary
model_arima = ARIMA(r_t, order = (4, 0, 4))
model_fit = model_arima.fit()
print(model_fit.summary())

# AIC and BIC for model parameters determination
def _arima_fit(orders, data):
  models = dict()
  for order in orders:
    model = ARIMA(data, order = order).fit()
    model_name = 'ARIMA({},{},{})'.format(order[0], order[1], order[2])
    print('{} --> AIC={}; BIC={}'.format(model_name, model.aic, model.bic))
    models[model_name] = model
  return models

orders = [(2, 0, 2), (2, 0, 0), (5, 0, 0), (0, 0, 5)]
models = _arima_fit(orders, r_t)