import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# Define date parsing function
dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')

# Load data
data = pd.read_csv('BSE Dataset\TATAMOTORS.csv', sep=',', index_col='Date', parse_dates=['Date'], date_parser=dateparse).fillna(0)

# Convert index to DatetimeIndex and set frequency
df_close = data['Close']
df_close.index = pd.to_datetime(df_close.index)
df_close = df_close.asfreq('D')  # Assuming daily frequency

# Plot close price
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Close Prices')
plt.plot(df_close)
plt.title('Tata Motors closing price')
plt.show()

# Test for stationarity
def test_stationarity(timeseries):
    # Remove NaN and infinity values
    timeseries = timeseries.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Determine rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    
    # Plot rolling statistics
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    # Dickey-Fuller test
    print("Results of Dickey-Fuller Test:")
    adft = adfuller(timeseries, autolag='AIC')
    output = pd.Series(adft[0:4], index=['Test Statistic', 'p-value', 'No. of Lags Used', 'Number of Observations Used'])
    for key, value in adft[4].items():
        output['Critical Value (%s)' % key] = value
    print(output)

test_stationarity(df_close)


# Decompose the time series
result = seasonal_decompose(df_close, model='multiplicative')
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(16, 9)
plt.show()

# Eliminate trend
rcParams['figure.figsize'] = 10, 6
df_log = np.log(df_close)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.plot(std_dev, color="black", label="Standard Deviation")
plt.plot(moving_avg, color="red", label="Mean")
plt.legend()
plt.title('Moving Average')
plt.show()

# Split data into train and test sets
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()

# Build ARIMA model
model = ARIMA(train_data, order=(3, 1, 2))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(544, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

# Plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='Training')
plt.plot(test_data, color='blue', label='Actual Stock Price')
plt.plot(fc_series, color='orange', label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
plt.title('Tata Motors Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Tata Motors Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Report performance
mse = mean_squared_error(test_data, fc)
print('Mean Squared Error (MSE):', mse)
mae = mean_absolute_error(test_data, fc)
print('Mean Absolute Error (MAE):', mae)
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('Root Mean Squared Error (RMSE):', rmse)
mape = np.mean(np.abs(fc - test_data) / np.abs(test_data))
