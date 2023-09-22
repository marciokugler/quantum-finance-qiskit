import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Define the parameters of the simulation
stock_ticker = 'AAPL'
start_date = '2023-01-01'
end_date = '2023-09-11'
start_date_pred = '2023-08-11'
end_date_pred = '2023-09-20'
future_dates = pd.date_range(start=start_date_pred, end=end_date_pred, freq='B')

# Get the historical stock prices from Yahoo Finance
stock_data = yf.download(stock_ticker, start=start_date, end=end_date)['Adj Close']
stock_data = stock_data.asfreq('B')
stock_data = stock_data.dropna()
# Create a DatetimeIndex object for the future dates
future_dates_index = pd.DatetimeIndex(future_dates)

# Set the index of the stock_data DataFrame to the DatetimeIndex object
stock_data.index = pd.DatetimeIndex(stock_data.index)

# Normalize the input features
scaler = StandardScaler()
X = np.arange(len(stock_data)).reshape(-1, 1)
X_scaled = scaler.fit_transform(X)
# Remove the rows with NaN values from the y variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, stock_data.values.reshape(-1, 1), test_size=0.2, random_state=42)

# Define the hyperparameters to tune
hyperparameters = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'n_jobs': [1, 2, 4, 8],
    'positive': [True, False]
}

# Create a grid search object
grid_search = GridSearchCV(LinearRegression(), hyperparameters, cv=5)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(f'Best hyperparameters: {grid_search.best_params_}')

# Use the best hyperparameters to train the model
model = LinearRegression(**grid_search.best_params_).fit(X_train, y_train)
# Use the model to predict the stock prices for the testing set
y_pred = model.predict(X_test)
# Predict the stock prices for a range of future dates using the linear regression model

future_prices_lr = []
for date in future_dates:
    future_price = model.predict(scaler.transform([[len(stock_data) + (date - future_dates[0]).days]])).reshape(-1)
    print(future_price)
    future_prices_lr.append(future_price)

# Extract the values from the NumPy arrays in the future_prices_lr list
future_prices_lr = [price[0] for price in future_prices_lr]

# Use ARIMA to predict the stock prices for the future dates
model_arima = ARIMA(stock_data, order=(1, 1, 1)).fit()
print (model_arima.summary())
future_prices_arima = model_arima.forecast(steps=len(stock_data))
future_prices_arima = [price for price in future_prices_arima.values]

#if len(future_prices_arima) > 0:
#    future_prices_arima = future_prices_arima.values.reshape(-1, 1)


# Print the predicted stock prices for the future dates using the linear regression model
#for i, date in enumerate(len(stock_data)):
#    print(f'Predicted stock price for {date.date()} (Linear Regression): {future_prices_lr[i]:.2f}')
#    print(f'Predicted ARIMA stock price for {date.date()} (ARIMA): {future_prices_arima[i]:.2f}')


# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (Linear Regression): {mse:.2f}')

#mse_arima = mean_squared_error(y_test[future_dates], future_prices_arima)
mse_arima = mean_squared_error(stock_data, future_prices_arima)

print(f'Mean Squared Error (ARIMA): {mse_arima:.2f}')

# Plot the actual and predicted stock prices with dates on the x-axis
plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Stock Prices')
plt.plot(stock_data.index[-len(y_test):], y_pred, label='Predicted Stock Prices (Linear Regression)')
plt.plot(stock_data.index[-len(y_test):], future_prices_arima[-len(y_test):], label='Predicted Stock Prices (ARIMA)')

plt.plot(future_dates, future_prices_lr, color='r', label='Predicted Prices (Linear Regression)')
#plt.plot(future_dates, future_prices_arima, color='g', label='Predicted Prices (ARIMA)')
plt.legend()
plt.title('Actual vs. Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()