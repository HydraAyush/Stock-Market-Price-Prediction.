import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# # Load data
# @st.cache
# def load_data():
#     data = pd.read_csv('BSE Dataset\ITC.csv')
#     return data

# Load data
@st.cache
def load_data():
    data = pd.read_csv('BSE Dataset\ITC.csv', parse_dates=['Date'], index_col='Date')
    return data


# ARIMA model
def arima_model(train_data, test_data):
    model = ARIMA(train_data, order=(5,1,0))  # You can tune the order parameter
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test_data))
    return predictions

# Creating DataFrame from forecast values
def create_forecast_df(test_data, predictions):
    forecast_df = pd.DataFrame({'Date': test_data.index, 'Actual': test_data.values, 'Predicted': predictions})
    return forecast_df

# Main function
def main():
    st.title('Stock Price Prediction')

    # Load data
    data = load_data()

    # Show raw data
    st.subheader('Raw Data')
    st.write(data)

    # Visualize closing prices
    st.subheader('Closing Prices')
    st.line_chart(data['Close'])

    # Manually preset train-test split
    train_size = st.number_input('Enter training data size:', min_value=1, max_value=len(data)-1, value=int(len(data)*0.8), step=1)
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

    # Build and evaluate ARIMA model
    predictions = arima_model(train_data['Close'], test_data['Close'])
    rmse = np.sqrt(mean_squared_error(test_data['Close'], predictions))

    # Show predictions
    st.subheader('Predictions vs Actual')
    fig, ax = plt.subplots()
    ax.plot(test_data.index, test_data['Close'], label='Actual')
    ax.plot(test_data.index, predictions, color='red', label='Predicted')
    ax.set_title('ARIMA Predictions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Show RMSE
    st.subheader('Root Mean Squared Error (RMSE)')
    st.write(rmse)

    # Create DataFrame from forecast values
    forecast_df = create_forecast_df(test_data['Close'], predictions)
    st.subheader('Forecast DataFrame')
    st.write(forecast_df)

if __name__ == '__main__':
    main()
