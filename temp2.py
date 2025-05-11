import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load data
@st.cache
def load_data():
    data = pd.read_csv('BSE Dataset\TATAMOTORS.csv', parse_dates=['Date'], index_col='Date')
    return data

# # ARIMA model
# def arima_model(train_data, test_data):
#     model = ARIMA(train_data, order=(5,1,0))  # Adjust the order parameter
#     model_fit = model.fit()
#     predictions = model_fit.forecast(steps=len(test_data))[0]  # Extract only the forecast values
#     return predictions

# ARIMA model
def arima_model(train_data, test_data):
    print("Train data shape:", train_data.shape)  # Debug statement
    print("Test data shape:", test_data.shape)    # Debug statement
    model = ARIMA(train_data, order=(3,1,4))  # Adjust the order parameter
    try:
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test_data))[0]  # Extract only the forecast values
        print("Model fitting successful.")  # Debug statement
    except Exception as e:
        print("Error in model fitting:", e)  # Debug statement
        predictions = np.zeros(len(test_data))  # Return array of zeros if model fitting fails
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

    # Ensure 'Close' column contains numeric values
    train_data['Close'] = pd.to_numeric(train_data['Close'], errors='coerce')
    test_data['Close'] = pd.to_numeric(test_data['Close'], errors='coerce')

    # Drop rows with missing 'Close' values
    train_data.dropna(subset=['Close'], inplace=True)
    test_data.dropna(subset=['Close'], inplace=True)

    # Build and evaluate ARIMA model
    predictions = arima_model(train_data['Close'], test_data['Close'])
    rmse = np.sqrt(mean_squared_error(test_data['Close'], predictions))

    # Show predictions
    st.subheader('Predictions vs Actual')
    fig, ax = plt.subplots()
    ax.plot(test_data.index, test_data['Close'], label='Actual', color='blue')
    ax.plot(test_data.index, predictions, label='Predicted', color='red')
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
