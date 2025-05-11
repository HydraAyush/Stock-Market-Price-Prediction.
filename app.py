import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import chart_studio.plotly as py
import plotly.graph_objs as go

#for offline plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot;
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import plotly.express as px
from tabulate import tabulate

st.title('Stock Market Prediction')
user_input = st.text_input('Enter Stock Ticker') 


df = pdr.get_data_tiingo(user_input, api_key = "928ebe7d59276048eabdb38dac26581ad01f8f74")
df.to_csv('Data.csv')
df=pd.read_csv('Data.csv')
df['date'] = pd.to_datetime(df['date'])


#Describing Data
st.subheader('Data from 2017 - 2022')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
layout = go.Layout(
        xaxis=dict(
                title='Date'
                ),
        yaxis=dict(
                title='Price ($)'
                )
        )


stock_data = [{'x':df['date'], 'y':df['close']}]
plot = go.Figure(data=stock_data, layout=layout)

#fig = plt.figure(figsize = (12,6))
#plt.plot(df.close.to_numpy())
#st.pyplot(fig)

plot.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
st.plotly_chart(plot)

#Describing Candle Stick Chart
st.subheader('Candle Stick Chart')
fig = go.Figure(data=[go.Candlestick(x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])
st.plotly_chart(fig)

#Plotting Moving Average 100
df1 = df.reset_index()['close']
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df1.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r', label = 'MA100')
plt.plot(df.close.to_numpy(), 'b', label = 'Actual Price')
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
st.pyplot(fig)

#Plotting Moving Average 100 and 200
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df1.rolling(100).mean()
ma200 = df1.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r', label = 'MA100')
plt.plot(ma200, 'g', label = 'MA200')
plt.plot(df.close.to_numpy(), 'b', label = 'Actual Price')
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
st.pyplot(fig)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

### splitting dataset into train and test split
training_size = int(len(df1)*0.65)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size,:],df1[training_size:len(df1),:1]

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

#Load my model
#model = load_model('keras_model.h5')

### Lets Do the prediction
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(df1), label = "Actual Price")
plt.plot(trainPredictPlot, label = "Train Prediction" )
plt.plot(testPredictPlot, label = "Test Prediction")
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.show()
st.pyplot(fig2)
#st.plotly_chart(fig2)

x_input=test_data[len(test_data) - 100:].reshape(1,-1)
# temp_input
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 30 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1,n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

# visualizing plot
st.subheader('Stock Market Forecasting for next 30 days')
fig3 = plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(df1[len(df1) - 100:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.show()
st.pyplot(fig3)

fig4 = plt.figure(figsize=(12,6))
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])
plt.show()
st.pyplot(fig4)

df3=scaler.inverse_transform(df3).tolist()

fig5 = plt.figure()
plt.title("Forecasting for next 30 days", fontsize=20)
plt.xlabel("No. of days", fontsize=15)
plt.ylabel("Closing Price ($)", fontsize=15)
plt.plot(df3)
plt.show()
st.plotly_chart(fig5)
#st.pyplot(fig5)
st.subheader('Forecasted values for next 30 days')
fd = pd.DataFrame(df3, columns = ["Forecasted Prices"])
st.table(fd.tail(30).reset_index(drop=True))
