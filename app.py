import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime


start='2010-01-01'
end = '2022-10-31'

st.title('Stock Market Prediction APP')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

#selected_stock = st.text_input("Select dataset for prediction")

#n_years = st.slider("Years Of Prediction:", 0, 4)
#period = n_years * 365

#user_input =st.text_input( 'yahoo', start, date)
#df= st.selectbox("Years of Prediction" , user_input)

#Describing the data
st.subheader('Data from 2010-2022(OCT)')
st.write(df.describe())

#visulizations of the site

# 1st we are making simple closing price chart
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.xlabel('Closing Price')
plt.ylabel('Time chart ')
st.pyplot(fig)



#now graph for ma100

st.subheader('Closing Price vs Time chart with ma100')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(df.Close)
plt.xlabel('Closing Price')
plt.ylabel('Time chart with ma100')
st.pyplot(fig)



#now graph for ma200

st.subheader('Closing Price vs Time chart with ma100 & ma200')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
plt.xlabel('Closing Price')
plt.ylabel('Time chart with ma100 & ma200')
st.pyplot(fig)


#splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
#Spliting data into x_train and y_train
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train,y_train=np.array(x_train), np.array(y_train)

#Load my model

model = load_model('keras_model.h1')

#Testing Part

past_100_days= data_training.tail(100)

final_df=past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test=np.array(x_test), np.array(y_test)


#making predictions

y_predicted=model.predict(x_test)

scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted =y_predicted*scale_factor
y_test=y_test*scale_factor

#final graph

st.subheader('Preditcions vs Original Price ')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b' , label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
