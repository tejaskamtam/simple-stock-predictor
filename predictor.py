#imports
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web ###############################
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
plt.style.use('fivethirtyeight')

#dataframe
df = web.DataReader('AAPL', data_source = 'yahoo', start='2012-01-01', end='2020-12-18') #######################
data = df.filter(['Close'])
dataset = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

#training data
train_len = math.ceil(len(dataset)*.8)
train_set = scaled[0:train_len, :]
x_train, y_train = [], []
for i in range(60, len(train_set)):
    x_train.append(train_set[i-60:i, 0])
    y_train.append(train_set[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = tf.keras.Sequential([keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1))])
model.add(keras.layers.LSTM(50, return_sequences=False))
model.add(keras.layers.Dense(25))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=7, epochs=1)

#testing data
test_set = scaled[train_len-60:, :]
x_test, y_test = [], dataset[train_len:, :]
for i in range(60, len(test_set)):
    x_test.append(test_set[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Visualize
predictions = scaler.inverse_transform(model.predict(x_test))
train, valid = data[:train_len], data[train_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
