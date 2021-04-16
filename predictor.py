import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# sample: Petrobras' stock
# df = web.DataReader('AAPL', data_source = 'yahoo', start = '2012-01-01', end = '2019-12-17')
df = pd.read_csv('PETR4.SA.csv')
# print(df)

# plt.figure(figsize = (16, 8))
# plt.title('close price history')
# plt.plot(df['Close'])
# plt.xlabel('date', fontsize = 10)
# plt.ylabel('close price', fontsize = 10)
# plt.show()

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.80)

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]

x_train, y_train = [], []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# re-shape for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# build model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

test_data = scaled_data[training_data_len - 60:, :]
x_test = []
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = dataset[training_data_len:, :]

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize = (16, 8))
plt.title('model')
plt.xlabel('date', fontsize = 18)
plt.ylabel('predicted price', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc = 'lower right')
plt.show()

# check the quota... didnt really understand (41:50)
# predict closing price in x days

copia_do_df = pd.read_csv('PETR4.SA.csv')
new_df = copia_do_df.filter(['Close'])

last_60_days = new_df[-60:].values

last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
print(last_60_days)
print(predicted_price)
