#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

def set_ticks():
  return [
    0, 124, # 0, 60, 124, 188,
    252, 377, # 252, 313, 377, 439,
    502, 628, # 502, 563, 628, 690,
    752, 876, # 752, 813, 876, 940,
    1004, 1129 # 1004, 1066, 1129, 1193
  ]

def set_labels():
  return [
    'Jan/2016', 'Jul/2016', # 'Jan/2016', 'Abr/2016', 'Jul/2016', 'Out/2016',
    'Jan/2017', 'Jul/2017', # 'Jan/2017', 'Abr/2017', 'Jul/2017', 'Out/2017',
    'Jan/2018', 'Jul/2018', # 'Jan/2018', 'Abr/2018', 'Jul/2018', 'Out/2018',
    'Jan/2019', 'Jul/2019', # 'Jan/2019', 'Abr/2019', 'Jul/2019', 'Out/2019',
    'Jan/2020', 'Jul/2020' # 'Jan/2020', 'Abr/2020', 'Jul/2020', 'Out/2020'
  ]

tickers = ['AAPL', 'BRK-B', 'GOOG', 'MSFT', 'V', 'AMZN', 'FB', 'JNJ', 'PG', 'WMT']

for ticker in tickers:
  df = pd.read_csv('sentiment_analysis/results/combined/{}.csv'.format(ticker))
  # print(df)

  data = df.filter(['Adj Close', 'Sentiment'])
  dataset = data.values
  # print(dataset)
  training_data_len = math.ceil(len(dataset) * 0.80)

  scaler = MinMaxScaler(feature_range = (0, 1))
  scaled_adj_close = scaler.fit_transform(dataset[:, 0].reshape(-1, 1))
  scaled_data = []
  for i in range(len(dataset)):
    scaled_data.append(np.array([scaled_adj_close[i][0], dataset[i][-1]]))
  scaled_data = np.array(scaled_data)

  train_data = scaled_data[0:training_data_len, :]

  x_train, y_train = [], []
  window_size = 60

  for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0:train_data.shape[1]-1])
    y_train.append(np.array([train_data[i, 0]])) # maybe remove np.array([])

  x_train, y_train = np.array(x_train), np.array(y_train)

  # re-shape for LSTM
  # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

  # build model
  model = Sequential()
  model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
  model.add(LSTM(50, return_sequences = False))
  model.add(Dense(25))
  model.add(Dense(1))

  model.compile(optimizer = 'adam', loss = 'mean_squared_error')
  model.fit(x_train, y_train, batch_size = 128, epochs = 50)

  test_data = scaled_data[training_data_len - window_size:, 0]
  x_test = []
  for i in range(window_size, len(test_data)):
    x_test.append(test_data[i-window_size:i])
  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  y_test = dataset[training_data_len:, 0]

  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)

  rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
  normalised_rmse = rmse/(max(y_test) - min(y_test))
  with open('plots/metrics', 'a') as f:
    f.write('{},{},{}\n'.format(ticker, rmse, normalised_rmse))

  train = data[:training_data_len]
  valid = data[training_data_len:]
  valid['Predictions'] = predictions

  plt.figure(figsize = (12, 8))
  # plt.title('model')
  plt.xlabel('Data')
  plt.ylabel('Cotação (USD)')
  plt.plot(train['Adj Close'])
  plt.plot(valid[['Adj Close', 'Predictions']])
  plt.xticks(set_ticks(), set_labels(), rotation = 45)
  plt.legend(['Treinamento', 'Validação', 'Previsão'], loc = 'lower right')
  plt.savefig('plots/{}.png'.format(ticker))

  # print('training_data_len: ', training_data_len)
  # print('tamanho train: ', len(train))
  # print('tamanho valid: ', len(valid))

  # predict closing price in x days

  # copia_do_df = pd.read_csv('sentiment_analysis/results/combined/{}.csv'.format(ticker))
  # new_df = copia_do_df.filter(['Adj Close'])

  # last_window_size_days = new_df[-window_size:].values

  # last_window_size_days_scaled = scaler.transform(last_window_size_days)

  # X_test = []
  # X_test.append(last_window_size_days_scaled)
  # X_test = np.array(X_test)
  # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

  # predicted_price = model.predict(X_test)
  # predicted_price = scaler.inverse_transform(predicted_price)
  # # print(last_window_size_days)
  # print(predicted_price)
