import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

tickers = ['FB']
# tickers = ['AAPL', 'BRK-B', 'GOOG', 'MSFT', 'V', 'AMZN', 'FB', 'JNJ', 'PG', 'WMT']

for ticker in tickers:
  df = pd.read_csv('sentiment_analysis/results/combined/{}.csv'.format(ticker))

  data = df.filter(['Adj Close'])
  dataset = data.values
  training_data_len = math.ceil(len(dataset) * 0.80)
  training_data_len = 1227

  scaler = MinMaxScaler(feature_range = (0, 1))
  scaled_data = scaler.fit_transform(dataset)

  train_data = scaled_data[0:training_data_len, :]

  x_train, y_train = [], []
  window_size = 120

  for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])

  x_train, y_train = np.array(x_train), np.array(y_train)

    # re-shape for LSTM
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

  # build model
  model = Sequential()

  # model.add(LSTM(64, return_sequences = True, input_shape = (x_train.shape[1], 1)))
  # model.add(LSTM(32, return_sequences=False))
  # model.add(Dense(1))

  model.add(LSTM(128, return_sequences = True, input_shape = (x_train.shape[1], 1)))
  model.add(LSTM(64, return_sequences = False))
  # model.add(LSTM(64, return_sequences = True))
  # model.add(LSTM(32, return_sequences = False))
  # model.add(LSTM(1, return_sequences = False))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(1, activation='linear'))

  # # model.add(Dense(25))
  # # model.add(Dense(1))

  # model.compile(optimizer = 'adam', loss = 'mean_squared_error')
  # history = model.fit(x_train, y_train, batch_size = 64, epochs = 100)

  # parada do xande
  test_data = scaled_data[training_data_len - window_size:, :]
  x_test = []
  for i in range(window_size, len(test_data)):
    x_test.append(test_data[i-window_size:i, 0])
  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  y_test = dataset[training_data_len:, :]

  model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
  history = model.fit(x_train, y_train, batch_size = 1, epochs = 250, validation_data=(x_test, scaler.fit_transform(y_test)))

  # predictions = model.predict(x_test)
  # predictions = scaler.inverse_transform(predictions)
  # ------

  # parada do gu
  listinha_bunitinha = scaled_data[training_data_len - window_size:training_data_len]
  a = np.transpose(listinha_bunitinha)
  a = np.reshape(a, (a.shape[0], a.shape[1], 1))

  b = [dataset[training_data_len:training_data_len+30]]
  c = model.predict(a)
  listinha_bunitinha = np.concatenate((listinha_bunitinha, c))

  for i in range(30):
    a2 = np.transpose(listinha_bunitinha[i:])
    a2 = np.reshape(a2, (a2.shape[0], a2.shape[1], 1))
    c2 = model.predict(a2)
    listinha_bunitinha = np.concatenate((listinha_bunitinha, c2))

  predictions = scaler.inverse_transform(listinha_bunitinha[window_size+1:])
  # ------

  # rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
  # normalised_rmse = rmse/(max(y_test) - min(y_test))
  # with open('plots/metrics_sem', 'a') as f:
  #   f.write('{},{},{}\n'.format(ticker, rmse, normalised_rmse))

  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, 'y', label='Erro no Treinamento')
  plt.plot(epochs, val_loss, 'r', label='Erro na Validação')
  plt.xlabel('Épocas')
  plt.ylabel('Erro')
  plt.legend()
  plt.savefig('erro.png')

  train = data[:training_data_len+1]
  valid = data[training_data_len:]
  valid['Predictions'] = predictions

  plt.figure(figsize = (12, 8))
  plt.title('model')
  plt.xlabel('date', fontsize = 14)
  plt.ylabel('predicted price', fontsize = 14)
  # plt.plot(train['Adj Close'])
  plt.plot(valid[['Adj Close', 'Predictions']])
  # plt.legend(['Train', 'Validation', 'Predictions'], loc = 'lower right')
  plt.legend(['Validation', 'Predictions'], loc = 'lower right')
  plt.savefig('single_feature_fb.pdf')

  # print('training_data_len: ', training_data_len)
  # print('tamanho train: ', len(train))
  # print('tamanho valid: ', len(valid))

  # predict closing price in x days

  # copia_do_df = pd.read_csv('GOOG.csv')
  # new_df = copia_do_df.filter(['Adj Close'])

  # last_window_size_days = new_df[-window_size:].values

  # last_window_size_days_scaled = scaler.transform(last_window_size_days)

  # X_test = []
  # X_test.append(last_window_size_days_scaled)
  # X_test = np.array(X_test)
  # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

  # predicted_price = model.predict(X_test)
  # predicted_price = scaler.inverse_transform(predicted_price)
  # print(last_window_size_days)
  # print(predicted_price)
