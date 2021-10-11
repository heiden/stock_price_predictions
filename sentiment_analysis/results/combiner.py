# run this to combine the sentiment result dataframes for every year

import pandas as pd

first_year = '2016'
years = ['2016', '2017', '2018', '2019', '2020', '2021']
tickers = [
  'AAPL', 'MSFT', 'AMZN', 'FB', 'GOOG', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
  'UNH', 'V', 'PG', 'HD', 'PYPL', 'DIS', 'ADBE', 'BAC', 'MA', 'CMCSA',
  'PFE', 'NFLX', 'CRM', 'CSCO', 'XOM', 'VZ', 'ABT', 'TMO', 'KO', 'ACN',
  'PEP', 'INTC', 'DHR', 'NKE', 'LLY', 'WMT', 'COST', 'AVGO', 'ABBV', 'T',
  'MRK', 'CVX', 'WFC', 'MDT', 'MCD', 'TXN', 'NEE', 'PM', 'ORCL'
]

for year in years:
  for ticker in tickers:
    print(ticker, year)
    df = pd.read_csv('{}/{}.csv'.format(year, ticker))
    is_first = year == first_year
    df.to_csv('combined/{}.csv'.format(ticker), header=is_first, index=False, mode='a')
