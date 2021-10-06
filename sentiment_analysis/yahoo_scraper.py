import os
import yfinance as yf

years = ['2016', '2017', '2018', '2019', '2020']

tickers = [
  'AAPL', 'MSFT', 'AMZN', 'FB', 'GOOG', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
  'UNH', 'V', 'PG', 'HD', 'PYPL', 'DIS', 'ADBE', 'BAC', 'MA', 'CMCSA', 'PFE',
  'NFLX', 'CRM', 'CSCO', 'XOM', 'VZ', 'ABT', 'TMO', 'KO', 'ACN', 'PEP',
  'INTC', 'DHR', 'NKE', 'LLY', 'WMT', 'COST', 'AVGO', 'ABBV', 'T', 'MRK',
  'CVX', 'WFC', 'MDT', 'MCD', 'TXN', 'NEE', 'PM', 'ORCL'
]

for year in years:
  if not os.path.exists('stocks/{}'.format(year)):
    os.makedirs('stocks/{}'.format(year))

  for ticker in tickers:
    data = yf.download(ticker, start='{}-01-01'.format(year), end='{}-12-31'.format(year))
    data.to_csv('stocks/{0}/{1}.csv'.format(year, ticker))
