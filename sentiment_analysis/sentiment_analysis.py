import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def avg(list):
  if len(list) == 0:
    return 0
  else:
    return sum(list) / len(list)

def export(dataframe, year, ticker):
  header = 'Date,Open,High,Low,Close,Adj Close,Volume,Sentiment\n'

  with open('results/{}/{}.csv'.format(year, ticker), 'w') as file:
    file.write(header)
    for line in dataframe.values:
      file.write(','.join(map(str, line)))
      file.write('\n')


# years = ['2016', '2017', '2018']
# tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'BRK-B', 'V', 'WMT', 'JNJ', 'PG']
years = ['2016']
tickers = ['GOOG']

for year in years:
  for ticker in tickers:
    print(ticker, year)
    df = pd.read_csv('news/{}/{}.csv'.format(year, ticker), delimiter=';')
    headlines = df['Headline'].values
    dates = [x for x in zip(df['Day'], df['Month'])]

    # using 1-31 for day and 1-12 for month to be clearer
    sentiments = [[[] for x in range(13)] for y in range(32)]
    analyzer = SentimentIntensityAnalyzer()

    for pair in zip(headlines, dates):
      headline, date = pair
      vader_sentiment_scores = analyzer.polarity_scores(headline)
      sentiments[date[0]][date[1]].append(vader_sentiment_scores['compound'])

    df = pd.read_csv('stocks/{}/{}.csv'.format(year, ticker))
    trading_dates = df['Date'].values

    sentiment_column = []
    for date in trading_dates:
      year, month, day = list(map(int, date.split('-')))
      sentiment_column.append(avg(sentiments[day][month]))

    df['Sentiment'] = sentiment_column

    export(df, year, ticker)
