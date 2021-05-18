import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def avg(list):
  if len(list) == 0:
    return 0
  else:
    return sum(list) / len(list)

tickers = ['GOOG']

for ticker in tickers:
  df = pd.read_csv('news/{}.csv'.format(ticker), delimiter=';')
  headlines = df['Headline'].values
  dates = [x for x in zip(df['Day'], df['Month'])]

  # using 1-31 for day and 1-12 for month to be clearer
  sentiments = [[[] for x in range(13)] for y in range(32)]
  analyzer = SentimentIntensityAnalyzer()

  for pair in zip(headlines, dates):
    headline, date = pair
    vader_sentiment_scores = analyzer.polarity_scores(headline)
    sentiments[date[0]][date[1]].append(vader_sentiment_scores['compound'])

  df = pd.read_csv('stocks/{}_2020.csv'.format(ticker))
  trading_dates = df['Date'].values

  sentiment_column = []
  for date in trading_dates:
    year, month, day = list(map(int, date.split('-')))
    sentiment_column.append(avg(sentiments[day][month]))

  df['Sentiment'] = sentiment_column

  with open('results/{}.csv'.format(ticker), 'w') as file:
    file.write(df.to_string())
