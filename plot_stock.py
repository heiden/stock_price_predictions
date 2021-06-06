import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sentiment_analysis/results/combined/BRK-B.csv')
# data = df.filter(['Adj Close'])

# plt.figure(figsize = (12, 8))
# # plt.title('model')
# plt.xlabel('Data')
# plt.ylabel('Cotacao')
# plt.plot(df['Adj Close'])
# plt.savefig('goog.png')

# sentiments = df['Sentiment']
# moving_avg_sentiment = [sentiments[0]]
# for i in range(1, len(sentiments)):
#   moving_avg_sentiment.append(sum(sentiments[0:i]) / i)

# plt.figure(figsize = (12, 8))
# plt.title('model')
# plt.xlabel('date')
# plt.ylabel('price')
# plt.plot(moving_avg_sentiment)
# plt.savefig('sentiment.png')

jan = 1006
mar = 1046
jul = 1131
ago = 1173

plt.figure(figsize = (12, 8))
plt.title('Sentimentos das Notícias do Ativo "BRK-B" (Berkshire Hathaway) para Jan/2020 até Mar/2020')
plt.xlabel('Sentimento')
plt.ylabel('Dias')
plt.hist([x for x in df['Sentiment'][jan:mar] if x != 0], bins=19, facecolor='turquoise', alpha=0.5, edgecolor='turquoise', linewidth=1.2)
plt.savefig('sentiment_brk_1.pdf')
