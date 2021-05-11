import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('TSLA.csv')
data = df.filter(['Close'])
dataset = data.values

plt.figure(figsize = (12, 8))
plt.title('model')
plt.xlabel('date', fontsize = 14)
plt.ylabel('price', fontsize = 14)
plt.plot(data['Close'])
plt.savefig('stock.png')
