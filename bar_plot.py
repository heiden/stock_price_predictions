from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

intervals = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']
price = [35.388, 63.738, 68.123, 61.294, 76.354, 87.489, 224.944, 202.243, 164.474]

# pg 1.95, 2.67, 2.88, 2.60, 3.23, 3.61, 9.29, 10.35, 9.79
# jnj 1.73, 2.83, 3.14, 3.82, 3.31, 6.04, 9.36, 9.32, 10.58
# wmt 1.54, 2.72, 2.93, 3.62, 3.26, 4.74, 5.62, 8.51, 7.53
# v 2.22, 4.20, 4.66, 3.91, 4.55, 6.63, 16.47, 15.01, 16.58
# brkb 2.43, 4.38, 4.68, 4.21, 5.25, 6.02, 15.48, 13.92, 14.32
# fb 2.87, 5.17, 5.32, 4.95, 6.15, 15.76, 18.90, 19.40, 21.34
# goog 17.47, 31.11, 35.93, 32.91, 43.74, 58.58, 122.78, 115.26, 109.60
# msft 2.46, 5.21, 6.50, 5.05, 6.76, 8.78, 19.87, 18.37, 26.87
# aapl 1.20, 2.23, 2.51, 2.54, 3.13, 3.78, 10.95, 11.55, 14.59


fig, ax = plt.subplots(figsize = (12, 8))

plt.xlabel('Intervalos (dias)')
plt.ylabel('RMSE')
ax.bar(intervals, price, color = 'turquoise', edgecolor = 'blue')
# ax.barh(intervals, price)

plt.savefig('intervals.pdf')
