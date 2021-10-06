import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

ticker = 'GOOG'

predictions = [
  1770.35107421875, 1711.9136962890625, 1730.005126953125, 1708.2916259765625, 1717.720458984375, 1750.5382080078125, 1749.4168701171875, 1763.6116943359375, 1688.0733642578125, 1753.9013671875, 1709.62060546875
]

real = [
  1728.239990, 1740.920044, 1735.290039, 1787.250000, 1807.209961, 1766.719971, 1746.550049, 1754.400024, 1740.180054, 1736.189941, 1790.859985
  # , 1886.900024, 1891.250000, 1901.050049, 1899.400024, 1917.239990, 1830.790039, 1863.109985, 1835.739990
]

plt.figure(figsize = (12, 8))
# plt.title('model')
plt.xlabel('Dia')
plt.ylabel('Cotação (USD)')
plt.plot(real, 'firebrick', label = 'Cotação Real')
plt.plot(predictions, 'limegreen', label = 'Cotação Prevista')
# plt.xticks(set_ticks(), set_labels(), rotation = 45)
plt.legend(loc = 'lower right')
plt.savefig('predictions_{}.pdf'.format(ticker))
