import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

ticker = 'AMZN'

predictions = [
  3156.51305, 3199.00109, 3108.60450, 3138.07063, 3113.87013, 3131.33527, 3171.55551, 3159.18126, 3133.68859, 3058.62879,
  3102.62312, 3183.04400, 3199.42031, 3240.62725, 3256.88927, 3232.63568, 3214.23040, 3198.62240, 3194.95988, 3256.23924,
  3276.02944, 3288.31648, 3279.46890, 3261.30119, 3244.20458, 3233.30671, 3228.62974, 3223.22324, 3206.26878, 3216.14455,
  3222.45597, 3224.03146, 3218.89682, 3214.22545, 3201.28340, 3172.37288, 3162.78093, 3155.74081, 3142.40917, 3137.85691,
  3102.07420, 3050.98327, 3032.46078, 2988.35670, 2975.51925, 2970.80949, 2998.11508, 3007.35807, 3020.49526, 3045.51605,
  3030.43650, 3020.29107, 3011.12391, 3003.97966, 2987.89966, 2973.91372, 2987.04037, 2989.28535, 3001.64275, 3013.09821,
  3030.63209, 3037.22132, 3048.84408, 3042.48157, 3055.11940, 3062.74809, 3048.36349, 3080.96635, 3111.55994, 3136.15080,
  3157.74630, 3149.35216, 3133.97465, 3121.61768, 3146.28307, 3166.97172, 3189.68145, 3243.40980, 3250.15381, 3233.91044,
  3221.67561, 3211.44730, 3198.22340, 3192.00292, 3176.78490, 3155.56970, 3130.35823, 3105.15139, 3088.94960, 3052.75416
]

real = [
  3186.62988, 3218.51001, 3138.37988, 3162.15991, 3182.69995, 3114.20996, 3120.83007, 3165.88989, 3127.46997, 3104.25000,
  3120.76001, 3263.37988, 3306.98999, 3292.22998, 3294.00000, 3326.12988, 3232.58007, 3237.62011, 3206.19995, 3342.87988,
  3380.00000, 3312.53002, 3331.00000, 3352.14990, 3322.93994, 3305.00000, 3286.58007, 3262.12988, 3277.70996, 3268.94995,
  3308.63989, 3328.22998, 3249.89990, 3180.73999, 3194.50000, 3159.53002, 3057.15991, 3092.92993, 3146.13989, 3094.53002,
  3005.00000, 2977.57006, 3000.45996, 2951.94995, 3062.85009, 3057.63989, 3113.59008, 3089.48999, 3081.67993, 3091.86010,
  3135.72998, 3027.98999, 3074.95996, 3110.87011, 3137.50000, 3087.07006, 3046.26001, 3052.03002, 3075.72998, 3055.29003,
  3094.08007, 3161.00000, 3226.72998, 3223.82006, 3279.38989, 3299.30004, 3372.19995, 3379.38989, 3400.00000, 3333.00000,
  3379.09008, 3399.43994, 3372.01001, 3334.68994, 3362.02002, 3309.04003, 3340.87988, 3409.00000, 3417.42993, 3458.50000,
  3471.31005, 3467.41992, 3386.48999, 3311.87011, 3270.54003, 3306.37011, 3291.61010, 3190.48999, 3223.90991, 3151.93994
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
