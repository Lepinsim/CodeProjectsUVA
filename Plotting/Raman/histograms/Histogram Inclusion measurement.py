# Histogram Inclusion measurement

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import scipy.stats


RH20 = pd.read_csv(r'D:\Experiments\Mix FeCl NaCl\Inclusions RH %Fe rpm 16092020\IncArea_20RH_25Fe.csv')
RH60 = pd.read_csv(r'D:\Experiments\Mix FeCl NaCl\Inclusions RH %Fe rpm 16092020\IncArea_60RH_25Fe.csv')

print(RH20['area'])

x = [np.log(RH20['area']),np.log(RH60['area'])]
colors = ['orange','blue']
print(x)

# # np.histogram(RH20['area'].to_numpy)
# plt.hist(RH20['area'], 50, density=True, facecolor='orange', alpha=0.75,histtype='bar')

# plt.hist(RH60['area'], 50, density=True, facecolor='blue', alpha=0.75, histtype='bar')
a, b ,c = plt.hist(x,50, density=True,histtype='bar', color=colors, alpha=0.75)

mu, sigma = scipy.stats.norm.fit(np.log(RH20['area']))
best_fit_line = scipy.stats.norm.pdf(b, mu, sigma)
print('rh20', mu, sigma)
ax1 = plt.plot(b, best_fit_line, color='orange')

mu, sigma = scipy.stats.norm.fit(np.log(RH60['area']))
best_fit_line = scipy.stats.norm.pdf(b, mu, sigma)
ax1 = plt.plot(b, best_fit_line, color='blue')
print('rh60', mu, sigma)

fontsize = 20
plt.xlabel('ln(Inclusion area)', fontsize=fontsize)
plt.ylabel('Probability', fontsize=fontsize)
plt.title('Probability density functions Inclusion size ', fontsize=fontsize)
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.xlim(0, 2500)
# plt.ylim(0, 0.03)
# plt.grid(True)
plt.xticks(range(1,10), fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # No decimal places


plt.show()
