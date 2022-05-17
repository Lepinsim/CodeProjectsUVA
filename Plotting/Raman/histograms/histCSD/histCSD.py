import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
df = pd.read_csv('Results.csv')
print(df)
x = np.sqrt(df['Area'])*1000/(np.pi)
# plt.hist(l, edgecolor='b')
# print (l)



fig, ax = plt.subplots()
mu = x.mean()
median = np.median(x)
sigma = x.std()
textstr = '\n'.join((
    r'$\mu=%.2f$' % (mu, ) + '$\mu$m',
    r'$\mathrm{median}=%.2f$' % (median, )+ '$\mu$m',
    # r'$\sigma=%.2f$' % (sigma, )+ '$\mu$m',
    r'N=' + str(len(x))))

ax.hist(x, edgecolor='b')
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax.set_ylabel('Count', fontsize=18)
ax.set_xlabel('Crystal size (µm)', fontsize=18)

plt.show()				