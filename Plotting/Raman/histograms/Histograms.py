import matplotlib.pyplot as plt
import pandas as pd

import os
# os.chdir('/my/directory')

print(os.getcwd())
data = pd.read_csv('Mass_surfaceTension_corrv4.csv')

fig, axes = plt.subplots(nrows=2,ncols=2)

fig.tight_layout(pad=3.0)

binsize = 10
axes[0][0].hist(data['Mass(mg)'],color='b', bins=binsize)
# Set a title and x-and y-axis labels
axes[0][0].set(
			# title='Mass m (mg)', 
 			ylabel='Count',
 			xlabel='m (mg)'
 			)

axes[0][1].hist(data['Contact perimeter']/1000,color='r', bins=binsize)
# Set a title and x-and y-axis labels
axes[0][1].set(
			# title='Contact perimeter L ($mu$m)', 
 			ylabel='Count',
 			xlabel='Contact perimeter L (mm)')

axes[1][0].hist(data['angle at tfall'],color='g', bins=binsize)
# Set a title and x-and y-axis labels
axes[1][0].set(
			# title='$theta$ at t$_fall$ deg)', 
 			ylabel='Count',
 			xlabel='$\\theta$ at t$_{fall}$ (deg)')

axes[1][1].hist(data[' m/L '],color='orange', bins=binsize)
# Set a title and x-and y-axis labels
axes[1][1].set(
			# title='$\theta$ at t$_{all}$ deg)', 
 			ylabel='Count',
 			xlabel='$g$m/L')

# plt.hist(data['Mass(mg)'])
# plt.show()


plt.savefig('Fig1_CrystalDistributions_bin' + str(binsize) + '.png')