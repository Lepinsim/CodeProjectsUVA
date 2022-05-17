import matplotlib.pyplot as plt
import pandas as pd


def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

data = pd.read_csv('Mass_surfaceTension_corrv4.csv')

fig, axes = plt.subplots(nrows=1,ncols=1)

# print(data['m/(L*sin(theta))'])
# data['m/(L*sin(theta))']= [data['m/(L*sin(theta))'].map(isnumber)]

# fig.tight_layout(pad=3.0)
# df = data['m/(L*sin(theta))'][data['m/(L*sin(theta))'].map(isnumber)]
# df = pd.to_numeric(df, errors='coerce')
# # print(pd.to_numeric(df, errors='coerce')

binsize = 10
# axes.hist(data[' m/L '],color='orange', bins=binsize)
# # Set a title and x-and y-axis labels
# axes.set(
# 			# title='$\theta$ at t$_{all}$ deg)', 
#  			ylabel='Count',
#  			xlabel='$g$*m/L')
# # plt.hist(data['Mass(mg)'])
# # plt.show()
axes.hist(data['Mass(mg)'],color='b', bins=binsize)
# Set a title and x-and y-axis labels
axes.set(
			# title='Mass m (mg)', 
 			ylabel='Count',
 			xlabel='m (mg)'
 			)

plt.savefig('Fig0_m' + str(binsize) + '.jpg')