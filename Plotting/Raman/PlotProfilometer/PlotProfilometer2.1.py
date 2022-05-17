import pandas as pd
import numpy as np
# from scipy import integrate
import matplotlib
import os
import seaborn as sns

import matplotlib.pyplot as plt

# names = [r'E:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\76.3_highres.csv',
# 		r'E:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\186.6.csv'
# ]

path = r'F:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\Profiles crystals at surface\surface crystal profiles - Copy'

x = np.linspace(0,0,500)
y = np.linspace(-10000,0,500)

os.chdir(path)
def withoutExtension(string):
	return int(string[:-6])

ListL = [[150, 1166],
		[0,1300],
		[0,1431.6],
		[120,1365],
		[100,1738.645],
		[150,2214], 
		[100,2122],
		[125,2035],
		[0,2800],
		[-500, 3357],
		[150,4470],
		[100,4460],]


ListH = [623.4205,
628.0775,
952.507,
610.517,
1126.623,
1217.165,
1388,
1317.645,
1508,
3000,
2378.62,
3131.5,]
  # Initialize file names
names = os.listdir()
names.sort(key=withoutExtension)
print(names)

fontsize = 20
sns.set()
# snscolor=  sns.diverging_palette(200, 15, s=75, l=60,n=10, center="dark")
snscolor = sns.hls_palette( len(names),)
sns.set_style("white")
# sns.set(rc={ 'figure.facecolor':'cornflowerblue'})
i = 0
xspacing = 0

for name in names[:]:



	data = pd.read_csv(name)
	# CenteringVector = [-ListL[i][0]-ListL[i][1]*0.5,-data.iloc[:,1].max()+ListH[i]]
	CenteringVector = [0,-data.iloc[:,1].max()+ListH[i]]
	# StackingVector = [1000*i*1.2, 0]
	StackingVector = [xspacing, 0]


	# plt.plot(data.iloc[:,0]+CenteringVector[0],data.iloc[:,1] + CenteringVector[1] + StackingVector[1], color=snscolor[i], label=name[:-4]+ 'mg')
	plt.plot(data.iloc[:,0]+CenteringVector[0]+StackingVector[0],data.iloc[:,1] + CenteringVector[1] + StackingVector[1], color=snscolor[i], label=name[:-4]+ 'mg')

	# plt.plot(data.iloc[:,0]-ListL[i][0]-ListL[i][1]*0.5,data.iloc[:,1].diff()-100*i, color=snscolor[i], label=name[:-4]+ 'mg')
	
	# sideleftY = data.iloc[:,1][data.iloc[:,1].diff()  >= 0.5*data.iloc[:,1].diff().max()]
	# sideleftX = data.iloc[:,0][data.iloc[:,1].diff()  >= 0.5*data.iloc[:,1].diff().max()]


	# coordinates = data[data.iloc[:,1].diff()  >= 0.05*data.iloc[:,1].diff().max()]
	# coordinates = (sideleftX +CenteringVector[0], sideleftY +CenteringVector[1] + StackingVector[1])
	# print(coordinates)
	# plt.plot(coordinates.iloc[:,0]+CenteringVector[0],coordinates.iloc[:,1] + CenteringVector[1] + StackingVector[1], 'ko')
	# mini = data.iloc[:,1].min()+ CenteringVector[1] + StackingVector[1]
	# mini =-2700
	maxiX =data.iloc[:,0].max()
	# mini =CenteringVector[1] + StackingVector[1]
	mini = 0
	# plt.plot(data.iloc[:,0]-ListL[i][0]-ListL[i][1]*0.5,np.linspace(mini,mini,len(data.iloc[:,0])), color=snscolor[i], )

	plt.fill_between(data.iloc[:,0]+CenteringVector[0]+StackingVector[0],data.iloc[:,1] + CenteringVector[1] + StackingVector[1], np.linspace(mini,mini,len(data.iloc[:,0])), color=snscolor[i], alpha=0.3)
	# ax2.fill_between(x, y1, y2, where=(y1 > y2), color='C0', alpha=0.3,
 #                 interpolate=True)
	xspacing += (ListL[i][0]+ListL[i][1])*1.20
	i += 1 

# plt.show()

# plt.plot(x,y, linewidth=1)

plt.legend(loc='best', fontsize= fontsize)

# data = pd.read_csv(names[1])

# plt.plot(data.iloc[:,0]-data.iloc[:,0].mean()-1150,data.iloc[:,1]-data.iloc[:,1].max())

plt.xlabel(r'Width ($\mu m$)', fontsize= fontsize)
plt.ylabel(r'Height ($\mu m$)', fontsize= fontsize)

plt.show()

# plt.savefig(r'E:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\Profiles crystals at surface\AllProfileStack_15oct.png')