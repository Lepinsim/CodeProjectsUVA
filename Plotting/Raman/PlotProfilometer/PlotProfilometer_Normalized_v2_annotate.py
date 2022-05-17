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
hfont = {'fontname':'Helvetica'}

fontsize = 20
sns.set()
# snscolor=  sns.diverging_palette(200, 15, s=75, l=60,n=10, center="dark")
snscolor = sns.hls_palette( len(names),)
sns.set_style("white")
# sns.set(rc={ 'figure.facecolor':'cornflowerblue'})
i = 0
xspacing = 0

for name in names[i:]:
	if i == 9:
		i += 1
		continue


	data = pd.read_csv(name)
	# CenteringVector = [-ListL[i][0]-ListL[i][1]*0.5,-data.iloc[:,1].max()+ListH[i]]
	CenteringVector = [-ListL[i][0],-data.iloc[:,1].max()+ListH[i]] 
	# StackingVector = [1000*i*1.2, 0]
	StackingVector = [xspacing, 0]
	x = (data.iloc[:,0] + CenteringVector[0])/ListL[i][1] + xspacing
	y = (data.iloc[:,1] + CenteringVector[1])/ListH[i] 

	

	maxiX =data.iloc[:,0].max()
	mini = 0

	plt.fill_between(x,y, np.linspace(mini,mini,len(data.iloc[:,0])), color=snscolor[i], alpha=0.3)
	plt.plot(x,y, color=snscolor[i], label=name[:-4]+ 'mg')

	plt.annotate(name[:-4]+ 'mg',(xspacing,1.02),**hfont)
	xspacing += 1.6
	i += 1 

# plt.show()

# plt.plot(x,y, linewidth=1)

# plt.legend(loc='best', fontsize= fontsize)

# data = pd.read_csv(names[1])

# plt.plot(data.iloc[:,0]-data.iloc[:,0].mean()-1150,data.iloc[:,1]-data.iloc[:,1].max())
plt.ylim(-.05,None)
# plt.xlim(None,xspacing+5)
plt.xlabel(r'Length norm $\frac{x-L}{L}$', fontsize= fontsize)
plt.ylabel(r'Height norm $\frac{y-H}{H}$', fontsize= fontsize)

plt.show()

# plt.savefig(r'E:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\Profiles crystals at surface\AllProfileStack_15oct.png')