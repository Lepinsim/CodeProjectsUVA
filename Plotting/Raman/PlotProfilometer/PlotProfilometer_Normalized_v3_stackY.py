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

path = r'F:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\Profiles crystals at surface\Profiles list only pres'

x = np.linspace(0,0,500)
y = np.linspace(-10000,0,500)

os.chdir(path)
def withoutExtension(string):
	return int(string[:-6])

names = [
'1.7.csv',
# '2.4.csv',
# '3.8.csv',
# '4.8.csv',
# '8.0.csv',
# '9.4.csv',
# '11.5.csv',
'12.1.csv',
# '24.3.csv',
# '38.6.csv',
# '76.3.csv',
'186.6.csv',
]
ListL = [[150, 1066],
		# [100,1300],
		# [100,1431.6],
		# [120,1265],
		# [100,1738.645],
		# [150,1214], 
		# [200,2122],
		[125,2835],
		# [100,2800],
		# [-500, 3357],
		# [-6050,4470],
		[100,4460],]


ListH = [-623.4205,
# 628.0775,
# 952.507,
# 610.517,
# 1126.623,
# 1217.165,
# 1388,
1317.645,
# 1508,
# 3000,
# 2378.62,
3131.5,]


#   # Initialize file names
# names = os.listdir()
# names.sort(key=withoutExtension)
# print(names)
hfont = {'fontname':'Helvetica'}

fontsize = 25
sns.set()
# snscolor=  sns.diverging_palette(200, 15, s=75, l=60,n=10, center="dark")
snscolor = sns.hls_palette( len(names))
sns.set_style("white")
# sns.set(rc={ 'figure.facecolor':'cornflowerblue'})
# print(i)
i = 0
mini= [200,-1100,-300]


xspacing = 0
Rnames =list(reversed(names[i:]))
i = 0
print(Rnames)
listHannotation =[-0.45, +0.8,+0.5 ]
for name in names:
### ListL = [x(1) where data starts; crystal width  ]
### ListH = [crystal height] 
	data = pd.read_csv(name)

	# print(name, ListL[i][0], ListH[i] , "\n")
	verticalFactor = -data.iloc[:,1].max()-(i**1.3)*1000+1000

	x = data.iloc[:,0]-ListL[i][0]-ListL[i][1]*0.5

	# y = data.iloc[:,1]-data.iloc[:,1].max()-(len(names)-i)*1000 # reverse order vertical
	y = data.iloc[:,1]+verticalFactor

	maxiX =data.iloc[:,0].max()
	# mini = y.min()
	print(name)
	# mini = verticalFactor+ListH[i]*0.5
	print(mini[i],np.linspace(mini[i]/1000,mini[i]/1000,len(data.iloc[:,0])))

	# print(ListH[i])
	# print(x.shape, y.shape, )

	plt.fill_between(x/1000,y/1000, np.linspace(mini[i]/1000,mini[i]/1000,len(data.iloc[:,0])), color=snscolor[i], alpha=0.3)
	plt.plot(x/1000,y/1000, color=snscolor[i], label=name[:-4]+ 'mg')

	plt.annotate(name[:-4]+ 'mg',(0, verticalFactor/1000+listHannotation[i]), fontsize= 12,ha='center', **hfont)
	xspacing += 1.6
	i += 1 

# plt.show()

# plt.plot(x,y, linewidth=1)

# plt.legend(loc='best', fontsize= fontsize)

# data = pd.read_csv(names[1])

# plt.plot(data.iloc[:,0]-data.iloc[:,0].mean()-1150,data.iloc[:,1]-data.iloc[:,1].max())
# plt.ylim(-35000,None)
# plt.xlim(-3000,3000)
plt.xlabel(r'Crystal width (mm)', fontsize= fontsize, **hfont)
plt.ylabel(r'Crystal height (mm)', fontsize= fontsize, **hfont)
print('ok')
plt.xticks(fontsize= fontsize -7)
plt.yticks(fontsize= fontsize -7)
plt.axis('equal')
plt.show()


# plt.savefig(r'E:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\Profiles crystals at surface\AllProfileStack_15oct.png')