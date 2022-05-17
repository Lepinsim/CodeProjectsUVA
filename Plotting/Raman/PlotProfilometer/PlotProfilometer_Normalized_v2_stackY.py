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

names = [
'1.7.csv',
# '2.4.csv',
# '3.8.csv',
'4.8.csv',
'8.0.csv',
# '9.4.csv',
'11.5.csv',
'12.1.csv',
# '24.3.csv',
# '38.6.csv',
'76.3.csv',
'186.6.csv',
]
ListL = [[150, 1166],
		[100,1300],
		[100,1431.6],
		[120,365],
		[100,1738.645],
		[150,1214], 
		[200,2122],
		[125,2035],
		[100,2800],
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
# hfont = {'fontname':'Helvetica'}

fontsize = 20
sns.set()
# snscolor=  sns.diverging_palette(200, 15, s=75, l=60,n=10, center="dark")
snscolor = sns.hls_palette( len(names),)
sns.set_style("white")
# sns.set(rc={ 'figure.facecolor':'cornflowerblue'})
# print(i)
i = 0

names = [
'1.7.csv',
# '2.4.csv',
# '3.8.csv',
'4.8.csv',
'8.0.csv',
# '9.4.csv',
'11.5.csv',
'12.1.csv',
# '24.3.csv',
# '38.6.csv',
'76.3.csv',
'186.6.csv',
]

xspacing = 0
Rnames =list(reversed(names[i:]))

print(Rnames)
for name in names:
	print(name)
	# if name=='76.3.csv':
	# 	next


	data = pd.read_csv(name)
	# CenteringVector = [-ListL[i][0]-ListL[i][1]*0.5,-data.iloc[:,1].max()]
	CenteringVector = [-ListL[i][0],-data.iloc[:,1].max()+10000+1000*i] 
	# CenteringVector = [0, 0]
	StackingVector = [1000*i*1.2, 0]
	# StackingVector = [0, 0]
	# x = (data.iloc[:,0] + CenteringVector[0])/ListL[i][1] 
	# y = (data.iloc[:,1] + CenteringVector[1])/ListH[i] - xspacing
	x = (data.iloc[:,0] + CenteringVector[0]) u
	y = (data.iloc[:,1] + CenteringVector[1])+ListH[i] 
	# print(a)
	# continue

	maxiX =data.iloc[:,0].max()
	mini = 0

	plt.fill_between(x,y, np.linspace(mini,mini,len(data.iloc[:,0])), color=snscolor[i], alpha=0.3)
	plt.plot(x,y, color=snscolor[i], label=name[:-4]+ 'mg')

	plt.annotate(name[:-4]+ 'mg',(0, CenteringVector[1]))
	xspacing += 1.6
	i -= 1 

# plt.show()

# plt.plot(x,y, linewidth=1)

# plt.legend(loc='best', fontsize= fontsize)

# data = pd.read_csv(names[1])

# plt.plot(data.iloc[:,0]-data.iloc[:,0].mean()-1150,data.iloc[:,1]-data.iloc[:,1].max())
# plt.ylim(-.05,None)
# plt.xlim(None,1)
plt.xlabel(r'Length norm $\frac{x-L}{L}$', fontsize= fontsize)
plt.ylabel(r'Height norm $\frac{y-H}{H}$', fontsize= fontsize)
print('ok')
plt.show()

# plt.savefig(r'E:\Experiments\Crystal Pendant\FocusedAnalysis\Profilometer\Profiles crystals at surface\AllProfileStack_15oct.png')