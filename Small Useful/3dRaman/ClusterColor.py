import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter


moddata3 = pd.read_csv(r'D:\Experiments\Mix FeCl NaCl\Code_3Dmapping\SequenceIntValues2.csv',header=None, skiprows=2)

# print(moddata3.iloc[:,1])
# moddata3.reshape(50,50,5)
array3 = moddata3.iloc[:,1].to_numpy()

array4 = array3.reshape(5,50,50)

print(array4)



from matplotlib import cm

# create a vertex mesh
xx, yy = np.meshgrid(np.linspace(0,100,50), np.linspace(0,100,50))

# create vertices for a rotated mesh (3D rotation matrix)
X =  xx 
Y =  yy
Z =  10*np.ones(X.shape)

# create some dummy data (20 x 20) for the image
data = array4[0]

# create the figure
fig = plt.figure()

# show the reference image
# ax1 = fig.add_subplot(121)
# ax1.imshow(data, cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0,1,0,1])

# generate the colors for your colormap
color1 = colorConverter.to_rgba('blue')
color2 = colorConverter.to_rgba('red')

# make the colormaps
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['green','blue'],256)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)

cmap2._init() # create the _lut array, with rgba values

# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
alphas = np.linspace(0, 0.8, cmap2.N+3)
cmap2._lut[:,-1] = alphas

# show the 3D rotated projection
ax2 = fig.add_subplot(111, projection='3d')

for i in range (array4.shape[0]):
	cset = ax2.contourf(X, Y, array4[i], 10, zdir='z', offset=i*10, cmap=cmap2)

ax2.set_xlabel('x (µm)')
ax2.set_ylabel('y (µm)')
ax2.set_zlabel('z (µm)')
# ax2.yaxis._axinfo['label']['space_factor'] = 3.0

ax2.set_zlim((0.,100.))

cbar = plt.colorbar(cset)
cbar.set_label('Presence of FeCl$_3$', rotation=270, labelpad=+20)
plt.show()
