from scipy.interpolate import RegularGridInterpolator
from numpy import linspace, zeros, array
import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from scipy.interpolate import griddata



moddata3 = pd.read_csv(r'D:\Experiments\Mix FeCl NaCl\Code_3Dmapping\SequenceIntValues.csv',header=None, skiprows=2)

# print(moddata3.iloc[:,1])
# moddata3.reshape(50,50,5)
array3 = moddata3.iloc[:,1].to_numpy()

array4 = array3.reshape(5,50,50)
# print(linspace(0,49))
# points = [linspace(0,49),linspace(0,49)]
points, b = np.mgrid[0:50:1, 0:50:1]
# points = (np.arange(0,50), np.arange(0,50))
# print(points)
values = array4[0][:,:]
print(np.shape(values), np.shape(points))
grid_x, grid_y = np.mgrid[0:50:1, 0:100:1]

grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')

# plt.plot

# plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0], points[:,1], 'k.', ms=1)
plt.title('Original')
plt.subplot(222)
plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')