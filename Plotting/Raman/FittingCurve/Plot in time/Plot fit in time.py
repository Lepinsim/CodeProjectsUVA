### Plots spectra in every directories at 'filepath' string address
### Does baseline substraction, normalization, peakdetection and plotting

import scipy as scipy
from scipy import optimize
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.preprocessing import RobustScaler

from scipy.interpolate import splev
from lmfit import Parameters, minimize, fit_report

import math
import os
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.signal import find_peaks

# import rampy as rp
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import signal
# from adjustText import adjust_text
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

from scipy.ndimage import median_filter

def RunningMedian(x, N):
    return median_filter(x[x != 0], N)

 ## Initialization
sns.set()
FMT_COEF = 'coef_%2.2i'

# def RunningMedian(x,N):
#     idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
#     b = [row[row>0] for row in x[idx]]
#     return np.array(map(np.median,b))
#     #return np.array([np.median(c) for c in b])  # This also works

# Working folder
# path = r'D:\Experiments\Mix FeSO4\FeSO4_NaCl early2022\Bulk\Raman Analysis\Data'
# path = r'C:\Users\Simon\surfdrive\Experiment\Mix FeSO4\FeSO4_NaCl early2022\Bulk\Raman Analysis\Data'
path = r'C:\Users\Simon\OneDrive - UvA\Documents\PHD\D8.Side projects\PYTHON1\Plotting\FittingCurve\Plot in time\Data'


# os.chdir(path)
#####################################################################################################################""
#####################################################################################################################""
#####################################################################################################################""

    # Start looping over folders containing spectra

directories = []
print(os.walk(path))
for subdir, dirs, files in os.walk(path):
    # directories = []
    # print(subdir,'end')
# 
    names = []
    for file in files:
        # print(os.path.join(subdir, file))
        filepath = os.path.join(subdir, file)
        if filepath.endswith('.csv'):
          names.append(filepath)
    directories.append(names)
    
# break
# print('stqrt',directories)
        

i=1
nSpectra = len(names)


  ## Initialize Figure parameters

#Curves colors
# snscolor=  sns.diverging_palette(200, 15, s=75, l=60,n=nSpectra+1, center="dark")
# snscolor=  sns.diverging_palette(200, 15, s=75, l=60,n=46, center="dark")
# snscolor=  sns.color_palette("tab10")
snscolor = ['#0000ffff','#7d7dffff','#ff7d7dff','#ff0000ff','#40ff40',
'#0000ffff','#7d7dffff','#ff7d7dff','#ff0000ff','#40ff40',
'#0000ffff','#7d7dffff','#ff7d7dff','#ff0000ff','#40ff40',]

sns.set_style("white")
sns.set_style("ticks")



# ax1.set_xbound(25,80)
i=1
# inialization empty containers
dfPeaks = pd.DataFrame()
dfPeaksPrint = pd.DataFrame()
emptyCol = np.linspace(0,14,15)


#####################################################################################################################""
#####################################################################################################################""
#####################################################################################################################""
  # Starting loop loading and plotting each in each directory 
  # for j in range(1):
for names in directories:
  fig, ax1 = plt.subplots()
  print(len(names))
  for j in range (len(names)):
        # Initializing variables
        name = names[j]
        # title = dfc.columns[i]
        # print(i, title)

        dfa = pd.DataFrame()
        # Rescaling
         # Interval between lines
        deltaY = 10*(-0.3-1/nSpectra)*i
         # scaling factor
        scale = 1

        ##Loading data/
        # if name.endswith('.csv'):
          # Load data in file named 'name' optional rolling average 
        dfb = pd.read_csv(name,header=None, skiprows=19, encoding='cp1252')

        
         

         ### Preprocessing
        # print(dfb.iloc[:,1])
        dfa = dfb.iloc[:,1]-deltaY
        dfascreened = RunningMedian(dfa.to_numpy(), 50)
        print(dfascreened)


        p, = ax1.plot(dfb.iloc[:,0],dfascreened,linewidth=1,
        # p, = ax1.plot(dfb.iloc[:,0],dfb.iloc[:,1]-deltaY,linewidth=1,
            color=snscolor[i],)
            # label=name[82:-4])
            # label=title)
            # label=title)
         


        i +=1
  plt.savefig(path+r'\Figure'+str(i))
  print(nSpectra)
  # plt.tick_params(axis ='both', which ='both', length = 3, color='k')
  fontsize = 20
  plt.xticks(fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
  # plt.xticks()
  # plt.yticks()
  # plt.xlim(None  ,1500)
  # plt.ylim(None,6)


  handles, labels = ax1.get_legend_handles_labels()
  # ax1.legend(reversed(handles), reversed(labels), loc='upper right', fontsize=7)


  # plt.xlabel('Raman Shift (rel. cm$^{-1}$)', fontsize=16, fontweight='bold')
  # plt.ylabel('Intensity (a. u.)', fontsize=16, fontweight='bold')
plt.show()




