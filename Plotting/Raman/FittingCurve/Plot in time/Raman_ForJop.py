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



def baseline_als(y, lam=100000, p=0.1, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def pad(a, reference_shape):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """

    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    # insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    # result[insertHere] = array
    result[:a.shape[0],:a.shape[1]] = a

    return result

def myFunc(e):
  return e[:2]

def index_of(arrval, value):
    """return index of array *at or below* value """
    if value < min(arrval):
        return 0
    return max(np.where(arrval <= value)[0])

def spline_eval(params, xdata, xknots, degree=3):
    coefs = [params[FMT_COEF % i].value for i in range(ncoefs-degree-1)]
    return splev(xdata, [xknots, coefs, degree])

def residual(params, xdata, ydata, xknots, degree=3):
    return ydata - spline_eval(params, xdata, xknots, degree=degree)


def _1gaussian(x, amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2)))


def _1Lorentzian(x, amp, cen, wid,y0):
    return amp*wid**2/((x-cen)**2+wid**2)+y0

#####################################################################################################################""
#####################################################################################################################""
#####################################################################################################################""
#####################################################################################################################""

 ## Initialization
sns.set()
FMT_COEF = 'coef_%2.2i'


# Working folder
# path = r'D:\Experiments\Mix FeSO4\FeSO4_NaCl early2022\Bulk\Raman Analysis\Data'
# path = r'C:\Users\Simon\surfdrive\Experiment\Mix FeSO4\FeSO4_NaCl early2022\Bulk\Raman Analysis\Data'
path = r'C:\Users\Simon\surfdrive\Experiment\Mix FeSO4\FeSO4_NaCl early2022\Mar22\Data_cleaned'

dirs =[r'C:\Users\Simon\surfdrive\Experiment\Mix FeSO4\FeSO4_NaCl early2022\Mar22\data\10a',
r'C:\Users\Simon\surfdrive\Experiment\Mix FeSO4\FeSO4_NaCl early2022\Mar22\data\10b',
r'C:\Users\Simon\surfdrive\Experiment\Mix FeSO4\FeSO4_NaCl early2022\Mar22\data\25a',
r'C:\Users\Simon\surfdrive\Experiment\Mix FeSO4\FeSO4_NaCl early2022\Mar22\data\25b',]
# os.chdir(path)
#####################################################################################################################""
#####################################################################################################################""
#####################################################################################################################""

    # Start looping over folders containing spectra

directories = []
print(os.walk(path))
for subdir, dirs, files in os.walk(path):
    # directories = []
    print(subdir,'end')

    names = []
    for file in files:
        # print(os.path.join(subdir, file))
        filepath = os.path.join(subdir, file)
        if filepath.endswith('.csv'):
          names.append(filepath)
    directories.append(names)
    
# break
print('stqrt',directories)
        

i=1
nSpectra = len(names)


  ## Initialize Figure parameters

#Curves colors
snscolor=  sns.diverging_palette(200, 15, s=75, l=60,n=nSpectra+1, center="dark")
snscolor=  sns.diverging_palette(200, 15, s=75, l=60,n=46, center="dark")
# snscolor=  sns.color_palette("tab10")
# snscolor = ['#0000ffff','#7d7dffff','#ff7d7dff','#ff0000ff']

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

        print(name)
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
         # Baseline removal
        dfa.loc[:,0] = dfb.loc[:,0]
        dfa.loc[:,1] = dfb.iloc[:,1]-baseline_als(dfb.iloc[:,1],100000,0.001)

    
         # Scale intensities to [0,1]
        dfanorm = scale*StandardScaler().fit_transform(dfa.values)[:,1]
         # Smoothing factor 
        x = moving_average(dfanorm, 5)


        p, = ax1.plot(dfa.iloc[:,0],dfanorm-deltaY,linewidth=1,
            color=snscolor[i],
            label=name[82:-4])
            # label=title)
            # label=title)
         
  #
        #detect peaks
        peaks2, p2H_dict = find_peaks(dfanorm, prominence=0.001, height=0.01, width=3) 
        #join peaks with intensities
        p2H_list = list(p2H_dict.items())
        p2H_array = np.array(p2H_list[0][1])
         # print(peaks2,p2H_array )
        peaks2b = np.array(list(zip(peaks2,p2H_array)))
         # print(peaks2b)

        try:
          peaks3 = peaks2b[peaks2b[:, 1].argsort()]
        except IndexError:
          print('no peaks founds')
          continue
      ############################# PLOT DESIGN #################

         # print(peaks3)
        for peak in peaks3[:,0]:
          # ax1.annotate(int(dfa.iloc[int(peak),0]),(dfa.iloc[int(peak),0], dfanorm[int(peak)]-deltaY), color=snscolor[i])
          ax1.annotate(int(dfa.iloc[int(peak),0]),(dfa.iloc[int(peak),0], dfanorm[int(peak)]-deltaY), color=snscolor[i])
        # plt.savefig(r'C:\Users\Simon\surfdrive\Experiment\Mix FeSO4\FeSO4_NaCl early2022\Mar22')

        i +=1

  print(nSpectra)
  # plt.tick_params(axis ='both', which ='both', length = 3, color='k')
  fontsize = 20
  plt.xticks(fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
  # plt.xticks()
  # plt.yticks()
  plt.xlim(None  ,1500)
  # plt.ylim(None,6)


  handles, labels = ax1.get_legend_handles_labels()
  ax1.legend(reversed(handles), reversed(labels), loc='upper right', fontsize=7)


  plt.xlabel('Raman Shift (rel. cm$^{-1}$)', fontsize=16, fontweight='bold')
  plt.ylabel('Intensity (a. u.)', fontsize=16, fontweight='bold')
plt.show()




