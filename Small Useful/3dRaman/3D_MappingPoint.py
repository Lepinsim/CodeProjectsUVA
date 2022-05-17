# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:19:06 2019

@author: Utilisateur
"""

import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt

data = pd.read_csv(r'D:\Experiments\Mix FeCl NaCl\Code_3Dmapping\Point_001.csv',header=None, skiprows=2)
# data2 = pd.read_csv(r'D:\Experiments\Mix FeCl NaCl\Code_3Dmapping\Line_001.csv',header=None, skiprows=2)
data3 = pd.read_csv(r'D:\Experiments\Mix FeCl NaCl\Code_3Dmapping\\3Dstack2.csv',header=0, skiprows=0)
# data = pd.read_csv(r'D:\Experiments\Mix FeCl NaCl\Code_3Dmapping\Point_001.csv',header=None)

# print(data.iloc[:,0][120:180])
# result = data.iloc[:, 1].apply(lambda x: integrate.quad(x,data.iloc[:,0]))
# result = data.iloc[:, 1].apply(lambda x: integrate.quad(x,100,200))
# result = integrate.quad(data.iloc[:, 1],100,200)
# data.set_index(data.iloc[:,0])
# 
# 120 =230cm-1 180=4.01E+02cm-1
# 150 =316cm

plt.fill(data.iloc[120:180,0],data.iloc[120:180,1].rolling(window=10).mean())
plt.plot(data.iloc[:,0],data.iloc[:,1].rolling(window=10).mean())
plt.show()

def integ(x):
    y = x[120:180].cumsum().max()
    return y 

moddata3 = data3.iloc[2:,:].apply(integ)



moddata3.to_csv(r'D:\Experiments\Mix FeCl NaCl\Code_3Dmapping\SequenceIntValues2.csv')



    