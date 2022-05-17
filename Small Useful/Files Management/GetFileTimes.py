# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:13:27 2020

@author: Utilisateur
"""

import os
import pandas as pd
from datetime import datetime

path = r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\New folder'
# poslistcsv = r'D:\Experiments\Crystal Pendant\TrackingAngle\list.csv'

# poslist = pd.read_csv(poslistcsv, skiprows=0)
print(os.getcwd())
print(not(os.getcwd().endswith(path)))
if not(os.getcwd().endswith(path)):
     os.chdir(path)

names = os.listdir() 
keptnames =[]
dates = []
datesNB = []
# print (list(poslist.iloc[:,0]))
for i in names:
    # i=int(i)
    if 'D850' in i:

	    dates.append(datetime.fromtimestamp(os.path.getmtime(i)).strftime('%Y-%m-%d %H:%M:%S'))
	    datesNB.append(os.path.getmtime(i))
	    keptnames.append(i)
##Check START,END and INTERVAL here
#print("dates", dates)
#print("keptnames", keptnames)
# df = pd.DataFrame({'dates':dates[::],'names':keptnames[::],'positions':poslist.iloc[:,0]})
df = pd.DataFrame({'dates':dates[::],'names':keptnames[::]})

# results = df.sort(['dates', 'names'], ascending=[1, 0])
results = df.sort_values('dates')
# a = (datesNB[2]-datesNB[1])/60
# print (a, "min interval")
print(results)

# plt.plot(df['dates'])
results.to_csv(path + 'listmetadataD850' + '.csv')
print (path)