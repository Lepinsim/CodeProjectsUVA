# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:13:27 2020
Create a csv file containing the names and creation times of the %path% directory files
Possibility to adjust the step

@author: Utilisateur
"""

import os
import pandas as pd
from datetime import datetime

### YOU NEED TO verify THE DIRECTORY
path = r'E:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment51_15062021'
#poslistcsv = r'D:\Experiments\Mix FeCl NaCl\Capillary exp\pos meas.csv'

#poslist = pd.read_csv(poslistcsv)
print(not(os.getcwd().endswith(path)))
if not(os.getcwd().endswith(path)):
     os.chdir(path)
print(os.getcwd())

names = os.listdir() 
dates = []
datesNB = []
step = 4

for name in names:
    dates.append(datetime.fromtimestamp(os.path.getmtime(name)).strftime('%Y-%m-%d %H:%M:%S'))
    datesNB.append(os.path.getmtime(name))
##Check START,END and INTERVAL here
df = pd.DataFrame({'dates':dates[85:506:step],'names':names[85:506:step]})
# results = df.sort(['dates', 'names'], ascending=[1, 0])
results = df.sort_values('dates')
# a = (datesNB[2]-datesNB[1])/60
# print (a, "min interval")
print (results)

# plt.plot(df['dates'])
os.chdir(r'E:\Experiments\Crystal Pendant\FocusedAnalysis\E51')
results.to_csv('listmetadata1in' + str(step) + '.csv')


