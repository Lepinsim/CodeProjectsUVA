# Crystal pendant 
# MassCrop +timestamps (mb scale)

import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\Experiments\Crystal Pendant\TrackingAngle\exp17_354\std.csv')
dates = pd.read_csv(r'D:\Experiments\Crystal Pendant\TrackingAngle\exp17_354\listmetadata1in1.csv')

dft = df.T
dft['dates'] = list(dates['dates'])[:-11]
# dft['dates'] = pd.to_datetime(dates['dates'][:-11])
i = 0
dft['time'] = pd.to_datetime(dft['dates']) - pd.to_datetime(dft['dates'][i])
print(dft['dates'])
# dft['time'] = 
print(dft['time'],)
print(dft[i].diff())


# dft['']
# print(dft)
# print(dates['dates'])
# print(dft['timestamps'])
# ---Plotting
# plt.locator_params(nbins=10)

fig = plt.figure(figsize=(12, 10), dpi=80)
ax1 = fig.add_subplot(111)
lines = ax1.plot(dft['time'][1:-1], dft[i][1:-1], label='Standard Deviation')
ax2 = ax1.twinx()
lines = ax2.plot(dft['time'][1:-1], dft[i][1:-1].rolling(10).mean().diff(), label='Std 2nd derivative', color='red')

ax1.set(title='Pixel variation (std. dev.) VS Time', xlabel='Time (days)', ylabel='Standard Deviation')
fig.legend(['Standard Deviation','Std 2nd derivative'],loc='best')
# lns = ax1 +ax2 
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)
# ax1.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
# ax1.xaxis.set_major_locator()

plt.show()
