
# Crystal pendant 
# MassCrop +timestamps (mb scale)

import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime
# paths 
path = r'D:\Experiments\Crystal Pendant\TrackingAngle\exp17'
if not(os.getcwd().endswith(path)):
     os.chdir(path)
# --- Timestamps --- #
names = os.listdir() 
dates = []
datesNB = []
step = 1
# Get dates and Filenames
for name in names:
    dates.append(datetime.fromtimestamp(os.path.getmtime(name)).strftime('%Y-%m-%d %H:%M:%S'))
    datesNB.append(os.path.getmtime(name))
##Check START,END and INTERVAL here
df = pd.DataFrame({'dates':dates[0::step],'names':names[0::step]})

results = df.sort_values('dates')
results.to_csv('liststd1in' + str(step) + '.csv')

# --- Load conditions --- #

ROIs = pd.read_csv(path + '\ROIset.csv')


# print(ROIs.iloc[1,3],ROIs.iloc[1,5])
print(ROIs.columns)
col = ROIs.columns

d = []
for x in range(len(ROIs.iloc[:,1])):
		
		d.append([])


# print(d[0].append(1))
for name in names:
# for name in names[:4]:
    print("d",d)
    if name.endswith('.jpg') and not(name.endswith('Crop.jpg')) and 'D5200' in name:
        
        if not(os.getcwd().endswith(path)):
            os.chdir(path)
        
        img = cv2.imread(name,-1)

     # cv2.imshow('image',img)

    # print(results['names'])
        for i in range(len(ROIs.iloc[:,1])):
        # for i in range(1):
            a = []
            try:    
                os.chdir(path+ '\\exp'+ str(ExpNum) + str(i))
            except:
                os.chdir(path)
                os.mkdir(path+ '\\exp'+ str(ExpNum) + str(i))
            print(name)
            print (i)    
            crop_img = img[ROIs.iloc[i,4]:(ROIs.iloc[i,4]+ROIs.iloc[i,6]),ROIs.iloc[i,3]:(ROIs.iloc[i,3]+ ROIs.iloc[i,5])]
            print('std', np.std(crop_img))
            print("i= ", i, d[i])
            a.append(np.std(crop_img))
            d[i].append(np.std(crop_img))

            # cv2.imshow('image',crop_img)
            # cv2.waitKey(0)


            cv2.imwrite(name[:-4] + str(i) +'PicCrop.jpg', crop_img)

print(d)
FilmStd = pd.DataFrame(d)
# FilmStd['dates'] = results['dates']
FilmStd.T 
FilmStd.to_csv(r'D:\Experiments\Crystal Pendant\TrackingAngle\exp17\std.csv')

print(FilmStd)