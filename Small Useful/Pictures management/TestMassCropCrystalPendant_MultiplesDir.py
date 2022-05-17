
# Crystal pendant 
# MassCrop +timestamps (mb scale)

import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime
# paths = [r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment11_14082020', 
#            r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment10_10082020',
#            r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment6_24072020',
           # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment4_17072020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment7_27072020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment8_31072020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment9_04082020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment26_0612020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment28_10122020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment28_10122020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment25_0312020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment16_29092020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment30_11012121',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment30_11012121\PendantCrystal_Experiment30_11012121-2',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment31_18012121',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment32_22012121',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment29_13122020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment23_27112020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment22_16112020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment21_26102020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment20_16102020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment19_12102020',
             # r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment18_07102020']
paths = [r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment33_25012121']

# ExpNum = ''
for path in paths:

    ExpNum = int(path[82:-9])
    # ExpNum = 301
    count=1
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
    # print(ROIs.columns)
    col = ROIs.columns
# Build ROI DataFrame
    d = []
    for x in range(len(ROIs.iloc[:,1])):
            
            d.append([])


# --- Start Cropping ---

    # Iterating over the files found in the 'path' directory (under conditions (pic, not already cropped, and from right camera))
    for name in names:
    # for name in names[:4]:
        # print("d",d)
        print(str(count) + '/' + str(len(names)*len(ROIs.iloc[:,1])))
        count=count+1
        if name.endswith('.jpg') and not(name.endswith('Crop.jpg')) and 'D5200' in name:
            #  (pic, not already cropped, and from right camera)
            if not(os.getcwd().endswith(path)):
                os.chdir(path)
            
            img = cv2.imread(name,-1)

         # cv2.imshow('image',img)

            #Iterating over the ROIs 
            for i in range(len(ROIs.iloc[:,1])):
            # for i in range(1):
                # print(str(count) + '/' + str(len(names)*4))
                count=count+1    
                a = []
                try:    
                    os.chdir(r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\ExtractedROI\exp'+ str(ExpNum) + '_'+ str(i))
                except:
                    # os.chdir(path)
                    os.mkdir(r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\ExtractedROI\exp'+ str(ExpNum) + '_'+ str(i))
                # print(name)
                # print (i)    
                # --- Actual Cropping ---
                crop_img = img[ROIs.iloc[i,4]:(ROIs.iloc[i,4]+ROIs.iloc[i,6]),ROIs.iloc[i,3]:(ROIs.iloc[i,3]+ ROIs.iloc[i,5])]
                # print('std', np.std(crop_img))
                # print("i= ", i, d[i])
                a.append(np.std(crop_img))
                d[i].append(np.std(crop_img))

                cv2.imwrite(name[:-4] + str(i) +'PicCrop.jpg', crop_img)

    print(d)
    FilmStd = pd.DataFrame(d)
    FilmStd['dates'] = results['dates']
    # FilmStd.T 
    FilmStd.to_csv(r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\ExtractedROI\Exp' + str(ExpNum) + '_std.csv')

    print(FilmStd)