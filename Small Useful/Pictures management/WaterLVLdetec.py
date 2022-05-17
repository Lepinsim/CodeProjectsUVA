# WaterLVLdetec

# created 17.12.2020

import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime
import scipy
import matplotlib.pyplot as plt

paths = [r'F:\Experiments\Crystal Pendant\FocusedAnalysis\E26_111_202.6\exp26_0']

# ExpNum = ''
for path in paths:
    # ExpNum = int(path[82:-9])
    ExpNum = 26
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
    print(paths)
# --- Start Loading ---
    w = 18
    # Iterating over the files found in the 'path' directory (under conditions (pic, not already cropped, and from right camera))
    for name in names[:2]:

        print(name)
    # for name in names[:4]:
        # print("d",d)
        print(str(count) + '/' + str(len(names)*4))
        count=count+1
        if name.endswith('.jpg') and 'D5200' in name:
            #  (pic, not already cropped, and from right camera)
            if not(os.getcwd().endswith(path)):
                os.chdir(path)
            print(name)
            img = cv2.imread(name,0)[:,589:589+w] 
            # print(img.dtype)   
            df['data'] = pd.DataFrame(img).mean(axis=1).rolling(10).mean()
            # df['2nd derivative'] = pd.DataFrame(img).mean(axis=1).rolling(10).mean().diff().diff()
            # df['2nd derivative'].max
            
            df['G1'] = pd.DataFrame(img).mean(axis=1).rolling(10).mean().diff()
            df['G2'] = pd.DataFrame(img).mean(axis=1).rolling(10).mean().diff().diff()
            window = 2
            # print(df['G2'])
            print(~df['G2'].between(-window,window) & ~df['G2'].isnull() & df['G2']>= 500)
            mask = ~df['G2'].between(-window,window) & ~df['G2'].isnull() & df['G1'] >= (df['G2'].max() * 0.3)
            print('g2 = ', df['G2'][mask].index)


            # print(G1[1])
# NEED TO PRINT MASK X AND G1 Y --> CONDITION DOESNT WORK AND DONT KNOW WHY
            # cv2.imshow('image',img)
            # fig = plt.plot(df)
            
            fig, ax1 =  plt.subplots()
            # ax1.plot(df['data'])
            # print(ax1)
            ax = ax1.twinx()
            ax.plot(df['G1'], color='red')

            # find the point in mask 
            ax.plot(mask)
            # print(mask.shape)
            # ax.plot(mask.idxmax[axis=0],mask.iloc(mask.idxmax[axis=0] )
            plt.show()
            # plt.close('all')
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()