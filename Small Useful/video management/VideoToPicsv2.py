# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:00:07 2020

@author: Utilisateur
"""

import cv2
vidcap = cv2.VideoCapture(r'D:\Experiments\Crystal Pendant\Maureen_2020\videoParse\VID_20200727_135748.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(r'D:\Experiments\Crystal Pendant\Maureen_2020\videoParse\VID_20200727_135748\Image'+str(count)+'.jpg', image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 0.05 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)