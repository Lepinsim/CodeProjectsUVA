# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:58:19 2020

@author: Utilisateur
"""

import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        print(pathOut)
        cv2.imwrite( "img_1.png", image)     # save frame as JPEG file
        count = count + 1

# if __name__=="__main__":
#     a = argparse.ArgumentParser()
#     a.add_argument(r'D:\Experiments\Crystal Pendant\Maureen_2020\videoParse\VID_20200727_135748.mp4', help="path to video")
#     a.add_argument(r'D:\Experiments\Crystal Pendant\Maureen_2020\videoParse\VID_20200727_135748\Image', help="path to images")
#     args = a.parse_args()
#     print(args)
#     extractImages(args.pathIn, args.pathOut)

extractImages(r'D:\Experiments\Crystal Pendant\Maureen_2020\videoParse\VID_20200727_135748.mp4',
              r'D:\Experiments\Crystal Pendant\Maureen_2020\videoParse\VID_20200727_135748')