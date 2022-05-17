# -*- coding: utf-8 -*-
"""
Created on 03.12.20
Test stitch pictures (SEM) 
NEed to reinstall OpenCV

@author: Simon
"""
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import pandas as pd
from datetime import datetime
#Load img, 0 for grayscale
# img = cv2.imread(r'D:\Experiments\Crystal Pendant\Calibprofilo\e17_pythontest\Pic_e17.5_Cropped.jpg', 0)
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.imshow('image', gray_image)
# print(img.shape)


### YOU NEED TO verify THE DIRECTORY
path = r'D:\Experiments\Crystal Pendant\Calibprofilo\e17_pythontest'

#poslist = pd.read_csv(poslistcsv)
print(not(os.getcwd().endswith(path)))
if not(os.getcwd().endswith(path)):
     os.chdir(path)
# print(os.getcwd())

names = os.listdir() 

# crop_img = img[:920, :]
# cv2.imwrite('Cropped.jpg', crop_img)
images = []
for name in names:
    #Load img, 0 for grayscale
	img = cv2.imread(name)
	# converts to float, bugs with, bugs w/o !
	img = cv2.resize(img, (0,0), fx=1, fy=1)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = img/np.amax(img)

	images.append(img)
	# print(img.dtype)
	# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
	# img = img/np.amax(img)

	cv2.imshow('image',img[:,:])

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# crop_img = img[:920, :]
	# cv2.imwrite(path + '\\' + name[:-4] + '_Cropped.jpg', crop_img)
	# print(path + name[:-4] + '_Cropped.jpg')

# find the key points and descriptors with SIF

# print(images[0].shape)
print('end')
#show img
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.imshow('image',crop_img)
# stitcher = cv2.createStitcher(False)
stitcher = cv2.Stitcher_create()
print('shap:' + str(len(images[0].shape)))
print(images[0].shape)

results = stitcher.stitch(images[0])
print(results)


# cv2.waitKey(0)
# cv2.destroyAllWindows()