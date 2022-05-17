import pandas as pd
import sys
from PIL import Image, ImageDraw
import cv2
import os
import moviepy.editor as mpy

import ffmpeg


# path = r'E:\Experiments\Crystal Pendant\FocusedAnalysis\E51\E51Compressed'

inputPath = r'E:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment51_15062021\\'
outputPath = r'E:\Experiments\Crystal Pendant\FocusedAnalysis\E51\E51Compressed'
os.chdir(outputPath)

# def ImportImg (t):
names = os.listdir()

# 	# im = Image.open(names[t])
# 	im = cv2.imread(names[t]) 

# 	return im

framelist = []

for name in names[:10]:
	im = cv2.imread(name) 
	height, width, layers = im.shape
	print (height, width)
	size = (width,height)
	framelist.append(im)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'E:\Experiments\Crystal Pendant\FocusedAnalysis\E51\project.mp4',fourcc, 25, (1280,720))
 
for i in range(len(framelist)):
    out.write(framelist[i])
    print(i)
out.release()





