import pandas as pd
import sys
from PIL import Image, ImageDraw
import cv2
import os

def compressImg (input_,output_, q=90):
	im = Image.open(input_)

	im.save(output_,optimize=True,quality=q)
	return None

inputPath = r'E:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment51_15062021\\'
outputPath = r'E:\Experiments\Crystal Pendant\FocusedAnalysis\E51\E51Compressed'
os.chdir(inputPath)

names = os.listdir()

for name in names[86::4]:
	print(name)
	compressImg(name, outputPath + '\\compressed'+name)


