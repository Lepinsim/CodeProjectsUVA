import pandas as pd
import sys
from PIL import Image, ImageDraw
import cv2
import os
import math

# def compressImg (input_,output_, q=90):
# 	im = Image.open(input_)

# 	im.save(output_,optimize=True,quality=q)
# 	return None

# inputPath = r'E:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment51_15062021\\'
outputPath = r'E:\Experiments\Crystal Pendant\FocusedAnalysis\E51\MeasureProjectedLengths_421pics.csv'
# os.chdir(inputPath)

file  = pd.read_csv(outputPath)

def DetectDuplicate(thelist):
  seen = set()
  duplicates =[]
  for x in thelist:
    if x in seen: 
    	duplicates.append(x)
    	# return duplicates

    seen.add(x)

  return duplicates


xlist = file.iloc[:,3]

l1 = [ ]
# names = os.listdir()
l2 = []
print(file)

ltot = []
alpha = []
actuallength = []
lbl = []
for i, x in enumerate(xlist[:]):
	# print(x,i)
	# if not()
	# print(i )

	diff = - x + xlist[i+1]
	# print(diff)
	if (i % 3) == 0:
		l1.append(diff)
		lsum = xlist[i+2] - xlist[i]
		ltot.append(lsum)
		a = 2* math.atan(diff/lsum - 0.5)
		alpha.append(a)
		actuallength.append(diff*(2**(0.5))/(math.cos(a) + math.sin(a)) )
		# print(file.iloc[i,-1])
		lbl.append(file.iloc[i,-1])


	if (i % 3) == 1:
		l2.append(diff)
		
		# ltot.append(xlist[i+1] - xlist[i-1])


	if i == xlist.shape[0]-2:
		break

print(DetectDuplicate(lbl))
# print(actuallength)
# print(len(ltot), len(l1))
# lf = 
lengthdf = pd.DataFrame({'l1':l1, 'l2':l2, 'ltot':ltot,'Crystal orientation (rad)':alpha,'Actual length':actuallength } )

lengthdf.to_csv(r'E:\Experiments\Crystal Pendant\FocusedAnalysis\E51\CrystalOrientation_421pics.csv')
# os.chdir(inputPath))

# for name in names[86:506:]:
# 	print(name)
# # 	compressImg(name, outputPath + '\\compressed'+name)

# pi = math.pi

# print(math.sin(1-(pi/4)), math.cos( 1))