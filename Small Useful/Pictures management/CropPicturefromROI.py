import pandas as pd
import sys
from PIL import Image, ImageDraw
import cv2


imgPath = r'E:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\PendantCrystal_Experiment51_15062021\\'
# ROIpath = 
names = pd.read_csv(r'E:\Experiments\Crystal Pendant\FocusedAnalysis\E51\listmetadata1in4.csv')

ROIfile = pd.read_csv(r'E:\Experiments\Crystal Pendant\FocusedAnalysis\E51\RoiCavity.csv')

# ROIfile = pd.read_csv(ROIpath)
# print(ROIfile)
# print(ROIfile.iloc[0,4])

extendWindow = 0
# Size of the image in pixels (size of original image)
# (This is not mandatory)

n = 105
step = 20

for name, ROI in zip(names.iloc[10:n,2][::step],names.iloc[10:n,3][::step]):
	# Opens a image in RGB mode
	# print(ROIfile.iloc[:,1]==ROI[-14:],'what',ROI[-14:],'what')
	# print(name,ROI)
	# print(ROI[:-15])

	im = Image.open(imgPath + name)
	# print(names.iloc[10:,2][10],(ROIfile.iloc[:,1] == ROI[-15:])[0])
	# sth = names.iloc[10:,2][names.iloc[:,2] == ROI[-15:]]
	sth = ROIfile[ROIfile.iloc[:,1] == ROI[-15:]]

	print(ROIfile.iloc[1,1],'\n',ROIfile.iloc[:,1] == ROI[-14:])
	# print(ROIfile.iloc[-10,1],'\n',ROI[-14:])
	# print('\n',ROI[-14:])
	print('sth', sth)
	# Setting the points for cropped image
	# Rectangle around focus
	left = sth.iloc[0,4] - extendWindow
	top = sth.iloc[0,5] - extendWindow
	right = sth.iloc[0,4] + sth.iloc[0,6]*(1+extendWindow)
	bottom = sth.iloc[0,5] + sth.iloc[0,7]*(1+extendWindow)
	shape = (left, top, right, bottom)

	center = (sth.iloc[0,4] + sth.iloc[0,6]*0.5,sth.iloc[0,5] + sth.iloc[0,7]*0.5)

	# coordinates windows centered on focus (width, height)
	window = (1000, 500)

	focusleft = center[0] - 0.5*window[0]
	focustop = center[1] - 0.5*window[1]
	focusright= center[0] + 0.5*window[0]
	focusbottom= center[1] + 0.5*window[1]

	focusshape = (focusleft,focustop, focusright,focusbottom)
	print (focusshape)
	# Cropped image of above dimension
	# (It will not change original image)
	

	# print(shape)
	# im1 = ImageDraw.Draw(im)
	# im1.rectangle(shape,fill='red', outline ="red")
	im = im.crop(focusshape)
 
# Shows the image in image viewer
	# im.show() 
	im.save(r'E:\Experiments\Crystal Pendant\FocusedAnalysis\E51\'Cropped' + name, "jpeg")
	# im.save(r'Desktop\Image.png', "PNG")
	print('end')
	# break
# cv2.waitKey(0)
