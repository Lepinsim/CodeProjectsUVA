# Clahe

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\ExtractedROI\exp6_4\PendantCrystal_Experiment6_24072020_D5200 (4530342)_0547_DAY_27_TIME_11-18-234PicCrop.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
# cv.imshow('img', img)
# plt.show()


# img = cv.imread('tsukuba_l.png',0)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(4,4))
cl1 = clahe.apply(img)
cv.imwrite('F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\ExtractedROI\exp6_4\clahe_2.jpg',cl1)
print('end')