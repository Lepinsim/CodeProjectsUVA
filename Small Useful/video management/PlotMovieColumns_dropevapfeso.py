import numpy as np
import cv2
import matplotlib.pyplot as plt

# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()

# bg = [[[0] * len(frame[0]) for _ in xrange(len(frame))] for _ in xrange(3)]


vid1 = cv2.VideoCapture('drop3 AddIsopropanol_atSaturation_2FPS_duration384s_v5microL.avi')
vid2 = cv2.VideoCapture('drop4 AddEthanol_atSaturation_20FPS_duration400s_v5microL_wScale.avi')


ar1 = np.array(vid1)
print(ar1.shape)
while(True):
    ret, frame1 = vid1.read()
    ret, frame2 = vid2.read()


    # # Resizing down the image to fit in the screen.
    # frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)

    # # creating another frame.
    # channels = cv2.split(frame)
    # frame_merge = cv2.merge(channels)

    # horizintally concatenating the two frames.
    final_frame = cv2.hconcat((frame1, frame2))

    # Show the concatenated frame using imshow.
    # cv2.imshow('frame',final_frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break