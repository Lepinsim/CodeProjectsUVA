import cv2
import os

image_folder = r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\ExtractedROI\exp26_0'
video_name = r'F:\Experiments\Crystal Pendant\Maureen - Pendant Crystal\ExtractedROI\video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 20, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
video.release()