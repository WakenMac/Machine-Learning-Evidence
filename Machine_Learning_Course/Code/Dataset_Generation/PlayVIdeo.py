# Author: Waken Cean C. Maclang
# Date Last Edited: October 31, 2025
# Course: Machine Learning
# Task: Learning Evidence

# PlayVideo.py 
#     Test Program that tries to play recorded videos using OpenCV

# Works with Python 3.10.0

import cv2
import time

path = 'Machine_Learning_Course\\Trial Recordings\\[1] ArUco Boarder.mp4'
file_name = path.split('\\')[-1]
cap = cv2.VideoCapture(path)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(500 / fps)
print(fps, delay)
# frame_time = 1 / fps

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow(file_name, frame)

    # Press the escape key to close
    if cv2.waitKey(delay) == 27: 
        break

cap.release()
cv2.destroyAllWindows()