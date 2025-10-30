# Author: Waken Cean C. Maclang
# Date Last Edited: October 31, 2025
# Course: Machine Learning
# Task: Learning Evidence

# ArucoArea.py 
#     A test program that gets the pixel area of an detected ArUco marker based on an uploaded picture

# Works with Python 3.10.0

import cv2
import numpy as np

img = cv2.imread("Machine_Learning_Course\\Images\\ArUco_Markers\\Detect\\aruco_marker_detect_100cm.jpg")
print(img.shape[:2])
img = cv2.resize(img, (640, 480))

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

corners, ids, _ = detector.detectMarkers(img)

points = []
corner_idx = [0, 1, 2, 3]   # Outer corner
# corner_idx = []   # Outer corner

if ids is not None:
    for i, corner_set in enumerate(corners):
        # Get the 4 corner points
        pts = corner_set[0].astype(int)

        for j in range(len(corner_set[0])):
            points.append(pts[corner_idx[j]].tolist())
        print(points)

        # Compute polygon area using OpenCV contourArea
        area = cv2.contourArea(pts)
        print(f"Marker ID {ids[i][0]} - Pixel Area: {area:.2f}")

else:
    print("No marker detected!")

# points[2], points[3] = points[3], points[2]
print('Points: \n', points)
points = np.reshape(points, shape=(4, 2))
# print(points)

detected_image = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
cv2.polylines(detected_image, [pts], True, (0, 255, 0), 2)
cv2.imshow("Marker with Area", detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()