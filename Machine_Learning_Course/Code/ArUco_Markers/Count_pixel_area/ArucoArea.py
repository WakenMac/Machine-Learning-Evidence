import cv2
import numpy as np

img = cv2.imread("aruco_marker.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

corners, ids, _ = detector.detectMarkers(gray)

if ids is not None:
    for i, corner_set in enumerate(corners):
        # Get the 4 corner points
        pts = corner_set[0].astype(int)
        
        # Compute polygon area using OpenCV contourArea
        area = cv2.contourArea(pts)
        print(f"Marker ID {ids[i][0]} - Pixel Area: {area:.2f}")
        
        # Draw the polygon for visualization
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
else:
    print("No marker detected!")

cv2.imshow("Marker with Area", img)
cv2.waitKey(0)
cv2.destroyAllWindows()