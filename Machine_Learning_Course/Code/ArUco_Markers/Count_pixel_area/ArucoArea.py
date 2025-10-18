import cv2
import numpy as np

img = cv2.imread("Machine_Learning_Course\\Images\\ArUco_Markers\\Detect\\aruco_marker_detect_four_100.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

corners, ids, _ = detector.detectMarkers(gray)
print(corners)
print(ids)

points = []
corner_idx = [0, 1, 2, 3]   # Outer corner
# corner_idx = []   # Outer corner

if ids is not None:
    for i, corner_set in enumerate(corners):
        # Get the 4 corner points
        pts = corner_set[0].astype(int)
        points.append(pts[corner_idx[i]].tolist())

        print(pts)
        # Compute polygon area using OpenCV contourArea
        area = cv2.contourArea(pts)
        print(f"Marker ID {ids[i][0]} - Pixel Area: {area:.2f}")
        
        # Draw the polygon for visualization (For each ArUco Marker)
        # cv2.polylines(img, [pts], True, (0, 255, 0), 2)

else:
    print("No marker detected!")

# points[2], points[3] = points[3], points[2]
points = np.reshape(points, shape=(4, 2))
print(points)

detected_image = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
cv2.polylines(detected_image, [points], True, (0, 255, 0), 2)
cv2.imshow("Marker with Area", detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()