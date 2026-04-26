import cv2
import numpy as np

# Load image
img = cv2.imread("Machine-Learning-Evidence/Machine_Learning_Course/Images/ArUco_Markers/Digital Piano/aruco_marker_detect_piano_2_blue_partial_200.png")
# img = cv2.imread("Machine_Learning_Course/Code/ArUco_Markers/Creating_an_aruco_marker/aruco_marker_detect_four_100.png")

# Convert to grayscale (ArUco works best on grayscale)
img = cv2.resize(img, [1280, 720])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Detect markers
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, rejected_pts = detector.detectMarkers(gray)

print(corners)

# Draw detected markers
detected_img = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
points = np.reshape([
        [204, 72],
        [1074, 72],
        [1074, 648],
        [204, 648]
    ], shape=(4, 2))
detected_img = cv2.polylines(detected_img, [points], True, (0, 255, 0), 2)

print("Detected IDs:", ids)
cv2.imshow("Detected Marker", detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()