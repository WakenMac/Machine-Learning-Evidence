import cv2

# Load image
img = cv2.imread("Machine_Learning_Course/Code/ArUco_Markers/Creating_an_aruco_marker/aruco_marker_detect_four_100.png")
img = cv2.imread("Machine_Learning_Course/Code/ArUco_Markers/Creating_an_aruco_marker/aruco_marker_detect_four_100.png")

# Convert to grayscale (ArUco works best on grayscale)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Detect markers
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, rejected_pts = detector.detectMarkers(gray)

# Draw detected markers
detected_img = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)

print("Detected IDs:", ids)
cv2.imshow("Detected Marker", detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()