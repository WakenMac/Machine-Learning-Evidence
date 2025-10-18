import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Marker details
marker_id = 0
marker_size = 300   # In pixels

# Creating the marker image
marker_images = []
for id in range(4):
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, marker_size)
    marker_images.append(marker_img)

# For loop to save the aruco markers in the given directory
for i in range(4):
    cv2.imwrite(f'Machine_Learning_Course/Images/ArUco_Markers/{marker_size}_px/aruco_marker_{marker_size}_{i}.png', marker_images[i])

# Save and display
# cv2.imwrite("Machine_Learning_Course/Learning_Evidence/ArUco_Markers/Creating_an_aruco_marker/aruco_marker_0.png", marker_img)
# cv2.imshow('ArUco Marker', marker_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()