import cv2
from cv2 import aruco
import numpy as np

def init_detector():
    """Initializes the aruco detector"""
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    return aruco.ArucoDetector(aruco_dict, parameters)

def draw_boarder(image, corners, ids):
    """
    Draws the inner border given the ArUco markers
    The image must contain 4 ArUco markers else the method will return the un-annotated image.
    """
    if ids is None or len(ids) != 4 or len(corners) != 4:
        return image
    
    corner_idx = [1, 0, 3, 2]
    points = [[], [], [], []]

    for i, corner_set in enumerate(corners):
        pts = corner_set[0].astype(int)
        points[ids[i][0]] = pts[corner_idx[ids[i][0]]].tolist()
    points = np.reshape(points, shape=(4, 2))
    return cv2.polylines(image, [points], True, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(1)
    
    detector = init_detector()

    if not cap.isOpened():
        print('Unable to access camera feed.')
        return
    else:
        while True:
            success, frame = cap.read()
            
            if not success:
                print('Unable to read frame')
            else:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(img)
                # detected_image = aruco.drawDetectedMarkers(img, corners, ids)
                detected_image = draw_boarder(img, corners, ids)
                
                final_image = cv2.cvtColor(detected_image, cv2.COLOR_GRAY2BGR)
                cv2.imshow('HomePiano: My AR Piano', final_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()

main()

