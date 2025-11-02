# Author: Waken Cean C. Maclang
# Date Last Edited: October 31, 2025
# Course: Machine Learning
# Task: Learning Evidence

# DetectArucoLive.py 
#     It consists of the entire Algorithmic Framework to Detect the AR Piano and Hands, whilst playing the piano key.

# Works with Python 3.10.0

import cv2
from cv2 import aruco
import numpy as np
from hand_press_detector import HandPressDetector

# Details were taken from (L = 640 x W = 480) dimension resized image
KNOWN_AREA = 25360
KNOWN_DISTANCE = 100    # In Centimeters

def init_detector():
    """
    Initializes the aruco detector.
    @returns   An ArUco Detector object suited to detect DICT_4X4_40 markers.
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    return aruco.ArucoDetector(aruco_dict, parameters)

def generate_boarder_points(corners, ids):
    """
    Method to get the 4 points of our boarder given the coordinates of the two upper ArUco markers

    @param corners  The array of 4-clockwise coordinates for each ArUco marker
    @param ids      The order of IDs detected by the ArUco detector.

    @returns        An array of four points for our boarder
    """
    corner_idx = [1, 0]
    points = [[], [], [], []]

    for i, corner_set in enumerate(corners):
        if i > 1 and ids[i] > 1:
            return None
        temp_i = 1 if i >= 1 else 0
        pts = corner_set[0].astype(int)
        points[ids[i][0]] = pts[corner_idx[ids[temp_i][0]]].tolist()

    # Trick to get the remaining bottom corners
    # The piano has a 2:3 ratio with its height and width respectively.
    point_distance = ((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[1][1]) ** 2) ** .5  # Distance formula
    y_upper = max(points[0][1], points[1][1])
    y_lower = int(point_distance * .6666) + y_upper  # Gets the proportion for the width
    points[2] = [points[1][0], y_lower]
    points[3] = [points[0][0], y_lower]

    points = np.reshape(points, shape=(4, 2))
    return points

def draw_boarder(image, corners, ids):
    """
    Draws the inner border given the ArUco markers
    The image must contain 2 or 4 ArUco markers else the method will return the un-annotated image.

    @param image     The image captured by our video capture device.
    @param corners   The array of 4-clockwise coordinates for each ArUco marker detected.
    @param ids       The order of IDs detected by the ArUco detector.

    @return   The original image (If there are a lack of corners or ids), or an annotated image with the ArUco marker border or the piano border.
    """
    if ids is None or len(ids) != 2 or len(corners) != 2:
        return image
    
    points = generate_boarder_points(corners, ids)
    if points is None:
        return image
    return cv2.polylines(image, [points], True, (0, 255, 0), 2)

def get_min_max(corners, ids):
    """
    Method to get the minimum and maximum values of our x and y variables of our border.
    This will be a pre-requisite to finding the length and height of our pixel border, as well as the key lengths.
    
    @param corners   The array of 4-clockwise coordinates for each ArUco marker detected.
    @param ids       The order of IDs detected by the ArUco detector.

    @returns A dictionary of maximum and minimum values of the points (x & y coordinates):
                {'x-min':x_min,
                'x-max':x_max,
                'y-min':y_min,
                'y-max':y_max}

    Note:   y-value increases as you move down the image.
    """
    points = generate_boarder_points(corners, ids)
    if points is None:
        return None
    return {'x-min':points[0][0],
            'x-max':points[1][0],
            'y-min':points[0][1],
            'y-max':points[3][1]}

def get_border_dimensions(border_values:dict):
    """
    Gets the height and width of our border
    @returns    An list containing the height and width of the pixel boarder
    """
    x = border_values['x-max'] - border_values['x-min']
    y = border_values['y-max'] - border_values['y-min']
    return [y, x]

def get_key_width(boarder_width:int) -> int:
    """
    Gets the length for each key
    @returns    An integer representing the width for each key
    """
    return int(boarder_width / 9)

def get_key_hovered(fingertip_coordinates:list, key_width:int, boarder_values:dict) -> str:
    """
    Finds which key the fingertip is hovering over.
    """
    if (fingertip_coordinates is None or len(fingertip_coordinates) != 2 or boarder_values is None or
        fingertip_coordinates[1] > boarder_values['y-max'] or fingertip_coordinates[1] < boarder_values['y-min']):
        return 'NA'
    
    xpixel_location = fingertip_coordinates[0] - (boarder_values['x-min'] + key_width)
    key = float(xpixel_location / key_width)

    if key < 0:
        return 'NA'
    elif key >= 0 and key <= 1:
        return 'A'
    elif key > 1 and key <= 2:
        return 'B'
    elif key > 2 and key <= 3:
        return 'C'
    elif key > 3 and key <= 4:
        return 'D'
    elif key > 4 and key <= 5:
        return 'E'
    elif key > 5 and key <= 6:
        return 'F'
    elif key > 6 and key <= 7:
        return 'G'
    else:
        return 'NA'

def get_aruco_area(corners) -> int :
    """
    Gets the average area for all ArUco markers detected in the image.
    """
    if corners is None or len(corners) == 0:
        return -1
    
    total_area = 0
    for corner_set in corners:
        pts = corner_set[0].astype(int)
        total_area += cv2.contourArea(pts)
    return int(total_area / len(corners))

def get_piano_distance(corners) -> float:
    new_area = get_aruco_area(corners)
    if new_area == -1:
        return -1
    return KNOWN_DISTANCE * (KNOWN_AREA / new_area) ** 0.5
    
def main(camera_index:int):
    """
    Main method to run the AR Piano Model.
    """
    cap = cv2.VideoCapture(camera_index)
    aruco_detector = init_detector()
    hand_press_detector = HandPressDetector()

    if not cap.isOpened():
        print('Unable to access camera feed.')
        return
    else:
        while True:
            success, frame = cap.read()
            if not success:
                print('Unable to read frame')
                continue

            img = frame.copy()
            corners, ids, _ = aruco_detector.detectMarkers(img)
            detected_image = aruco.drawDetectedMarkers(img, corners, ids)
            detected_image = draw_boarder(img, corners, ids)

            detect_hands = corners is not None and ids is not None and len(corners) == 2 and len(ids) == 2

            if detect_hands:
                mp_img = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
                mp_img.flags.writeable = False
                results = hand_press_detector.hands.process(mp_img)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        pressed, fingertip_coords = hand_press_detector.detect_press(hand_landmarks, detected_image.shape)
                        hand_press_detector.draw_landmarks(detected_image, hand_landmarks, pressed)

                        boarder_values = get_min_max(corners, ids)
                        if boarder_values:
                            key_width = get_key_width(get_border_dimensions(boarder_values)[1])
                            key_hovered = get_key_hovered(fingertip_coords, key_width, boarder_values)

                            if pressed and key_hovered != 'NA':
                                print(f'Key {key_hovered} pressed!')

            cv2.imshow('HomePiano: My AR Piano', detected_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

main(0)