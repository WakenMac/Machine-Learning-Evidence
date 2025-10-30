# Author: Waken Cean C. Maclang
# Date Last Edited: October 31, 2025
# Course: Machine Learning
# Task: Learning Evidence

# DetectArucoLive.py 
#     It consists of the entire Algorithmic Framework to Detect the AR Piano and Hands, whilst playing the piano key.

# Works with Python 3.10.0

import cv2
from cv2 import aruco
import mediapipe as mp
import numpy as np

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

def init_media_pipe_tools():
    """
    Initializes MediaPipe's tools for hand-coordinate marking
    @returns    A list of MediaPipe tools for hand marking and detection
    """
    return [mp.solutions.hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2
    ), mp.solutions.hands, mp.solutions.drawing_utils]

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

    # if ids is None or len(ids) != 4 or len(corners) != 4:
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

def get_fingertip_coordinates(mp_hands, shape:list, hand_landmarks):
    """
    Gets MediaPipe's normalized fingertip coordinates and transforms them to get the pixel fingertip coordinates
    
    @param mp_hands  To use the HandLandmark Enums
    @param shape     The height and width of the image
    @param results   The landmarks for one or two hand(s)

    @returns         A list containing the x and y pixel coordinate of the tip of the index finger.
    """
    # Normalized fingertip coordinates
    fingertip_coordinates = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y]
    h, w, = shape

    # Pixel fingertip coordinates
    fingertip_coordinates[0] = int(w * fingertip_coordinates[0])
    fingertip_coordinates[1] = int(h * fingertip_coordinates[1])
    return fingertip_coordinates

def get_key_width(boarder_width:int) -> int:
    """
    Gets the length for each key
    @returns    An integer representing the width for each key
    """
    return int(boarder_width/ 9)

def get_key_hovered(fingertip_coordinates:list, key_width:int, boarder_values:dict) -> str:
    """
    Finds which key the fingertip is hovering over.
    The results may be inaccurate for slanted boarders (i.e., AR Piano is rotated).
    Calculations are made immediately to prevent repetitive if-else statements for error handling. Instead, error handling is made post-calculations.

    @param fingertip_coordinates    A list containing the x and y pixel coordinate of the index 
                                    finger
    @param key_width                The width for each piano key
    @param boarder_values           A dictionary containing the maximum and minimum values for our 
                                    x and y coordinates. See get_border_dimensions() to see the 
                                    key-value pairs

    @returns    A key from (A, B, C, ... , G) if the fingertip's coordinate is within the border 
                and NA if not.

    EXAMPLE:

    [1] A fingertip is hovering over a key.

    Boarder's coordinates are
    [[102 268]
     [323 268]
     [323 426]
     [102 426]]

    Fingertip's coordinates are [133, 289] for x, and y respectively
    Key Length is ((323 - 102) / 9), which is 25.

    xpixel_location = 133 - (102 + 25) = 6
    key = float(6 / 25) = 0.24

    Thus, the fingertip is hovering over the A key. The function returns 'A'

    [2] A fingertip is not within the boarder.
    Boarder's coordinates are
    [[102 268]
     [323 268]
     [323 426]
     [102 426]]

    Fingertip's coordinates are [35, 289] for x, and y respectively
    Key Length is ((323 - 102) / 9), which is 25.

    xpixel_location = 35 - (102 + 25) = -92   # This is a sign that the fingertip is not within the boarder
    key = float(-92 / 25) = -3.68

    Thus the fingertip is not in the boarder. The function returns 'NA'

    [3] The fingertip is over the boarder's x_max coordinate.
    Boarder's coordinates are
    [[102 268]
     [323 268]
     [323 426]
     [102 426]]

    Fingertip's coordinates are [400, 289] for x, and y respectively
    Key Length is ((323 - 102) / 9), which is 25.

    xpixel_location = 400 - (102 + 25) = 273   # This is a sign that the fingertip is not within the boarder
    key = float(273 / 25) = 10.92

    Thus the fingertip is beyond the boarder's x-max value. The function returns 'NA'
    """
    # Error catching
    if (fingertip_coordinates is None or len(fingertip_coordinates) != 2 or boarder_values is None or
        fingertip_coordinates[1] > boarder_values['y-max'] or fingertip_coordinates[1] < boarder_values['y-min']):
        return 'NA'
    
    # Calculates which key the fingertip is hovering over
    # Checks if the fingertip's x-coordinate is within the border (+ means yes, - means no)
    xpixel_location = fingertip_coordinates[0] - (boarder_values['x-min'] + key_width)
    
    # Gets the value representing which key it is hovering over
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

    @param corners    The corners of the ArUco markers detected
    @returns          The average area of all markers detected
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

    # Corners are not present
    if new_area == -1:
        return -1

    # Uses the Pinhole Camera Model
    return KNOWN_DISTANCE * (KNOWN_AREA / new_area) ** 0.5
    

def main(camera_index:int):
    """
    Main method to run the AR Piano Model.

    @param camera_index:int     The index on which camera to use
                                0 for DroidCam, 1 for Laptop Cam
    """
    cap = cv2.VideoCapture(camera_index)
    aruco_detector = init_detector()
    hand_detector, mp_hands, mp_drawing = init_media_pipe_tools()

    detect_hands = mark_hands = False
    results = None
    distance = 0

    if not cap.isOpened():
        print('Unable to access camera feed.')
        return
    else:
        while True:
            success, frame = cap.read()
            detect_hands = mark_hands = False
            key_hovered = 'NA'

            if not success:
                print('Unable to read frame')
            else:
                # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = frame.copy()

                # Detects the ArUco Markers
                corners, ids, _ = aruco_detector.detectMarkers(img)
                detected_image = aruco.drawDetectedMarkers(img, corners, ids)
                detected_image = draw_boarder(img, corners, ids)
                distance = get_piano_distance(corners)

                if corners is not None and ids is not None and len(corners) == 2 and len(ids) == 2:
                    detect_hands = True

                if detect_hands:    
                    # Detects media pipe's finger coordinates and draws them to the image
                    mp_img = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
                    mp_img.flags.writeable = False
                    results = hand_detector.process(mp_img)

                    if results is not None and results.multi_hand_landmarks is not None:
                        mark_hands = True

                if mark_hands:
                    hand_landmarks = results.multi_hand_landmarks[0] # Gets the second hand

                    mp_drawing.draw_landmarks(detected_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    fingertip_coordinates = get_fingertip_coordinates(mp_hands, detected_image.shape[:2], hand_landmarks)

                    # Gets the key hovered
                    boarder_values = get_min_max(corners, ids)
                    key_width = get_key_width(get_border_dimensions(boarder_values)[1])
                    key_hovered = get_key_hovered(fingertip_coordinates, key_width, boarder_values)
                    print(f'Finger at {fingertip_coordinates} pressed {key_hovered} with distance of {distance}')
                
                # Pa plug nalang ng audio player natin here please~
                # I left some code documentations na din for the get_key_hovered() method
                if key_hovered != 'NA':
                    pass
                
                # final_image = cv2.cvtColor(detected_image, cv2.COLOR_GRAY2BGR)
                cv2.imshow('HomePiano: My AR Piano', detected_image)

                # Code to end reading the image.
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
        
        cap.release()
        cv2.destroyAllWindows()

# For DroidCam Client
main(0)

# For Laptop Webcam
# main(1)