# Author: Waken Cean C. Maclang
# Date Last Edited: April 26, 2025
# Course: Machine Learning
# Task: Learning Evidence

# DetectArucoLive.py 
#     It consists of the entire Algorithmic Framework to Detect the AR Piano and Hands, whilst playing the piano key.
#     Designed to be used for live use.

# Works with Python 3.14.2

import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
import mediapipe as mp 
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import RunningMode, HandLandmarker, HandLandmarkerOptions 
import time

# Details were taken from (L = 640 x W = 480) dimension resized image
# Captured: 1280, 720
# Resized: 960, 540

KNOWN_AREA = 25360    # For the (L = 640 x W = 480) dimension
# KNOWN_AREA = 76872      # For the (L = 1280 x W = 720) dimension
KNOWN_DISTANCE = 10    # In Centimeters
H_matrix = None
minimum_quality = [1280, 720]
high_quality = [1920, 1080]

def init_detectors(camera_index:int):
    """
    Initializes the aruco detector.
    @returns   An ArUco Detector object suited to detect DICT_4X4_40 markers.
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    model_path = 'Machine-Learning-Evidence\Machine_Learning_Course\Code\hand_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = HandLandmarkerOptions(
        base_options=base_options, 
        num_hands = 2,
        running_mode = RunningMode.VIDEO
    )
    hand_detector = HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, minimum_quality[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, minimum_quality[1])
    return [cap, aruco_detector, hand_detector]

def generate_boarder_points(corners):
    """
    Method to get the 4 points of our boarder given the coordinates of the two upper ArUco markers

    @param corners  The array of 4-clockwise coordinates for each ArUco marker (Starts at the upper left)
    @param ids      The order of IDs detected by the ArUco detector.

    @returns        A 2-dimensional array of four points for our boarder

    Pseudocode:
    1. Arrange P1 and P2 together
    2. Solve for the angle near the point that's lower (One that's split in two)
    3. Calculate the adjacent angle (arcsin)
    4. Calculate the distance between P1 & P2, then get 2/3 of it (Lets call that L)
    5. Use the pendulum formula (sin and cos with L)
    """
    if corners is None or len(corners) != 2:
        return None

    points = [[], [], [], []]
     
    # Example of the corners variable: 
    # (array([[[522., 295.],
    #     [597., 300.],
    #    [606., 383.],
    #    [522., 373.]]], dtype=float32), array([[[1088.,  337.],
    #    [1160.,  335.],
    #    [1187.,  397.],
    #    [1114.,  396.]]], dtype=float32))

    # Compares the x-coordinate of the ArUco marker's point A (i.e., first point)
    if (corners[0][0][0][0] > corners[1][0][0][0]):
        points[0] = corners[1][0][1].astype(int).tolist()
        points[1] = corners[0][0][0].astype(int).tolist()
    else:
        points[0] = corners[0][0][1].astype(int).tolist()
        points[1] = corners[1][0][0].astype(int).tolist()
        
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]

    # Calculate for the angle of the lower point
    hypotenuse = np.sqrt(dx**2 + dy**2)
    rad_angle = np.arctan2(dy, dx)
    adj_angle = rad_angle + (np.pi / 2)

    # Solve for the x y coords using L and trigo
    height = int(hypotenuse * .6666)
    new_x, new_y = [int(height * np.cos(adj_angle)), int(height * np.sin(adj_angle))]

    points[2] = [
        points[1][0] + new_x,
        points[1][1] + new_y
    ]

    points[3] = [
        points[0][0] + new_x,
        points[0][1] + new_y
    ]

    points = np.reshape(points, shape=(4, 2))
    return points

def draw_boarder(image, points):
    """
    Draws the inner border given the set of 4 points generated from generate_piano_boarder() function
    The image must contain 4 points else the method will return the un-annotated image.

    @param image     The image captured by our video capture device.
    @param corners   The array of 4-clockwise coordinates for each ArUco marker detected.
    @param ids       The order of IDs detected by the ArUco detector.

    @return   The original image (If there are a lack of corners or ids), or an annotated image with the ArUco marker border or the piano border.
    """
    if points is None or len(points) != 4:
        return image
    return cv2.polylines(image, [points], True, (0, 255, 0), 2)

def apply_homography(image, piano_boarder) -> None:
    """
    A method that applies Homographical Transformations to the captured image as an image 
    pre-processing task. This also saves the generated H matrix to the 'H_matrix' global
    variable

    @param piano_boarder The array of 4-clockwise coordinates for each ArUco marker detected.
    """
    global H_matrix

    if (piano_boarder is None):
        return image

    canonical_coordinates = np.array([
        [204, 72],
        [1074, 72],
        [1074, 648],
        [204, 648]
    ], dtype=np.float32)

    H_matrix, _ = cv2.findHomography(piano_boarder, canonical_coordinates)
    return cv2.warpPerspective(image, H_matrix, (minimum_quality[0], minimum_quality[1]))

def get_min_max(points):
    """
    Method to get the minimum and maximum values of our x and y variables of our border.
    This will be a pre-requisite to finding the length and height of our pixel border, as well as the key lengths.
    
    @param points   The array of 4-clockwise points of the piano boarder.

    @returns A dictionary of maximum and minimum values of the points (x & y coordinates):
                {'x-min':x_min,
                'x-max':x_max,
                'y-min':y_min,
                'y-max':y_max}

    Note:   y-value increases as you move down the image.
    """
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
    """
    Calculates the average distance from the piano by getting the ratio of the area of the ArUco
    markers detected in the image with respect to the reference point (KNOWN DISTANCE of 100cm and
    KNOWN AREA of 78k pixels)

    @return distance The calculated distance of the camera to the paper piano
    """
    new_area = get_aruco_area(corners)
    if new_area == -1:
        return -1
    return KNOWN_DISTANCE * (KNOWN_AREA / new_area) ** 0.5

def trackFingers(hand_landmarks:list, piano_boarder):
    boarder_values = get_min_max(piano_boarder)
    if boarder_values is None:
        return
    key_width = get_key_width(get_border_dimensions(boarder_values)[1])

    # key_hovered = get_key_hovered(fingertip_coords, key_width, boarder_values)
    #     # if pressed and key_hovered != 'NA':
    #     if key_hovered != 'NA':
    #         print(f'Key {key_hovered} pressed!')

def handleImageOverlay(image, text):
    org = (10, 30)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (255, 255, 255)  # White
    thickness = 2
    lineType = cv2.LINE_AA

    return cv2.putText(image, text, org, fontFace, fontScale, color, 
                                thickness, lineType)

def appendRecordedLandmarks(data_dict:dict, H_matrix:list, frame_count:int, landmark_index:int, x_coord:int, y_coord:int):
    data_dict.get('frame_index').append(frame_count)
    data_dict.get('hand_landmark').append(landmark_index)

    canonical_points = cv2.perspectiveTransform(
        np.array([[[x_coord, y_coord]]], dtype=np.float32),
        H_matrix
    )[0][0]
    data_dict.get('x_coords').append(canonical_points[0])
    data_dict.get('y_coords').append(canonical_points[1])
    
def saveRecordedLandmarks(data_dict:dict) -> None:
    """
    Records the saved coordinates into a Pandas DataFrame in long format. The csv is stored in:
    Machine-Learning-Evidence\Machine_Learning_Course\Code\ArUco_Markers\Detecting_an_aruco_marker\\test_coordinates.csv

    @param data_dict           Contains the dictionary of data that will be saved as a CSV with
                               the ff values:
                                - frame_index        Number of the frame processed
                                - hand_landmark      Index of the joint (See MediaPipe Docu as a guide)
                                - x_coords           X-coordinate of the given joint/landmark
                                - y_coords           Y-coordinate of the given joint/landmark
    """
    data_path = 'Machine-Learning-Evidence\Machine_Learning_Course\Code\ArUco_Markers\Detecting_an_aruco_marker\\test_coordinates.csv'
    data = None
    try:
        data = pd.read_csv(data_path)
    except (FileNotFoundError):
        print('No File named recordings.csv exists.')

    new_data = pd.DataFrame(data_dict)

    if data is not None:
        pd.concat([data, new_data], axis=0, ignore_index=True).to_csv(data_path, index=False)
    else:
        new_data.to_csv(data_path, index=False)

def main(camera_index:int):
    """
    Main method to run the AR Piano Model.
    """
    lowest_fps = 10000
    frame_count = 1
    data_dict = {
        'frame_index': [],
        'hand_landmark':[],
        'x_coords':[],
        'y_coords':[]
    }
    prev_frame_time = distance = 0
    transformed_image = None
    cap, aruco_detector, hand_detector = init_detectors(camera_index)

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
            piano_boarder = generate_boarder_points(corners)
            detected_image = draw_boarder(img, piano_boarder)

            markers_detected = corners is not None and ids is not None and \
                piano_boarder is not None and len(piano_boarder) == 4 and len(ids) == 2

            if markers_detected:
                distance = get_piano_distance(corners)
                transformed_image = apply_homography(detected_image, piano_boarder)
                image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                frame_timestamp_ms = int(time.time() * 1000)
                result = hand_detector.detect_for_video(mp_image, frame_timestamp_ms)
                if result.hand_landmarks:
                    for hand_landmark in result.hand_landmarks:
                        for i, landmark in enumerate(hand_landmark):
                            h, w, c = detected_image.shape
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            appendRecordedLandmarks(data_dict, H_matrix, frame_count, i, x, y,)
                            # cv2.circle(detected_image, (x, y), 5, (0, 255, 0), -1)

                    # Method for the piano playing logic
                    trackFingers(result.hand_landmarks[0], piano_boarder)
            
            # Handle FPS
            current_time = time.time()
            time_elapsed = current_time - prev_frame_time
            if time_elapsed > 0:
                fps = 1 / time_elapsed
            else:
                fps = 0
            prev_frame_time = current_time
            lowest_fps = fps if fps < lowest_fps and fps > 0 else lowest_fps 
            text = f'Piano Distance: {distance:.2f} cm. FPS: {fps:.1f}'
            detected_image = handleImageOverlay(detected_image, text)
            
            frame_count += 1
            transformed_image = detected_image
            cv2.imshow('HomePiano', transformed_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        saveRecordedLandmarks(data_dict)
        cap.release()
        cv2.destroyAllWindows()
        print('Lowest FPS gained:', lowest_fps)

main(0)
