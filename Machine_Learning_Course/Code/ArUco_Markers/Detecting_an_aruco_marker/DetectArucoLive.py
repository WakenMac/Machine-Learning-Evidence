import cv2
from cv2 import aruco
import numpy as np

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
    pass

def draw_boarder(image, corners, ids):
    """
    Draws the inner border given the ArUco markers
    The image must contain 4 ArUco markers else the method will return the un-annotated image.

    @param image     The image captured by our video capture device.
    @param corners   The array of 4-clockwise coordinates for each ArUco marker detected.
    @param ids       The order of IDs detected by the ArUco detector.

    @return   The original image (If there are a lack of corners or ids), or an annotated image with the ArUco marker border or the piano border.
    """
    if ids is None or len(ids) != 4 or len(corners) != 4:
        return image
    
    corner_idx = [1, 0, 3, 2]
    points = [[], [], [], []]

    for i, corner_set in enumerate(corners):
        pts = corner_set[0].astype(int)
        points[ids[i][0]] = pts[corner_idx[ids[i][0]]].tolist()
    points = np.reshape(points, shape=(4, 2))
    print(points)
    return cv2.polylines(image, [points], True, (0, 255, 0), 2)

def get_min_max(corners, ids):
    """
    Method to get the minimum and maximum values of our x and y variables of our border.
    This will be a pre-requisite to finding the length and height of our pixel border, as well as the key lengths.
    
    @param corners   The array of 4-clockwise coordinates for each ArUco marker detected.
    @param ids       The order of IDs detected by the ArUco detector.

    @returns A dictionary of values:
                {'x-min':x_min,
                'x-max':x_max,
                'y-min':y_min,
                'y-max':y_max}
    """
    pass

def get_border_dimensions(border_values:dict):
    """
    Gets the height and width of our border
    @returns    An list containing the height and width of the pixel boarder
    """
    x = border_values['x-max'] - border_values['x-min']
    y = border_values['y-max'] - border_values['y-min']
    return [y, x]

def get_key_length(boarder_width:int):
    """
    Gets the length for each key
    @returns    The pixel length for each key
    """
    return int(boarder_width/ 9)

def get_key_hovered(fingertip_coordinates:list, key_length:int, boarder_values:dict):
    """
    Finds which key the fingertip is hovering over.
    The results may be inaccurate for slanted boarders (i.e., AR Piano is rotated).
    Calculations are made immediately to prevent repetitive if-else statements for error handling. Instead, error handling is made post-calculations.

    @returns    A key from (A, B, C, ... , G) if the fingertip's coordinate is within the border and NA if not.

    EXAMPLE:

    [1] A fingertip is hovering over a key

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

    [2] A fingertip is not within the boarder
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

    [3] The fingertip is over the boarder's x_max coordinate
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
    xpixel_location = fingertip_coordinates[0] - (boarder_values['x-min'] + key_length)
    
    # Gets the value representing which key it is hovering over
    key = float(xpixel_location / key_length)

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
                # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = frame.copy()

                # Detects the ArUco Markers
                corners, ids, _ = detector.detectMarkers(img)
                detected_image = aruco.drawDetectedMarkers(img, corners, ids)
                detected_image = draw_boarder(img, corners, ids)

                # Detects media pipe's finger coordinate

                # Gets the key hovered
                boarder_values = get_min_max(corners, ids)
                width = get_border_dimensions(boarder_values)[1]
                key_length = get_key_length(width)
                key_hovered = get_key_hovered([], key_length, boarder_values)

                # final_image = cv2.cvtColor(detected_image, cv2.COLOR_GRAY2BGR)
                cv2.imshow('HomePiano: My AR Piano', detected_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()

main()

