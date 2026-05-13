# Author: Waken Cean C. Maclang
# Date Last Edited: April 26, 2026
# Course: Machine Learning
# Task: Learning Evidence

# DetectArucoLive.py 
#     It consists of the entire Algorithmic Framework to Detect the AR Piano and Hands, whilst playing the piano key.
#     Designed to be used for data collection

# Works with Python 3.14.2

import os
import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
import mediapipe as mp 
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode, HandLandmarker, HandLandmarkerOptions 
import time
import keyboard
from pathlib import Path

# Details were taken from (L = 640 x W = 480) dimension resized image
# Captured: 1280, 720
# Resized: 960, 540

KNOWN_AREA = 25360    # For the (L = 640 x W = 480) dimension (Most accurate)
# KNOWN_AREA = 76872      # For the (L = 1280 x W = 720) dimension
KNOWN_DISTANCE = 10    # In Centimeters
H_matrix = None
minimum_quality = [1280, 720]
high_quality = [1920, 1080]
fps = total_frames = None
PIANO_BORDER_GLOBAL = None
led_lit = [0, 0, 0, 0, 0, 0, 0]

# Manipulate this:
HEADLESS_MODE = True # Option to show the image

def init_detectors(video_path:str):
    """
    Initializes the aruco detector.
    @param camera_index

    @returns   An ArUco Detector object suited to detect DICT_4X4_40 markers.
    """
    global fps, total_frames

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    model_path = 'Machine-Learning-Evidence\Machine_Learning_Course\Code\hand_landmarker.task'
    base_options = BaseOptions(model_asset_path=model_path)
    options = HandLandmarkerOptions(
        base_options=base_options, 
        num_hands = 2,
        running_mode = RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, minimum_quality[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, minimum_quality[1])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    hand_detector = HandLandmarker.create_from_options(options)
    return [cap, aruco_detector, hand_detector]

def printCapDetails(cap, video_path: str) -> bool:
    """
    Method to print out the details of the cap
    If the wrong video is being played, stops the Frame Processor from starting
    """
    file_name = os.path.basename(video_path)
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps > 0:
        duration_sec = total_frames / fps
        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)
        duration_str = f"{mins}m {secs}s"
    else:
        duration_str = "Unknown"

    print("="*45)
    print(" 🎬 VIDEO METADATA REPORT ")
    print("="*45)
    print(f"File Name:    {file_name}")
    print(f"File Size:    {file_size_mb:.2f} MB")
    print(f"Resolution:   {width} x {height} pixels")
    print(f"Framerate:    {fps:.2f} FPS")
    print(f"Total Frames: {total_frames}")
    print(f"Duration:     {duration_str} ({duration_sec:.2f} total seconds)")
    print("="*45)

    user_input = input('If ready to proceed to capture data, enter "Start".')
    if user_input != 'Start':
        return False
    return True

def load_existing_roi(video_name):
    """Checks the CSV for existing ROIs for this specific video."""
    roi_csv_path = Path(r'Machine-Learning-Evidence\Machine_Learning_Course\Code\Data Collection') / "roi_configs.csv"
    
    if roi_csv_path.exists():
        df_roi = pd.read_csv(roi_csv_path)
        # Filter for current video
        video_data = df_roi[df_roi['video_name'] == video_name]
        
        if not video_data.empty:
            print(f"--> Found saved ROIs for {video_name}. Loading...")
            # Reconstruct the list of 7 boxes (x, y, w, h)
            boxes = []
            for i in range(1, 8):
                row = video_data[video_data['LED_no'] == i].iloc[0]
                boxes.append((int(row['x']), int(row['y']), int(row['threshold'])))
            return boxes
    return None

def save_new_roi(video_name, frame_idx, roi_boxes):
    """Saves the 7 selected ROIs to the CSV."""
    roi_csv_path = Path(r'Machine-Learning-Evidence\Machine_Learning_Course\Code\Data Collection') / "roi_configs.csv"
    
    data_list = []
    for i, box in enumerate(roi_boxes):
        x, y, w, h = box
        center_x = x + (w // 2)
        center_y = y + (h // 2)
        
        data_list.append({
            'video_name': video_name,
            'frame_no': frame_idx,
            'LED_no': i + 1,
            'x': center_x, 'y': center_y
        })
    
    df_new = pd.DataFrame(data_list)
    # Append to CSV (header only if new file)
    df_new.to_csv(roi_csv_path, mode='a', header=not roi_csv_path.exists(), index=False)
    print(f"--> ROIs for {video_name} saved to {roi_csv_path.name}")

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
    
def saveDistanceData(video_name, distance, file_path):
    """
    Saves the static distance of a video to a CSV file.
    Columns: video_name, distance
    """
    csv_path = Path(file_path)
    
    # Prepare the new row
    new_entry = pd.DataFrame([{'video_name': video_name, 'distance': distance}])

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Avoid duplicate entries for the same video
        if video_name in df['video_name'].values:
            return
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_csv(csv_path, index=False)
    print(f"--> Distance for {video_name} saved: {distance:.2f} cm")

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
    """
    Appends the recorded landmark coordinates to the dictionary.
    This is a pre-requisite step before we finalize and append our data to the csv. 
    """
    data_dict.get('frame_index').append(frame_count)
    data_dict.get('hand_landmark').append(landmark_index)

    canonical_points = cv2.perspectiveTransform(
        np.array([[[x_coord, y_coord]]], dtype=np.float32),
        H_matrix
    )[0][0]
    data_dict.get('x_coords').append(canonical_points[0])
    data_dict.get('y_coords').append(canonical_points[1])

def saveRecordedLandmarks(
    data_dict:dict, 
    file_path:str) -> None:

    """
    Records the saved coordinates into a Pandas DataFrame in long format

    @param data_dict           Contains the dictionary of data that will be saved as a CSV with
                               the ff values:
                                - user               ID for the person being recorded
                                - video_name         Name of the video recording
                                - frame_index        Number of the frame processed
                                - hand_landmark      Index of the joint (See MediaPipe Docu as a guide)
                                - x_coords           X-coordinate of the given joint/landmark
                                - y_coords           Y-coordinate of the given joint/landmark

    @param file_path           The file path to access the csv file.
    """
    data = None
    try:
        data = pd.read_csv(file_path)
    except (FileNotFoundError):
        print('No File named recordings.csv exists.')

    new_data = pd.DataFrame(data_dict)

    if data is not None:
        pd.concat([data, new_data], axis=0, ignore_index=True).to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, index=False)

def trackLEDs(video_name, frame_count, frame, roi_boxes):
    """
    Checks the center pixel of each ROI for LED activation and logs to CSV.
    """
    global led_lit
    # Define the output path (Same directory as the script)
    csv_path = Path(r'Machine-Learning-Evidence\Machine_Learning_Course\Code\Data Collection') / "led_ground_truth.csv"
    
    # LED States (1 for Lit, 0 for Unlit)
    led_states = []
    
    for i, (x, y, threshold) in enumerate(roi_boxes):
        h_img, w_img, _ = frame.shape
        r_min = r_max = 150
        if 0 <= x < w_img and 0 <= y < h_img:
            b, g, r = frame[y, x]

            # Simple threshold: If Red component is high, LED is lit
            # You may need to adjust '200' based on your LED brightness
            if r >= threshold: 
                led_states.append(1)
                led_lit[i] = led_lit[i] + 1
            else:
                led_states.append(0)

            if r > r_max: 
                r_max = r
            if r < r_min: 
                r_min = r

        else:
            led_states.append(0)

    cols = ['file_name', 'frame'] + [f'led_{i+1}' for i in range(len(roi_boxes))]
    df_row = pd.DataFrame([[video_name, frame_count] + led_states], columns=cols)
    df_row.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)

def main(user:str, video_path:str, file_path:str, start_frame:int, distance_file_path:str):
    global PIANO_BORDER_GLOBAL, H_matrix, HEADLESS_MODE
    distance = -1
    video_name = video_path.split('\\')[-1]

    cap, aruco_detector, hand_detector = init_detectors(video_path)
    frame_count = start_frame
    data_dict = {
        'user': user, 'video_name': video_name, 'frame_index': [],
        'hand_landmark':[], 'x_coords':[], 'y_coords':[]
    }

    if not cap.isOpened():
        print('Unable to access camera feed.')
        return

    status = printCapDetails(cap, video_path)
    if not status: return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    success, capture = cap.read()
    if not success: return
    
    # ROI selection must remain interactive if data is missing
    roi_boxes = load_existing_roi(video_name)
    if roi_boxes is None:
        # Temporarily force display for ROI selection
        frame = cv2.rotate(capture.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, [960, 540])
        roi_boxes = []
        i = 0
        while i < 7:
            window_name = f"Select LED for Key {i}"
            box = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
            if box[0] == 0: continue
            roi_boxes.append(box[0:2])
            cv2.destroyWindow(window_name)
            i += 1
        save_new_roi(video_name, start_frame, roi_boxes)
            
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Processing {video_name} (Headless: {HEADLESS_MODE})...")

    while True:
        success, capture = cap.read()
        if not success: break
        
        # Core transformation frame
        frame = cv2.rotate(capture.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, [960, 540])
        
        # [1] Ground Truth Logging (Always runs)
        trackLEDs(video_name, frame_count, frame, roi_boxes)

        # [2] Piano Border / Homography Logic
        if PIANO_BORDER_GLOBAL is None:
            corners, ids, _ = aruco_detector.detectMarkers(frame)
            if corners is not None and ids is not None and len(ids) == 2:
                PIANO_BORDER_GLOBAL = generate_boarder_points(corners)
                distance = get_piano_distance(corners)
                if distance != -1:
                    saveDistanceData(video_name, distance, distance_file_path)

        if PIANO_BORDER_GLOBAL is not None:
            # [3] CRITICAL: apply_homography MUST be called to update global H_matrix
            apply_homography(frame, PIANO_BORDER_GLOBAL)

            # [4] Hand Detection Logic
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            frame_timestamp_ms = int((frame_count / fps) * 1000)
            result = hand_detector.detect_for_video(mp_image, frame_timestamp_ms)

            # [5] Data Logging
            if result.hand_landmarks:
                for hand_landmark in result.hand_landmarks:
                    for i, landmark in enumerate(hand_landmark):
                        h, w, _ = frame.shape
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        # Saves coordinates using the H_matrix generated in step 3
                        appendRecordedLandmarks(data_dict, H_matrix, frame_count, i, x, y)
                        
                        # Only draw if not in headless mode
                        if not HEADLESS_MODE:
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # [6] UI Rendering Toggle
            if not HEADLESS_MODE:
                cv2.imshow('HomePiano (Transformed)', frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    saveRecordedLandmarks(data_dict, file_path)
                    return "QUIT"
            else:
                # In Headless, we check keyboard directly without waitKey latency
                if keyboard.is_pressed('q'):
                    saveRecordedLandmarks(data_dict, file_path)
                    return "QUIT"
        else:
            if not HEADLESS_MODE:
                cv2.putText(frame, "Waiting for ArUco...", (50, 50), 0, 1, (0,0,255), 2)
                cv2.imshow('HomePiano (Transformed)', frame)
                cv2.waitKey(1)

        frame_count += 1

    saveRecordedLandmarks(data_dict, file_path)
    print(f'LEDs lit up: {led_lit}')
    cap.release()
    if not HEADLESS_MODE: cv2.destroyAllWindows()
    return True

BASE_VIDEO_DIR = Path(r'D:\Datasets\Publishing Research_Vid Recordings')
DETECTOR_CSV_PATH = Path(r'Machine-Learning-Evidence\Machine_Learning_Course\Code\Data Collection\frame_detector.csv')
COORDINATE_DATASET_PATH = r'Machine-Learning-Evidence\Machine_Learning_Course\Code\Data Collection\hand_coordinates_dataset.csv'
DISTANCE_DATASET_PATH = Path(r'Machine-Learning-Evidence\Machine_Learning_Course\Code\Data Collection\distance.csv')

if __name__ == "__main__":
    if not DETECTOR_CSV_PATH.exists():
        print(f"Error: {DETECTOR_CSV_PATH} not found. Run FrameDetector first.")
    else:
        # 1. Load the progress from your detector script
        df_detector = pd.read_csv(DETECTOR_CSV_PATH)
        
        # 2. Identify already processed videos from the coordinates dataset
        processed_videos = []
        if Path(COORDINATE_DATASET_PATH).exists():
            df_coords = pd.read_csv(COORDINATE_DATASET_PATH)
            if 'video_name' in df_coords.columns:
                processed_videos = df_coords['video_name'].unique().tolist()
                print(f"Skipping {len(processed_videos)} already processed videos.")

        # 3. Filter for videos that have a valid start frame (> 0)
        videos_to_process = df_detector[df_detector['start_frame'] > 0]
        
        print(f"Found {len(videos_to_process)} videos in the detector queue.\n")

        for index, row in videos_to_process.iterrows():
            filename = row['file_name']
            
            # --- SKIP LOGIC ---
            if filename in processed_videos:
                continue # Skip to the next video
            
            start_f = int(row['start_frame'])
            full_video_path = str(BASE_VIDEO_DIR / filename)

            print(f"\n[BATCH] Processing: {filename}")
            print(f"[BATCH] Seeking to Frame: {start_f}")

            # Reset the global border for each new video to ensure clean detection
            PIANO_BORDER_GLOBAL = None
            
            # Execute the main processing method
            # Extracts the 'user' name from the filename prefix (e.g., 'Cap' from 'Cap_Phase1...')
            user_id = filename.split('_')[0]
            
            status = main(
                user=user_id, 
                video_path=full_video_path, 
                file_path=COORDINATE_DATASET_PATH, 
                start_frame=start_f,
                distance_file_path=DISTANCE_DATASET_PATH
            )

            if not status:
                break
            
        print("\nBatch processing complete.")