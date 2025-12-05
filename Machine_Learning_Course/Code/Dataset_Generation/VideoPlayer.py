# Author: Waken Cean C. Maclang
# Date Last Edited: October 31, 2025
# Course: Machine Learning
# Task: Learning Evidence

# VideoPlayer.py 
#     Class to greate 

# Works with Python 3.10.0

import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import keyboard
from Machine_Learning_Course.Code.Dataset_Generation.QueuedList import QueuedList as ql

class VideoPlayer:
    """
    Author: Waken Cean C. Maclang
    Date Last Edited: October 31, 2025
    Course: Machine Learning
    Task: Learning Evidence

    VideoPlayer.py 
        A class designed to generate the dataset needed for key hover and press detection

        It mainly handles the following:
            [1] Playing video recordings frame per frame
            [2] Detecting Hand placements and movements
            [3] Saving the hand heuristics and movements to the dataset
    """

    # Static Final/Constant Variables
    __VID_PATH = 'Machine_Learning_Course\\Piano_Recordings'
    __VID_LIST = os.listdir(__VID_PATH)

    @classmethod
    def get_vid_list(cls):
        return os.listdir(cls.__VID_PATH)

    def __init__(self, 
                 vid_path:str = 'Machine_Learning_Course\\Trial Recordings\\[1] ArUco Boarder.mp4',
                 dataset_path:str = None,
                 with_delay:bool = True,
                 starting_frame: int = 0,
                 key_distance_cm:float = 0,
                 key_distance_px:int = 0):
        
        # Prepares the video
        self._prepareVideo(vid_path, with_delay)

        # Prepares the media pipe parts
        self.hand_detector, self.mp_hands, self.mp_drawing = self._init_media_pipe_tools()

        # Prepares to load the dataset
        self.dataset = self._init_dataset(dataset_path)

        # Time variables
        self.frame_count = starting_frame if starting_frame >= 0 else 0
        self.elapsed_frames = 0
        self.secs = self.mins = 0

        # Dataset dictionary
        self.temp_data = {
            'video_name': self.file_name,
            'frame': [],
            'fingertip': 'index',
            'tip2dip': [],
            'tip2pip': [],
            'tip2mcp': [],
            'tip2wrist': [],
            'disp':[],
            'velocity_size':[],     # The change in average joint distance (tip2dip ... tip2mcp)
            'velocity_disp':[],     # The distance from the previous point to the new point
            'acceleration_disp':[],
            'distance_cm': [],     # Distance of the camera to the piano (meters)
            'is_hovering':[]
        }

        self.temp_key_hist = []

        # Prepare variables for velocity calculation
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.ql_size = ql(fps)
        self.ql_disp = ql(fps)
        self.old_coor = None
        self.new_coor = None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
        self.KNOWN_DISTANCE = key_distance_cm
        self.KNOWN_PX_WIDTH = key_distance_px

    def _init_media_pipe_tools(self):
        """
        Initializes MediaPipe's tools for hand-coordinate marking
        @returns    A list of MediaPipe tools for hand marking and detection
        """
        return [mp.solutions.hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1
        ), mp.solutions.hands, mp.solutions.drawing_utils]

    def _init_dataset(self, dataset_path):
        """
        Initializes the dataset to be used to store the hand-heuristics data
        @param dataset_path  file_path to the csv dataset
        @returns             A loaded DataFrame or an Empty DataFrame if an invalid path is passed.
        """
        dataset = None
        try:
            dataset = pd.read_csv(dataset_path)
        except Exception:
            print(f'Unable to load dataset through path {dataset_path}. \nMaking a new dataset.')
            dataset = pd.DataFrame(data=None, index=None, 
                                   columns=['tip_to_dip', 'tip_to_mcp', 'tip_to_twist', 'distance_cm', 'velocity'])
        return dataset

    def _prepareVideo(self, vid_path:str = None, with_delay:bool = True):
        """
        Helper method that prepares the variables related to playing the video
        """
        self.vid_path = vid_path
        self.file_name = vid_path.split('\\')[-1]
        self.cap = cv2.VideoCapture(self.vid_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adds delay to the time cv2 renders frames.
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = 15 if self.fps == 25 else int(500 / self.fps) if with_delay else 1

    def print_video_info(self):
        string = f'''
            Now playing video. Details:
            Playing video feed from: {self.vid_path} 
            Frames per sec: {self.fps}
            Total frames: {self.total_frames}
            Delay: {self.delay}
            Distance: {self.calculate_distance_formula(52)}
        '''
        print(string)

    def get_hand_pixel_coordinates(self, hand_landmarks, hand_joint_index: int = 0):
            index_x = hand_landmarks.landmark[hand_joint_index].x
            index_y = hand_landmarks.landmark[hand_joint_index].y
            return [int(index_x * 960), int(index_y * 540)]

    def detect_hands(self, img:None):
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(img_rgb)
        if results.multi_hand_landmarks:
            # MediaPipe is set to only detect one hand.
            hand_landmarks = results.multi_hand_landmarks[0]
            pixel_x, pixel_y = self.get_hand_pixel_coordinates(hand_landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP)
            cv2.circle(img_rgb, (pixel_x, pixel_y), 2, (0, 255, 0), thickness = 3)
            return [hand_landmarks, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)]
        
        return [None, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)]
            # Draws the hand points to the image
            # self.mp_drawing.draw_landmarks(img_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def append_dataset(self, hand_landmarks, is_hovering:bool = True):
        """

        """

        def get_point_distance(pointA, pointB) -> int:
            """
            Gets the PIXEL distance/displacement among two points given their x and y axis
            
            @param pointA   A list of x and y coordinate of the first Point 
            @param pointB   A list of x and y coordinate of the second Point

            @returns        The integer distance between 2 points
            """
            x = int(pointA[0] - pointB[0]) ** 2
            y = int(pointA[1] - pointB[1]) ** 2
            return np.sqrt((x + y))

        # TODO: Implement the 2 types of velovity given the FPS
        if hand_landmarks is None:
            return

        dict_keys = ['tip2dip', 'tip2pip', 'tip2mcp', 'tip2wrist']
        index_index = self.mp_hands.HandLandmark.INDEX_FINGER_TIP # Index of the index finger's tip HAHAHHA
        indices = [
            self.mp_hands.HandLandmark.INDEX_FINGER_DIP, 
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP, 
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,  
            self.mp_hands.HandLandmark.WRIST
        ]

        size = 0
        self.new_coor = self.get_hand_pixel_coordinates(hand_landmarks, index_index)
        for i, index in enumerate(indices):
            dist = get_point_distance(
                self.new_coor,
                self.get_hand_pixel_coordinates(hand_landmarks, index)
            )
            size += dist
            self.temp_data[dict_keys[i]].append(dist)
        self.temp_data['frame'].append(self.frame_count)
        self.temp_data['is_hovering'].append(is_hovering)
        
        accuracy = 0
        disp = 0
        # Adds the calculations to the QueuedList
        if self.elapsed_frames < 1 or self.old_coor is None:
            self.ql_disp.append(0)
        else:
            disp = get_point_distance(self.old_coor, self.new_coor)
            previous_velocity = self.ql_disp.get_mean()
            self.ql_disp.append(disp)
            new_velocity = self.ql_disp.get_mean()
            accuracy = previous_velocity - new_velocity

        # self.ql_size.append(self.temp_data['tip2wrist'][-1])

        # ql_size focuses on the average change in finger length within a second
        # ql_disp is the average change in displacement of the fingertip within a second
        self.ql_size.append(size / 4)
        
        self.temp_data['disp'].append(disp)
        self.temp_data['acceleration_disp'].append(accuracy)
        self.temp_data['velocity_disp'].append(self.ql_disp.get_mean()) # Change in valocity per second
        self.temp_data['velocity_size'].append(self.ql_size.get_mean())
        self.temp_data['distance_cm'].append(self.calculate_distance_formula(number_of_keys=52))

        self.old_coor = self.new_coor

    def calculate_distance_formula(self, number_of_keys:int = 1):
        # numerator = 960 / (number_of_keys * 2.3)
        # print(f'Distance = focal_length / {numerator}')
        
        # Focal length (px) * (object size / pixel width)
        # return 1425 * ((number_of_keys * 2.3) / 960)
        return 930 * ((number_of_keys * 2.3) / 960)

    def runVideo(self, csv_name='my_temp_data.csv'):
        self.print_video_info()
        pause = False
        is_hovering = True

        # cv2.putText() related variables
        org = (10, 30)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (255, 255, 255)  # White
        thickness = 2
        lineType = cv2.LINE_AA

        while True:
            if pause:
                cv2.waitKey(self.delay)
                key = self.get_pressed_key()

                if key:
                    self.temp_key_hist.append(key)
                
                if key == 'space':
                    pause = not pause
                elif key == 'r':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_count = 1
                    pause = not pause
                elif key == 'esc':
                    break
                continue

            success, frame = self.cap.read()
            if not success:
                raise Exception('Invalid path has been passed.')
            
            frame = cv2.resize(frame, (960, 540))

            # Calculate the time
            float_secs = self.frame_count / self.fps
            secs = int(float_secs % 60)
            mins = int(float_secs / 60)

            text = f'Frame #{self.frame_count}, Time: {mins:02}:{secs:02}'
            frame = cv2.putText(frame, text, org, fontFace, fontScale, color, 
                                thickness, lineType)

            # Method to detect the hands and add the coordinates to the dataframe
            (hand_landmarks, frame) = self.detect_hands(frame)
            cv2.imshow(self.file_name, frame)

            cv2.waitKey(self.delay)
            key = self.get_pressed_key()

            if key == 'esc':   # ESC key
                break
            elif key == 'q':
                is_hovering = False
            elif key == 'space':
                pause = not pause
            elif key == 'r':
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 1

            # Create the numerical features from the landmarks and add them to the dataset
            self.append_dataset(hand_landmarks, is_hovering)

            is_hovering = True
            self.frame_count += 1
            self.elapsed_frames += 1

            # A object used to track the keys presses made while collecting the dataset
            self.temp_key_hist.append(key)

        self.cap.release()
        cv2.destroyAllWindows()

        temp_df = pd.DataFrame(self.temp_data)
        # print(temp_df.drop(['frame', 'video_index', 'video_name', 'fingertip', 'is_hovering', 'distance_cm'], axis=1).head(20))      
        print(temp_df.drop(['frame', 'video_index', 'video_name', 'fingertip', 'is_hovering'], axis=1).head(20))      
        df = pd.DataFrame(temp_df)
        print('\n\nNumber of key presses: ', df['is_hovering'][df['is_hovering'] == False].count())

        if self.dataset.size == 0:
            df.to_csv(f'Machine_Learning_Course\Code\Dataset_Generation\{self.file_name.replace(" ", "_")[:-3]}csv', index=False)
        else:
            pd.concat(
                [self.dataset, df], axis=0, ignore_index=True
            ).to_csv(f'Machine_Learning_Course\Code\Dataset_Generation\{self.file_name.replace(" ", "_")[:-3]}csv', index=False)

    def get_pressed_key(self):
        """Return the key that is currently pressed, or None."""
        # keyboard.read_event() blocks, so we won’t use that.
        # Instead we scan common printable keys.
        for key in list("abcdefghijklmnopqrstuvwxyz0123456789") + ['space', 'esc']:
            if keyboard.is_pressed(key):
                return key
        return None

    def get_video_player(video_index:int = 0, dataset_path:str = None, starting_frame:int = 0):
        if video_index < 0 or video_index > len(VideoPlayer.get_vid_list()) - 1:
            print('ERROR: Invalid video index passed.')
            return

        player = VideoPlayer(os.path.join(VideoPlayer._VideoPlayer__VID_PATH, VideoPlayer.get_vid_list()[video_index]), 
                            dataset_path, False, starting_frame)
        player.temp_data['video_index'] = video_index
        return player
