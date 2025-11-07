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
import mediapipe as mp

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
        return cls.__VID_LIST

    def __init__(self, 
                 vid_path:str = 'Machine_Learning_Course\\Trial Recordings\\[1] ArUco Boarder.mp4',
                 dataset_path:str = None,
                 with_delay:bool = True,
                 starting_frame: int = 0):
        
        # Prepares the video
        self._prepareVideo(vid_path, with_delay)

        # Prepares the media pipe parts
        self.hand_detector, self.mp_hands, self.mp_drawing = self._init_media_pipe_tools()

        # Prepares to load the dataset
        self.dataset = self._init_dataset(dataset_path)

        # Time variables
        self.frame_count = 1
        self.secs = self.mins = 0

        starting_frame = starting_frame if starting_frame >= 0 else 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

    def _init_media_pipe_tools(self):
        """
        Initializes MediaPipe's tools for hand-coordinate marking
        @returns    A list of MediaPipe tools for hand marking and detection
        """
        return [mp.solutions.hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
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

        # Adds delay to the time cv2 renders frames.
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = 15 if self.fps == 25 else int(500 / self.fps) if with_delay else -1
        
        print(self.fps, self.delay)

    def detect_hands(self, img:None):
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    def _append_dataset(self, img):
        """

        """

        pass

    def runVideo(self):
        pause = False

        # cv2.putText() related variables
        org = (10, 30)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (255, 255, 255)  # White
        thickness = 2
        lineType = cv2.LINE_AA

        while True:
            if pause:
                key = cv2.waitKey(self.delay) & 0xFF
                if key == ord(' '):
                    pause = not pause
                elif key == ord('r'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_count = 1
                    pause = not pause
                elif key == 27:
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
            self.frame_count += 1

            frame = self.detect_hands(frame)
            # append_dataset

            cv2.imshow(self.file_name, frame)

            key = cv2.waitKey(self.delay) & 0xFF

            if key == 27:   # ESC key
                break
            elif key == ord(' '):
                pause = not pause
            elif key == ord('r'):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 1

        self.cap.release()
        cv2.destroyAllWindows()

    def get_video_player(video_index:int = 0, dataset_path:str = None, starting_frame:int = 0):
        if video_index < 0 or video_index > len(VideoPlayer.get_vid_list()) - 1:
            print('ERROR: Invalid video index passed.')
            return

        return VideoPlayer(os.path.join(VideoPlayer._VideoPlayer__VID_PATH, VideoPlayer.get_vid_list()[video_index]), 
                           dataset_path, True, starting_frame)

print(VideoPlayer.get_vid_list())
# player = VideoPlayer.get_video_player(1, None, 240)
player = VideoPlayer.get_video_player(0, None, 0)
player.runVideo()