# Author: Waken Cean C. Maclang
# Date Last Edited: October 31, 2025
# Course: Machine Learning
# Task: Learning Evidence

# VideoPlayer.py 
#     Class to greate 

# Works with Python 3.10.0

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

    def __init__(self, 
                 vid_path:str = 'Machine_Learning_Course\\Trial Recordings\\[1] ArUco Boarder.mp4',
                 dataset_path:str = None,
                 with_delay:bool = True):
        
        # Prepares the video
        self._prepareVideo(vid_path, with_delay)

        # Prepares the media pipe parts
        self.hand_detector, self.mp_hands, self.mp_drawing = self._init_media_pipe_tools()

        # Prepares to load the dataset
        self.dataset = self._init_dataset(dataset_path)

        # Time variables
        self.frame_count = 1
        self.secs = self.mins = 0

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
        if with_delay:
            self.delay = int(500 / self.fps)
        else:
            self.delay = -1

    def _append_dataset(self, img):
        """

        """

        pass

    def runVideo(self):
        pause = False

        # cv2.putText() related variables
        org = (10, 30)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
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
            float_secs = self.frame_count / 60
            secs = int(float_secs % 60)
            mins = int(float_secs / 60)

            text = f'Frame #{self.frame_count}, Time: {mins:02}:{secs:02}'
            frame = cv2.putText(frame, text, org, fontFace, fontScale, color, 
                                thickness, lineType)
            self.frame_count += 1

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

player = VideoPlayer('Machine_Learning_Course\\Trial Recordings\\[1] ArUco Boarder.mp4', None, True)
player.runVideo()