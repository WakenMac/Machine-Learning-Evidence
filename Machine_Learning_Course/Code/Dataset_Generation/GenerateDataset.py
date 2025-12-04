# Author: Waken Cean C. Maclang
# Date Last Edited: October 31, 2025
# Course: Machine Learning
# Task: Learning Evidence

# GenerateDataset.py 
#     Program used to manage the videos to be played and appended to the dataset.

# Works with Python 3.10.0

# I need to learn how to use Python Modules :'>
from Machine_Learning_Course.Code.Dataset_Generation.VideoPlayer import VideoPlayer as vp

dataset_path = 'Machine_Learning_Course\\Code\\Dataset_Generation\\my_temp_data.csv'

# player = vp('Machine_Learning_Course\\Trial Recordings\\[1] ArUco Boarder.mp4')

print(vp.get_vid_list())
# player = vp.get_video_player(2, dataset_path, 240)
# player = vp.get_video_player(1, dataset_path, 110)
player = vp.get_video_player(0, dataset_path, 100)
player.runVideo()