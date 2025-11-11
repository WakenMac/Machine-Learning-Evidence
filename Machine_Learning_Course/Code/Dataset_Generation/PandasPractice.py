
import pandas as pd
import random

temp_dict = {
    'video_index':[],
    'frame': [],
    'xy_distance':[],
    'distance_cm':100
}

for i in range(10):
    temp_dict['frame'].append(i)
    temp_dict['video_index'].append(0)
    temp_dict['xy_distance'].append(random.randint(0, 5))

temp_df = pd.DataFrame(temp_dict)
print(temp_df.head())