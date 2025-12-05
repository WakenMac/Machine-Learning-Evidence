import pandas as pd
import numpy as np
from plotnine import ggplot
from plotnine import aes, geom_histogram, labs

# Dataset Annotation
temp_df = pd.concat(
    [
    pd.read_csv('Machine_Learning_Course\\Code\\Dataset_Generation\\Debussy_-_Rêverie.csv'),
    pd.read_csv('Machine_Learning_Course\\Code\\Dataset_Generation\\Debussy_-_Clair_de_Lune.csv'),
    pd.read_csv('Machine_Learning_Course\\Code\\Dataset_Generation\\Chopin_-_Marche_Funèbre_(Funeral_March).csv'),
    pd.read_csv('Machine_Learning_Course\\Code\\Dataset_Generation\\Ludovico_Einaudi_-_Divenire.csv')
    ], axis=0, ignore_index=True
)

temp_df['prev_is_hovering'] = temp_df['is_hovering'].shift(1)
temp_df['prev_is_hovering'] = temp_df['prev_is_hovering'].fillna(True)

conditions = [
    # 1. HOLD: Prev=False (Pressed/Held), Current=False (Pressed/Held)
    (temp_df['prev_is_hovering'] == False) & (temp_df['is_hovering'] == False),
    
    # 2. PRESS: Prev=True (Hovering/Released), Current=False (Pressed/Held)
    (temp_df['prev_is_hovering'] == True) & (temp_df['is_hovering'] == False),
    
    # 3. RELEASE: Prev=False (Pressed/Held), Current=True (Hovering/Released)
    (temp_df['prev_is_hovering'] == False) & (temp_df['is_hovering'] == True),
    
    # 4. HOVER: Prev=True (Hovering/Released), Current=True (Hovering/Released)
    (temp_df['prev_is_hovering'] == True) & (temp_df['is_hovering'] == True)
]

actions = ['hold', 'press', 'release', 'hover']

temp_df['action'] = np.select(conditions, actions, default='undefined')
temp_df = temp_df[['video_name', 'video_index', 'frame', 'fingertip', 'tip2dip', 'tip2pip', 'tip2mcp', 'tip2wrist', 'disp',
       'velocity_size', 'velocity_disp', 'acceleration_disp', 'distance_cm', 'is_hovering', 'prev_is_hovering', 'action']]
temp_df.to_csv('Machine_Learning_Course\\Code\\Dataset_Generation\\main_dataset.csv', index=False)

# End of dataset concatenation

df = pd.read_csv('Machine_Learning_Course\\Code\\Dataset_Generation\\Debussy_-_Rêverie.csv')
df[df['is_hovering'] == True].describe()
df[df['is_hovering'] == False].describe()

plots = []
for col in temp_df.columns[3:-3]:
    plot = (
        ggplot(data=temp_df)
        + geom_histogram(
            aes(x=col, fill='action'),
            binwidth=5
        )
        + labs(
            title=f'{col} representation',
            x='Pixel Distance',
            y='Frequency'
        )
    )
    plots.append(plot)
plots[0]
