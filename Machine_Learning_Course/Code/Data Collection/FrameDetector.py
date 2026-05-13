import cv2
import keyboard
import time
from pathlib import Path
import pandas as pd

# Configurations
BASE_VIDEO_PATH = Path(r'E:\\Waks - Academics\\Publishing Research_Vid Recordings')
SAVE_DIR = Path(r'Machine-Learning-Evidence\\Machine_Learning_Course\\Code\\Data Collection')
CSV_NAME = "frame_detector.csv"
FULL_CSV_PATH = SAVE_DIR / CSV_NAME

def frame_by_frame_player_v2(full_path, filename):
    cap = cv2.VideoCapture(full_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {filename}.")
        return True # Skip to next video if this one fails

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    paused = False

    ret, frame = cap.read()
    if not ret:
        print(f"Video {filename} is empty.")
        return True
        
    display_frame = frame.copy()

    # --- DEBOUNCE VARIABLES ---
    last_action_time = 0
    cooldown_delay = 0.05 # Slightly higher cooldown for Enter key stability

    while True:
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        annotated_frame = cv2.rotate(display_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        annotated_frame = cv2.resize(annotated_frame, [960, 540])
        
        status = "PAUSED" if paused else "PLAYING"
        text = f"Frame: {current_frame_pos} / {total_frames} | {status}"
        cv2.putText(annotated_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame-by-Frame Player", annotated_frame)
        cv2.waitKey(5)

        current_time = time.time()
        can_press = (current_time - last_action_time) > cooldown_delay

        # 1. QUIT SESSION
        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            cap.release()
            return "QUIT"
            
        # 2. SAVE FRAME & NEXT
        if keyboard.is_pressed('enter') and can_press:
            print(f"SAVED: {filename} at Frame {current_frame_pos}")
            cap.release()
            return current_frame_pos

        # 3. CONTROLS
        if keyboard.is_pressed('r'): 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if keyboard.is_pressed('space') and can_press:
            paused, last_action_time = not paused, current_time
        if (keyboard.is_pressed('right') or keyboard.is_pressed('d')) and can_press:
            paused, last_action_time = True, current_time
            ret, frame = cap.read()
            if ret: 
                display_frame = frame.copy()
        if (keyboard.is_pressed('left') or keyboard.is_pressed('a')) and can_press:
            paused, last_action_time = True, current_time
            new_pos = max(0, current_frame_pos - 2) 
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            if ret: 
                display_frame = frame.copy()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            display_frame = frame.copy()

video_files = [
    # --- CAP ---
    ['Cap_Phase1_60_L.mp4', 102],
    ['Cap_Phase1_60_R.mp4', 183],
    ['Cap_Phase1_60_Extra.mp4', 109],
    ['Cap_Phase1_100_L.mp4', 0],
    ['Cap_Phase1_100_R.mp4', 0],
    ['Cap_Phase1_140_L.mp4', 0],
    ['Cap_Phase1_140_R.mp4', 0],
    ['Cap_Phase2_60_L.mp4', 0],
    ['Cap_Phase2_60_R.mp4', 0],
    ['Cap_Phase2_100_L.mp4', 0],
    ['Cap_Phase2_100_R.mp4', 0],
    ['Cap_Phase2_140_L.mp4', 0],
    ['Cap_Phase2_140_R.mp4', 0],

    # --- EARL ---
    ['Earl_Phase1_60_Both.mp4', 0],
    ['Earl_Phase1_100_Both.mp4', 0],
    ['Earl_Phase1_140_Both.mp4', 0],

    # --- IAN ---
    ['Ian_Phase1_60_Both.mp4', 0],
    ['Ian_Phase1_100_Both.mp4', 0],
    ['Ian_Phase1_140_Both.mp4', 0],
    ['Ian_Phase2_60_L.mp4', 0],
    ['Ian_Phase2_60_R.mp4', 0],
    ['Ian_Phase2_100_Both.mp4', 0],
    ['Ian_Phase2_140_Both.mp4', 0],

    # --- JAMES ---
    ['James_Phase1_60_Both.mp4', 0],
    ['James_Phase1_100_Both.mp4', 0],
    ['James_Phase1_140_Both.mp4', 0],
    ['James_Phase2_140_Both.mp4', 0],
    ['James_Phase5_Right.mp4', 0],

    # --- JULSE ---
    ['Julse_Phase1_60_Both.mp4', 0],
    ['Julse_Phase1_100_Both.mp4', 0],
    ['Julse_Phase1_140_Both.mp4', 0],
    ['Julse_Phase2_60_Both.mp4', 0],

    # --- STELLAR ---
    ['Stellar_Phase1_60_L.mp4', 0],
    ['Stellar_Phase1_60_R.mp4', 0],
    ['Stellar_Phase1_100_L.mp4', 0],
    ['Stellar_Phase1_100_R.mp4', 0],
    ['Stellar_Phase1_140_L.mp4', 0],
    ['Stellar_Phase1_140_R.mp4', 0],
    ['Stellar_Phase2_60_L.mp4', 0],
    ['Stellar_Phase2_60_R.mp4', 0],
    ['Stellar_Phase2_100_L.mp4', 0],
    ['Stellar_Phase2_100_R.mp4', 0],
    ['Stellar_Phase2_140_L.mp4', 0],
    ['Stellar_Phase2_140_R.mp4', 0],

    # --- THEO ---
    ['Theo_Phase1_60_Both.mp4', 0],
    ['Theo_Phase1_100_Both.mp4', 0],
    ['Theo_Phase1_140_Both.mp4', 0],
    ['Theo_Phase3_140_Both.mp4', 0],

    # --- WAKS ---
    ['Waks_Phase1_60_Both.mp4', 0],
    ['Waks_Phase1_100_Both.mp4', 0],
    ['Waks_Phase1_140_Both.mp4', 0],
    ['Waks_Phase2_60_Both.mp4', 0],
    ['Waks_Phase2_100_Both.mp4', 0],
    ['Waks_Phase2_140_Both.mp4', 0]
]

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load existing CSV or create new one
if FULL_CSV_PATH.exists():
    df = pd.read_csv(FULL_CSV_PATH)
    print(f"Loaded existing tracker from {FULL_CSV_PATH}")
else:
    df = pd.DataFrame(video_files, columns=['file_name', 'start_frame'])
    print("Created new tracker database.")

print("\n--- CONTROLS ---")
print("ENTER : Set Frame & Next | SPACE : Pause | Arrows : Step | Q : Save & Exit\n")

try:
    # Iterate through videos where start_frame is 0
    for index, row in df.iterrows():
        if row['start_frame'] != 0:
            continue # Skip already processed videos
        
        filename = row['file_name']
        full_path = str(BASE_VIDEO_PATH / filename)
        
        result = frame_by_frame_player_v2(full_path, filename)
        
        if result == "QUIT":
            break
        elif result == "SKIP":
            continue
        else:
            # Update the dataframe
            df.at[index, 'start_frame'] = result

finally:
    # Always save on exit
    df.to_csv(FULL_CSV_PATH, index=False)
    print(f"\nProgress saved to: {FULL_CSV_PATH}")
    cv2.destroyAllWindows()
