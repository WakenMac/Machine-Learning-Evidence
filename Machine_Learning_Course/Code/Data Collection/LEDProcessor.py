import cv2
import time
import keyboard
import pandas as pd
from pathlib import Path

# --- CONFIGURATIONS ---
BASE_VIDEO_PATH = Path(r"D:\Datasets\Publishing Research_Vid Recordings")
DETECTOR_CSV = Path(r'Machine-Learning-Evidence\Machine_Learning_Course\Code\Data Collection\frame_detector.csv')
ROI_CSV_PATH = Path(r'Machine-Learning-Evidence\Machine_Learning_Course\Code\Data Collection\roi_configs.csv')

# Globals for Mouse Tracking
orig_x = orig_y = 0
global_frame = None

def get_pixel_coords(event, x, y, flags, param):
    global orig_x, orig_y, global_frame
    if event == cv2.EVENT_MOUSEMOVE:
        if global_frame is not None:
            h, w = global_frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                orig_x, orig_y = x, y

def save_roi_to_csv(video_name, frame_no, coords_list, threshold):
    """
    Saves the 7 coordinates using the threshold from the 8th click.
    """
    data_list = []
    # Only iterate through the first 7 entries (the actual LED coordinates)
    for i in range(7):
        x, y = coords_list[i]
        data_list.append({
            'video_name': video_name,
            'frame_no': frame_no,
            'LED_no': i + 1,
            'x': x,
            'y': y,
            'threshold': threshold
        })
    
    df_new = pd.DataFrame(data_list)
    df_new.to_csv(ROI_CSV_PATH, mode='a', header=not ROI_CSV_PATH.exists(), index=False)
    print(f"\n[SUCCESS] Saved 7 LEDs for {video_name} using Shiner Threshold: {threshold}")

def process_video_leds(full_path, video_name, start_frame):
    global global_frame, orig_x, orig_y
    
    cap = cv2.VideoCapture(str(full_path))
    if not cap.isOpened():
        print(f'Video titled {video_name} is not found.')
        return "SKIP"

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    paused = True
    last_action_time = 0
    cooldown = 0.2
    
    coords_list = [] # Will hold 7 pairs of (x, y)
    locked_threshold = None
    
    cv2.namedWindow("LED Processor")
    cv2.setMouseCallback("LED Processor", get_pixel_coords)

    print(f"\n>>> Processing: {video_name}")
    print("STEP 1: Hover and ENTER for LED 1 through 7 (Positions).")
    print("STEP 2: Find a frame where an LED shines. Hover it and ENTER (Threshold).")
    
    ret, frame = cap.read()
    
    # Loop until we have 7 coordinates + 1 threshold trigger
    while len(coords_list) < 8:
        current_time = time.time()
        can_press = (current_time - last_action_time) > cooldown

        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                ret, frame = cap.read()

        if ret and frame is not None:
            # Consistent with FrameProcessor display
            display_img = cv2.rotate(frame.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
            display_img = cv2.resize(display_img, [960, 540])
            global_frame = display_img
            
            b, g, r = global_frame[orig_y, orig_x]
            curr_f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            status = "PAUSED" if paused else "PLAYING"
            
            # UI State Logic
            if len(coords_list) < 7:
                header = f"SELECT COORDINATES: LED {len(coords_list)+1}/7"
                color = (0, 255, 0) # Green
            else:
                header = "SELECT SHINING LED (For Threshold)"
                color = (0, 255, 255) # Yellow
            
            cv2.putText(display_img, f"{header} | {status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_img, f"Frame: {curr_f} | Hover R-Value: {r}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw circles for coordinates already picked
            for i, pt in enumerate(coords_list):
                if i < 7: # Only draw the first 7 as they are the real LED locations
                    cv2.circle(display_img, (pt[0], pt[1]), 4, (0, 0, 255), -1)

            cv2.imshow("LED Processor", display_img)
            cv2.waitKey(1)

        # Keyboard Controls
        if (keyboard.is_pressed('q') or keyboard.is_pressed('esc')):
            cap.release()
            return "QUIT"

        if keyboard.is_pressed('space') and can_press:
            paused = not paused
            last_action_time = current_time

        if (keyboard.is_pressed('right') or keyboard.is_pressed('d')) and can_press:
            paused = True
            ret, frame = cap.read()
            last_action_time = current_time
                
        if (keyboard.is_pressed('left') or keyboard.is_pressed('a')) and can_press:
            paused = True
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr_pos - 2))
            ret, frame = cap.read()
            last_action_time = current_time

        # ENTER Logic
        if keyboard.is_pressed('enter') and can_press:
            if len(coords_list) < 7:
                # Add the LED coordinate
                coords_list.append((orig_x, orig_y))
                print(f"Set Position {len(coords_list)}: ({orig_x}, {orig_y})")
            else:
                # This is the 8th click: Capture the shining threshold
                _, _, locked_threshold = global_frame[orig_y, orig_x]
                coords_list.append((orig_x, orig_y)) # Fill the 8th slot to break loop
                print(f"Captured Threshold from Shiner: {locked_threshold}")
            
            last_action_time = current_time

    # Final Save
    save_roi_to_csv(video_name, start_frame, coords_list, locked_threshold)
    cap.release()
    cv2.destroyWindow("LED Processor")
    return "NEXT"

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not DETECTOR_CSV.exists():
        print("Error: frame_detector.csv not found.")
    else:
        df_detector = pd.read_csv(DETECTOR_CSV)
        queue = df_detector[df_detector['start_frame'] > 0]
        
        existing_vids = []
        if ROI_CSV_PATH.exists():
            existing_vids = pd.read_csv(ROI_CSV_PATH)['video_name'].unique().tolist()

        for _, row in queue.iterrows():
            v_name = row['file_name']
            if v_name in existing_vids:
                continue
                
            v_path = BASE_VIDEO_PATH / v_name
            status = process_video_leds(v_path, v_name, row['start_frame'])
            
            if status == "QUIT":
                break

    cv2.destroyAllWindows()