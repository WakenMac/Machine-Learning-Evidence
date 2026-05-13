import cv2
import time
import keyboard
from pathlib import Path

# --- CONFIGURATION (Paste your details here) ---
BASE_VIDEO_FOLDER = Path(r"D:\Datasets\Publishing Research_Vid Recordings")
VIDEO_NAME = "Cap_Phase1_60_L.mp4"  # [5] Paste your video name here
START_FRAME = 102                   # [4] Set your starting frame here

# Globals for Mouse Tracking
orig_x = orig_y = 0
global_frame = None

def get_pixel_coords(event, x, y, flags, param):
    """[2] Mouse movement detector"""
    global orig_x, orig_y, global_frame
    if event == cv2.EVENT_MOUSEMOVE:
        if global_frame is not None:
            h, w = global_frame.shape[:2]
            # Safety check to keep coordinates within the resized frame
            if 0 <= x < w and 0 <= y < h:
                orig_x, orig_y = x, y

def run_pixel_analyzer():
    global global_frame, orig_x, orig_y
    
    full_path = BASE_VIDEO_FOLDER / VIDEO_NAME
    cap = cv2.VideoCapture(str(full_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not find or open {full_path}")
        return

    # [4] Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    
    paused = True
    last_action_time = 0
    cooldown = 0.15
    
    cv2.namedWindow("Pixel Analyzer")
    cv2.setMouseCallback("Pixel Analyzer", get_pixel_coords)

    print(f"🎬 Analyzing: {VIDEO_NAME}")
    print("--- KEYBINDS ---")
    print("SPACE : Play/Pause")
    print("RIGHT/D : Next Frame | LEFT/A : Prev Frame")
    print("Q/ESC : Quit")

    while True:
        current_time = time.time()
        can_press = (current_time - last_action_time) > cooldown

        # Handle Playback Logic
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
                ret, frame = cap.read()
        else:
            # When paused, we keep the last read frame
            ret = True

        if ret and 'frame' in locals():
            # Standard Pre-processing (Rotation/Resize)
            display_img = cv2.rotate(frame.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
            display_img = cv2.resize(display_img, [960, 540])
            global_frame = display_img
            
            # [3] Detection of 'r' value based on mouse location
            b, g, r = global_frame[orig_y, orig_x]
            
            # UI Overlays
            curr_f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            status = "PAUSED" if paused else "PLAYING"
            
            cv2.putText(display_img, f"{VIDEO_NAME} | {status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"Frame: {curr_f} | X: {orig_x} Y: {orig_y} | R-VALUE: {r}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Pixel Analyzer", display_img)
            cv2.waitKey(1)

        # [1] The Keybinds Gimmick
        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            break

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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pixel_analyzer()