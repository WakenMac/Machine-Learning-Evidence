import cv2
import time
import keyboard
import pandas as pd
from pathlib import Path

orig_x = orig_y = 0
r_min = r_max = 0
global_frame = None

# 1. Define the function that handles mouse events
def get_pixel_coords(event, x, y, flags, param):
    global orig_x, orig_y, global_frame
    if event == cv2.EVENT_MOUSEMOVE and (orig_x != x or orig_y != y):
        orig_x = x
        orig_y = y
        # 'x' and 'y' are the exact pixel coordinates
        print(f"Mouse Position: X={x}, Y={y}")


# 2. Setup your video/image display
BASE_VIDEO_PATH = Path(r'E:\\Waks - Academics\\Publishing Research_Vid Recordings')
CSV_PATH = Path(r'Machine-Learning-Evidence\\Machine_Learning_Course\\Code\\Data Collection\\frame_detector.csv')

vid_list = pd.read_csv(CSV_PATH)
current_frame_pos = vid_list['start_frame'][0]
cap = cv2.VideoCapture(BASE_VIDEO_PATH / vid_list['file_name'][0]) # or your video path
cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)

ret, frame = cap.read()
last_action_time = 0
cooldown_delay = 0.05 # Slightly higher cooldown for Enter key stability
paused = False

cv2.namedWindow("Mouse Tracker")
cv2.setMouseCallback("Mouse Tracker", get_pixel_coords)

if not ret: 
    print('Unable to access video')

else: 
    while True:
        current_time = time.time()
        can_press = (current_time - last_action_time) > cooldown_delay

        # 1. KEYBOARD INPUTS
        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            break
            
        if keyboard.is_pressed('space') and can_press:
            paused = not paused
            last_action_time = current_time

        # Step Forward: Right Arrow or 'D'
        if (keyboard.is_pressed('right') or keyboard.is_pressed('d')) and can_press:
            paused = True
            ret, frame = cap.read()
            last_action_time = current_time
                    
        # Step Backward: Left Arrow or 'A'
        if (keyboard.is_pressed('left') or keyboard.is_pressed('a')) and can_press:
            paused = True
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 2))
            ret, frame = cap.read()
            last_action_time = current_time

        # 2. THE PAUSE GIMMICK
        # Only read a new frame if the video is NOT paused
        if not paused:
            ret, frame = cap.read()
            if not ret: # Loop video if it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                        
        if ret and frame is not None:
            # Pre-process image for display
            display_img = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            display_img = cv2.resize(display_img, [960, 540])
            global_frame = display_img # Update the global frame for the mouse callback

            # Draw the BGR info on the screen for easier reading
            # Note: orig_y and orig_x come from the mouse callback
            b, g, r = global_frame[orig_y, orig_x]
            
            status = "PAUSED" if paused else "PLAYING"
            cv2.putText(display_img, f"Status: {status} | BGR: ({b},{g},{r})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Mouse Tracker", display_img)
        
        # 4. Mandatory waitKey to allow OpenCV to process window events
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()