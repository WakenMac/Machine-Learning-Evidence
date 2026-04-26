import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import RunningMode, HandLandmarker, HandLandmarkerOptions 
import time

model_path = 'Machine-Learning-Evidence\Machine_Learning_Course\Code\hand_landmarker.task'
cap = mp_hands = mp_drawing = hands = None
detector = None

minimum_quality = [1280, 720]
high_quality = [1920, 1080]

def init():
    global cap, mp_hands, mp_drawing, hands, detector

    cap = cv2.VideoCapture(0) # Use this if the webcam is off
    # cap = cv2.VideoCapture(1) # Use this if Iriun webcam is on
    # cap = cv2.VideoCapture(2) # Use this if the webcam is on

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, minimum_quality[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, minimum_quality[1])

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode = RunningMode.VIDEO,
        num_hands=2,
    )
    detector = HandLandmarker.create_from_options(options)
    print(cap.get(cv2.CAP_PROP_FPS))


def processImage(image):
    # Resizing the Image
    # [h, w] = image.shape[:2]
    # desired_width = 640
    # aspect_ratio = w/float(h)
    # desired_height = int(h / aspect_ratio)
    # resized_image = cv2.resize(image, (desired_width, desired_height))
    resized_image = cv2.resize(image, (960, 540))

    # Passing the image to mediapipe
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    frame_timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    # Draws the points to the image_bgr
    if result.hand_landmarks:
        for hand_landmark in result.hand_landmarks:
            for landmark in hand_landmark:
                h, w, c = image.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Draw a green circle on each joint
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    print(image.shape)
    return image

def main():
    init()
    global cap

    if not cap.isOpened():
        print('Error: Could not access video stream')
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            print('Error: Could not read frame')
        
        frame = processImage(frame)

        cv2.imshow('Droid Cam Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

main()
