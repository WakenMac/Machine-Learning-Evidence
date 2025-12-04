import cv2
import mediapipe as mp

cap = mp_hands = mp_drawing = hands = None

def init():
    global cap, mp_hands, mp_drawing, hands

    # cap = cv2.VideoCapture(0) # Use this if the webcam is off
    # cap = cv2.VideoCapture(0) # Use this for the Iriun Webcam
    cap = cv2.VideoCapture(2) # Use this if the webcam is on
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode = False,
        min_detection_confidence = 0.5,
        max_num_hands = 2
    )

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
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Draws the points to the image_bgr
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmark,
                mp_hands.HAND_CONNECTIONS
            )
            print(
                hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,
                sep=' | '
            )
        return image_bgr
    else: 
        return resized_image

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
