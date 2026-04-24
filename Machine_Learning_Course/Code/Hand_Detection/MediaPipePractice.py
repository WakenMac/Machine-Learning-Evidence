# Use this code: py -3.10 Machine_Learning_Course\Learning_Evidence\media_pipe_practice.py

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

def screenshot(image):
    """Making a screenshot of the image"""
    output_path = 'C:\\Users\\Waks\\Downloads\\USEP BSCS\\Coding\\Machine-Learning-in-Java-main\\Machine-Learning-in-Java\\Machine_Learning_Course\\Learning_Evidence\\annotated_image.jpg'
    cv2.imwrite(output_path, image)

# Initialize MediaPipe Hands solution
# max_num_hands=1 ensures that only one hand is detected
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Indexes of the fingers to be used later
indices = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
           mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

# Use the 'with' statement for proper resource management
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:

    # Start the video capture from the webcam (device 0)
    cap = cv2.VideoCapture(1)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while cap.isOpened():
        # Read a frame from the webcam
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB, which is what MediaPipe expects
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable
        image.flags.writeable = False

        # Process the image and detect hands
        # MediaPipe accepts RGB images to draw on
        results = hands.process(image)

        # Method to turn the read data into a dataframe
        path = 'C:\\Users\\Waks\\Downloads\\USEP BSCS\\Coding\\Machine-Learning-in-Java-main\\Machine-Learning-in-Java\\Machine_Learning_Course\\Learning_Evidence\\temp_coords.csv'
        coords = []
        temp_coord = []
        # if results.multi_hand_landmarks:
        #     hand_marks = results.multi_hand_landmarks[0] # Gets the positions of the first hand
            
        #     for index in iter(indices):
        #         temp_coord.append(hand_marks.landmark[index].z)

        #     array = np.array(temp_coord).reshape(1, -1)
        #     coords.append(temp_coord)
        #     temp_coord = []
            
        #     df = pd.DataFrame(coords)
        #     df.to_csv(path)
        #     print('Hand Coordinates saved')
        #  break

        # for index, landmark in hand_marks.landmarks:
        #     temp_coord.append(landmark.x)
        #     temp_coord.append(landmark.y)
        #     temp_coord.append(landmark.z)
        #     print(f'Index {index}: {landmark.x}')
        #     print(f'Index {index}: {landmark.y}')
        #     print(f'Index {index}: {landmark.z}')

        # Mark the image back as writeable
        image.flags.writeable = True

        # Convert the image back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the hand annotations on the image
        if results.multi_hand_landmarks:

            # Coordinates will still be printed out regardless if it is seen on the video or not so long as the coordinates are seen
            # Let z be the distance of how close or far a point is from the camera
            # The lower the z value (the more negative) = closer
            # The higher the z value (the more positive) = farther
            hand_mark = results.multi_hand_landmarks[0]
            # print(hand_mark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z)

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand connections (skeleton) and landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                a = float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x) ** 2
                b = float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y) ** 2
                print(np.sqrt(a + b))

        # screenshot(image)
        # break
    
        # Display the processed image
        cv2.imshow('MediaPipe Hand Detection', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()