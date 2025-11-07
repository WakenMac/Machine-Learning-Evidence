import cv2
import mediapipe as mp
import os

WEB_CAM = 0
UNTOUCH_FOLDER = "src/training_data/untouched"
TOUCH_FOLDER = "src/training_data/touched"
# CAPTURE TOUCHED n UNTOUCHED
# track finger
# save images

def Capture(save_folder, finger_state, finger, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    print('--------------------------------------------------------------------Press "q" to continue.')
    cap = cv2.VideoCapture(WEB_CAM)
    iter = 0
    with mp_hands.Hands(min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Process image once for color conversion
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for finger_tip_id in finger:
                        finger_tip = hand_landmarks.landmark[finger_tip_id]
                        tip_x, tip_y = int(finger_tip.x * image.shape[1]), int(finger_tip.y * image.shape[0])
                        box_size = int(40 / 2)
                        cv2.rectangle(image, (tip_x - box_size, tip_y - box_size),
                                      (tip_x + box_size, tip_y + box_size), (0, 255, 0), 2)

                        # Crop and save ROI only if it's valid
                        roi = image_rgb[tip_y - box_size:tip_y + box_size, tip_x - box_size:tip_x + box_size]
                        if roi.size > 0:
                            filename = os.path.join(save_folder, f'finger-{finger_state}{iter}.png')
                            cv2.imwrite(filename, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                            print(f'Saved image: {filename}')
                            iter += 1
                            if iter >= 150:
                                cap.release()
                                cv2.destroyAllWindows()
                                return

            cv2.imshow(f'{finger_state} SAVING', image)
            if cv2.waitKey(5) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()




def clear_training_data():
    for file in os.listdir(TOUCH_FOLDER):
        os.remove(f'{TOUCH_FOLDER}/{file}')

    for file in os.listdir(UNTOUCH_FOLDER):
        os.remove(f'{UNTOUCH_FOLDER}/{file}')
    print("TRAINING DATA CLEARED")

def delete_model():
    model = os.listdir("models")
    if "touch_detection_model.h5" in model:
        model.remove("touch_detection_model.h5")
        print("model removed")
    else:
        print("model not present")

def Try(finger):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(WEB_CAM)
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for finger_tip_id in finger:  # Landmark IDs for all five fingers' tips
                        finger_tip = hand_landmarks.landmark[finger_tip_id]
                        height, width, _ = image.shape
                        tip_x, tip_y, tip_z = int(finger_tip.x * width), int(finger_tip.y * height), finger_tip.z

                        box_size = int(40 // 2)  # Adjust the size of the box as needed
                        box_color = (0, 255, 0)  # Green color

                        # Coordinates of the rectangle
                        x1, y1 = tip_x - box_size, tip_y - box_size
                        x2, y2 = tip_x + box_size, tip_y + box_size

                        # Draw a square box around the finger tip
                        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            cv2.imshow('Tocuh tracking', image)
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



