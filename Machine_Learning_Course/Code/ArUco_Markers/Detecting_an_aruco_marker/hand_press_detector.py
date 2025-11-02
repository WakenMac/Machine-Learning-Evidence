import mediapipe as mp
import numpy as np
from collections import deque
import time
import cv2

class HandPressDetector:
    def __init__(self, debounce_time=0.15, filter_size=5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_press_time = 0
        self.debounce_time = debounce_time
        self.prev_fingertip_z = None
        self.fingertip_history = deque(maxlen=filter_size)

    def get_fingertip_coordinates(self, hand_landmarks, shape):
        fingertip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        h, w, _ = shape
        x, y, z = int(fingertip.x * w), int(fingertip.y * h), fingertip.z

        self.fingertip_history.append((x, y, z))

        avg_x = int(np.mean([pos[0] for pos in self.fingertip_history]))
        avg_y = int(np.mean([pos[1] for pos in self.fingertip_history]))
        avg_z = np.mean([pos[2] for pos in self.fingertip_history])

        return avg_x, avg_y, avg_z

    def detect_press(self, hand_landmarks, shape):
        x, y, z = self.get_fingertip_coordinates(hand_landmarks, shape)
        pressed = False

        palm = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        palm_x, palm_y, palm_z = int(palm.x * shape[1]), int(palm.y * shape[0]), palm.z

        dist = np.sqrt((x - palm_x)**2 + (y - palm_y)**2)

        if self.prev_fingertip_z is not None:
            velocity = z - self.prev_fingertip_z
            if velocity > 0.005 and dist < 100:  # Press heuristic
                current_time = time.time()
                if current_time - self.last_press_time > self.debounce_time:
                    pressed = True
                    self.last_press_time = current_time

        self.prev_fingertip_z = z
        return pressed, (x, y)

    def draw_landmarks(self, image, hand_landmarks, pressed=False):
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )
        if pressed:
            fingertip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = image.shape
            x, y = int(fingertip.x * w), int(fingertip.y * h)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)