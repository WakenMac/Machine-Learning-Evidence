import cv2
import mediapipe as mp
import numpy as np

# --- 1. Camera Calibration Parameters (Placeholder values) ---
# You MUST replace these with the values from your own camera calibration
# Focal length (fx, fy) and optical center (cx, cy)
camera_matrix = np.array([
    [600, 0, 320],
    [0, 600, 240],
    [0, 0, 1]
], dtype="double")
dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion for this example

# --- 2. 3D Hand Model (in a real-world unit, e.g., meters) ---
# These are the proportional 3D coordinates of key hand landmarks.
# This is a simplified model of a hand at rest. The origin (0,0,0) is at the wrist.
hand_model_3d = np.array([
    (0, 0, 0),         # 0. Wrist
    (0, -0.05, 0),     # 1. Thumb CMC
    (0, -0.08, -0.01), # 2. Thumb MCP
    (0, -0.1, -0.02),  # 3. Thumb IP
    (0, -0.11, -0.03), # 4. Thumb Tip
    (-0.02, -0.08, 0), # 5. Index Finger MCP
    (-0.03, -0.12, 0), # 6. Index Finger PIP
    (-0.04, -0.16, 0), # 7. Index Finger DIP
    (-0.05, -0.18, 0), # 8. Index Finger Tip
    (0.02, -0.08, 0),  # 9. Middle Finger MCP
    (0.03, -0.12, 0),  # 10. Middle Finger PIP
    (0.04, -0.16, 0),  # 11. Middle Finger DIP
    (0.05, -0.18, 0),  # 12. Middle Finger Tip
    (0.04, -0.08, 0),  # 13. Ring Finger MCP
    (0.05, -0.12, 0),  # 14. Ring Finger PIP
    (0.06, -0.16, 0),  # 15. Ring Finger DIP
    (0.07, -0.18, 0),  # 16. Ring Finger Tip
    (0.06, -0.07, 0),  # 17. Pinky Finger MCP
    (0.07, -0.10, 0),  # 18. Pinky Finger PIP
    (0.08, -0.14, 0),  # 19. Pinky Finger DIP
    (0.09, -0.16, 0)   # 20. Pinky Finger Tip
], dtype="double")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Create a list of 2D points from the MediaPipe landmarks
            image_points = []
            for landmark in hand_landmarks.landmark:
                height, width, _ = frame.shape
                pixel_x = int(landmark.x * width)
                pixel_y = int(landmark.y * height)
                image_points.append([pixel_x, pixel_y])
            image_points = np.array(image_points, dtype="double")
            
            # --- 3. Use PnP to estimate the 3D pose of the hand ---
            # rvecs: rotation vectors, tvecs: translation vectors
            success, rvecs, tvecs = cv2.solvePnP(
                hand_model_3d, 
                image_points, 
                camera_matrix, 
                dist_coeffs
            )

            if success:
                # The distance is the magnitude of the z-component of the translation vector
                distance_cm = tvecs[2][0] * 100 # Convert meters to cm
                cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Optionally, re-project the 3D model points to 2D for visualization
                # This helps to see if the PnP estimation is correct
                reprojected_points, _ = cv2.projectPoints(hand_model_3d, rvecs, tvecs, camera_matrix, dist_coeffs)
                reprojected_points = np.int32(reprojected_points).reshape(-1, 2)
                for point in reprojected_points:
                    cv2.circle(frame, tuple(point), 5, (255, 0, 0), -1)

    cv2.imshow("Hand 3D Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
