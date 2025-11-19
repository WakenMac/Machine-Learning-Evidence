# Kinematic Feature Calculation in `PreparePianoMotionDataset.py`

This document provides a detailed explanation of how kinematic features are calculated from the raw 3D hand tracking data in the `PreparePianoMotionDataset.py` script. The primary logic for these calculations is located in the `extract_features` method of the `PianoMotionDataProcessor` class.

## Core Logic

The `extract_features` method processes one frame of kinematic data at a time. For each frame, it calculates a set of features that describe the motion and posture of the hand. These features are then used to train a machine learning model to detect key presses.

### Key Joints

The feature extraction process relies on the 3D coordinates of three key joints from the hand skeleton:

*   **Index Fingertip:** Joint index `8`
*   **Index Finger DIP (Distal Interphalangeal) Joint:** Joint index `6`
*   **Wrist:** Joint index `0`

These indices correspond to the standard MediaPipe/MANO hand model.

## Feature Breakdown

Here is a detailed breakdown of each kinematic feature calculated by the script:

### 1. Finger Position

*   **Features:** `finger_position_x`, `finger_position_y`, `finger_position_z`
*   **Description:** These features represent the raw 3D coordinates of the index fingertip in the current frame.
*   **Code Snippet:**
    ```python
    fingertip = current_frame[fingertip_idx]
    features['finger_position_x'] = float(fingertip[0])
    features['finger_position_y'] = float(fingertip[1])
    features['finger_position_z'] = float(fingertip[2])
    ```

### 2. Finger Velocity

*   **Feature:** `finger_velocity`
*   **Description:** This feature measures the speed of the index fingertip. It is calculated as the Euclidean distance between the fingertip's position in the current and previous frames, divided by the duration of a single frame.
*   **Formula:** `velocity = || fingertip_current - fingertip_previous || / frame_duration`
*   **Code Snippet:**
    ```python
    if frame_idx > 0:
        prev_frame = kinematics[frame_idx - 1]
        prev_fingertip = prev_frame[fingertip_idx]
        velocity = np.linalg.norm(fingertip - prev_fingertip) / self.frame_duration
        features['finger_velocity'] = float(velocity)
    else:
        features['finger_velocity'] = 0.0
    ```

### 3. Finger Acceleration

*   **Feature:** `finger_acceleration`
*   **Description:** This feature measures the rate of change of the fingertip's velocity. It is calculated as the difference between the velocity in the current and previous frames, divided by the frame duration.
*   **Formula:** `acceleration = (velocity_current - velocity_previous) / frame_duration`
*   **Code Snippet:**
    ```python
    if frame_idx > 1:
        prev_frame = kinematics[frame_idx - 1]
        prev_prev_frame = kinematics[frame_idx - 2]

        prev_fingertip = prev_frame[fingertip_idx]
        prev_prev_fingertip = prev_prev_frame[fingertip_idx]

        velocity_current = np.linalg.norm(fingertip - prev_fingertip) / self.frame_duration
        velocity_prev = np.linalg.norm(prev_fingertip - prev_prev_fingertip) / self.frame_duration

        acceleration = (velocity_current - velocity_prev) / self.frame_duration
        features['finger_acceleration'] = float(acceleration)
    else:
        features['finger_acceleration'] = 0.0
    ```

### 4. Posture Feature

*   **Feature:** `posture_feature`
*   **Description:** This feature captures the curvature of the index finger by measuring the Euclidean distance between the fingertip and the DIP joint. A smaller distance indicates a more curved finger.
*   **Formula:** `posture = || fingertip - dip_joint ||`
*   **Code Snippet:**
    ```python
    posture = np.linalg.norm(fingertip - dip_joint)
    features['posture_feature'] = float(posture)
    ```

### 5. Euclidean Distance

*   **Feature:** `euclidean_distance`
*   **Description:** This feature is identical to the `posture_feature`. It is included in the feature set under a different name.
*   **Code Snippet:**
    ```python
    features['euclidean_distance'] = float(posture)
    ```

### 6. Distance From Wrist

*   **Feature:** `distance_from_wrist`
*   **Description:** This feature provides context about the fingertip's position relative to the hand's base. It is the Euclidean distance between the fingertip and the wrist.
*   **Formula:** `distance = || fingertip - wrist ||`
*   **Code Snippet:**
    ```python
    distance_from_wrist = np.linalg.norm(fingertip - wrist)
    features['distance_from_wrist'] = float(distance_from_wrist)
    ```

### 7. Depth Feature

*   **Feature:** `depth_feature`
*   **Description:** This feature is the z-coordinate of the fingertip. It is a critical feature for distinguishing between a key press and a hover, as a lower z-value typically indicates that the finger is closer to the piano keys.
*   **Code Snippet:**
    ```python
    features['depth_feature'] = float(fingertip[2])
    ```
