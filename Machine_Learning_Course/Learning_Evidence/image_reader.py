# Author: Maclang, Waken Cean C.
# Date: September 15, 2025

# Enables you to read an image of a hand and returns an annotated version

import cv2
import mediapipe as mp

# Define file paths
input_path = "C:\\Users\\Waks\\Downloads\\75bd947c-1273-44c7-adf4-9315646a2065.jpg"
output_path = "C:\\Users\\Waks\\Downloads\\USEP BSCS\\Coding\\Machine-Learning-in-Java-main\\Machine-Learning-in-Java\\Machine_Learning_Course\\Learning_Evidence\\annotated_image2.jpg"

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,  # Use static_image_mode for single-image processing
    min_detection_confidence=0.5,
    max_num_hands=2)

# Read the image using OpenCV
image = cv2.imread(input_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image from {input_path}")
    exit()

# Get the original dimensions
(h, w) = image.shape[:2]

# Define a desired width for the resized image.
# A smaller width will make processing faster.
desired_width = 640
aspect_ratio = w / float(h)
desired_height = int(desired_width / aspect_ratio)

# Resize the image while maintaining the aspect ratio
resized_image = cv2.resize(image, (desired_width, desired_height))

# Convert the BGR image to RGB for MediaPipe processing
image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# To improve performance, optionally mark the image as not writeable
image_rgb.flags.writeable = False

# Process the image and detect hands
results = hands.process(image_rgb)

# Mark the image back as writable to allow drawing
image_rgb.flags.writeable = True

# Convert the image back to BGR for drawing with OpenCV
annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# Draw the hand annotations on the image
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

# Save the annotated image to the specified output path
print(annotated_image.shape[:2])
# annotated_image = cv2.resize(annotated_image, (w, h))
cv2.imwrite(output_path, annotated_image)

print(f"Annotated image saved to: {output_path}")

# Optional: Display the annotated image for a few seconds
cv2.imshow("Annotated Hand Image", annotated_image)
cv2.waitKey(5000) # Displays the image for 5 seconds
cv2.destroyAllWindows()