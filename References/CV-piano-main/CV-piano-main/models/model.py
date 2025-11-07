import cv2
from tensorflow.keras import models
import numpy as np
import sys
sys.path.append("src")
import os

TRACKING_BOX_RESOLUTION = (40 , 40)

model_list = os.listdir("models")   

def Predict(img):
  model = models.load_model("models/touch_detection_model.h5")
  resized_img = cv2.resize(img, TRACKING_BOX_RESOLUTION)
  # Add the batch dimension and normalize pixel values
  data = np.expand_dims(resized_img/255, axis=0)
  # Make the prediction
  prediction = model.predict(data)
  print(prediction)
  if prediction > 0.502:
      return 0
  else:
      return 1
