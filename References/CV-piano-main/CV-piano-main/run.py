import os
import time
from src import fetch_data, train_model, piano

WEB_CAM = 0
INDEX_FINGER = 8
MIDDLE_FINGER = 12
RING_FINGER = 16
PINKY_FINGER = 20
FINGER = [INDEX_FINGER]
UNTOUCH_FOLDER = "src/training_data/untouched"
TOUCH_FOLDER = "src/training_data/touched"

def fetch_train_data():
    print(
        "The First window is try window so that you can adjust the finger position, adjust hand position so that box will cover  finger tip and finger tip\n1. Window after try will be touch train window\n\t do not lift any finger, move fingers slowly on the paper to get all angles\n2. After this window untouch train window will pop up\n\t lift fingers so that it can take pics of finger tips for training\n\t Then model will be trained and you should see the prediction window for Index finger")
    print("Press Y to move for training stage")
    while 1:
        key = input(">> ")
        if key.lower() == 'y':
            break
    time.sleep(2)
    fetch_data.Try(FINGER)
    time.sleep(2)
    fetch_data.Capture(TOUCH_FOLDER, "touched", FINGER)
    time.sleep(2)
    fetch_data.Capture(UNTOUCH_FOLDER, "untouched", FINGER)

    train_model.start_training()

    print("Model Training Complete")
    time.sleep(3)

run = True

while run:
    model_list = os.listdir("models")
    if "touch_detection_model.h5" not in model_list:
        print("We need to train model on your finger's data")
        fetch_data.clear_training_data()
        fetch_train_data()

    else:

        print("-------------*MENU*-------------\n[1] Retrain model\n[2] Start Paper Piano\n[3] Exit")
        check = True
        while check:
            opt = int(input())
            if opt == 1:
                check = False
                fetch_data.clear_training_data()
                fetch_train_data()
            elif opt == 2:
                check = False
                print("Adjust paper accordingly until you see mesh of keys and press 'q'")
                time.sleep(3)
                piano.start_piano(INDEX_FINGER)
            elif opt == 3:
                check = False
                run = False

