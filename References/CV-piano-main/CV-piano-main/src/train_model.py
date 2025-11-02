import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import os

UNTOUCH_FOLDER = "src/training_data/untouched"
TOUCH_FOLDER = "src/training_data/touched"
def start_training():
  
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

  
    SEED = 2  


    total_samples = len(os.listdir(TOUCH_FOLDER) + os.listdir(UNTOUCH_FOLDER))
    data = tf.keras.utils.image_dataset_from_directory(
        "src/training_data",
        image_size=(40,40),
        batch_size=total_samples // 5,
        validation_split=0.2,  
        subset="training",  
        seed=SEED  
    )

    
    val_data = tf.keras.utils.image_dataset_from_directory(
        "src/training_data",
        image_size=(40,40),
        batch_size=total_samples // 5,
        validation_split=0.2,
        subset="validation",
        seed=SEED  
    )

    model = Sequential([
        Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(40,40,3)),  
        MaxPooling2D(),
        Conv2D(8, (3,3), padding='same', activation='relu'),  
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),  
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',  
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                  metrics=['accuracy'])
    print(model.summary())

    hist = model.fit(data, epochs=23, validation_data=val_data)  


    model.save('models/touch_detection_model.h5')

