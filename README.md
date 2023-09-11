# Chest Cancer Detection - Documentation

## Project Overview

- Project Name: Chest Cancer Detection
- Dataset: [Chest CTScan Images on Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- Model Development Code: [Kaggle Notebook](https://www.kaggle.com/code/mrappplg/chest-cancer-detection/notebook)

## Team Members

- Atharva Pawar
- Aditya Vyas
- Harsh Trivedi

## Project Description

This project, "Chest Cancer Detection," is a mini-project in the field of data science applied to healthcare. The objective is to develop a deep learning model that can classify chest CT scan images into four categories: Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, and Normal. The project aims to assist in early cancer detection and improve healthcare diagnostics.

## Model Development and Training

### Model Architecture

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))  # Four classes

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Model Training

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=121,  # You can adjust the number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

```

# Deployment
The trained model is deployed on a website accessible at https://dshc-chest-cancer-detection.atharvapawar.repl.co/. Users can upload chest CT scan images, and the website provides predictions for the uploaded images.

# Conclusion
The Chest Cancer Detection project demonstrates the application of deep learning in healthcare for early cancer detection. The model is capable of classifying chest CT scan images into four distinct categories, providing valuable diagnostic assistance to medical professionals.

For more details, refer to the Kaggle Notebook containing the code used in model development.

