# Custom Gesture Recognition Model

This repository hosts a custom-built deep learning model for recognizing hand gestures, created entirely from scratch. The model is trained to recognize gestures such as "thumbs up," "thumbs down," "fist," and "five," and can be utilized for gesture-based controls in interactive systems.

## Model Overview

- **Model Type**: Custom-built gesture recognition model
- **Framework**: Keras (saved in .h5 format)
- **Purpose**: Detects and classifies hand gestures for touchless interfaces and user-interactive applications

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Specifications](#model-specifications)
5. [Development Roadmap](#development-roadmap)

## Features

- **Custom Model Architecture**: Developed from scratch to recognize specific hand gestures with high accuracy.
- **Gesture Classification**: Distinguishes between gestures such as thumbs up, thumbs down, fist, and five.
- **Real-Time Application Ready**: Built for real-time applications, enabling smooth gesture-based interactions.

## Installation

Clone this repository:
    ```bash
    git clone https://github.com/SinethB/gesture-recognition-model.git
    cd gesture-recognition-model
    ```

## Usage

### Loading the Model

```python
from tensorflow.keras.models import load_model

# Load the custom gesture recognition model
model = load_model('gesture_recognition_model.h5')
```
### Making Predictions
The model requires images to be pre-processed to the correct size and format. Here’s an example usage:
```python
import cv2
import numpy as np

# Load and pre-process the image
image = cv2.imread('path_to_image.jpg')
image = cv2.resize(image, (64, 64))  # Adjust as per model input size
image = np.expand_dims(image, axis=0)

# Predict gesture
prediction = model.predict(image)
predicted_class = np.argmax(prediction)
print("Predicted Gesture:", predicted_class)
```
## Model Specifications
- **File**: gesture_recognition_model.h5
- **Input**: RGB images resized to 64x64 pixels
- **Output Classes**: Thumbs Up, Thumbs Down, Fist, and Five
- **Training Dataset**: Custom images gathered and labeled for each gesture

## Development Roadmap
Possible future improvements and extensions:

- **Additional Gestures**: Expand model capability to recognize more gestures.
- **Performance Optimization**: Adjust model parameters to improve accuracy and speed.
- **Integration**: Use this model in interactive applications, such as gesture-controlled screens or virtual reality environments.

