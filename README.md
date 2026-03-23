# Webcam Image Classifier
A beginner machine learning application that classifies a pen or a phone from a live webcam feed.

 ![Image](https://github.com/user-attachments/assets/e606c063-ca80-4654-be87-4b3d97af38f9)

 # Features
 - Real-time webcam input 
 - Classifies between a pen and phone
 - Displays the predicited label on screen
 - Used a trained machine learning model

# Tech used 
- Python
- OpenCV
- Numpy
- Tensorflow / Keras
- Teachable Machine

# How it works
- The app opens the webcam and captures frames.
- Each frame is then resized to the dimentions expected by the model.
- The image is then normalised so that the pixel values match what the model expects.
- Then the model makes a prediction and gives us confidence scores.
- The app then selects the label with the highest confidence and displays it on the webcam frame.

# Files 
- classifier.py : Main script for the image 
- keras_model.h5 : The trained Machine Learning model itself
- labels.txt : Class labels used by the model

# How to run
- Install the required package versions 
- Run the script

  --- Using Bash :
  Python classifier.py

# What I learned 
- How to use OpenCV to capture live webcam frames in python.
- How to load and use a trained model in python.
- How to prepare image data so that it matches the format the model expects.
- How to use model predictions to classify objects in real time.
- How to debug issues with the model loading, image shape, and prediction flow.
- How to display real-time predictions on-screen.
