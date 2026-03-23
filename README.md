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
- Each frame is then resized for the model to read/use.
- The image is then normalised so that the model can read pixels accurately.
- Then the model makes a prediction and gives us confidence scores.
- The app then selects the label with the highest confidence and displays it on the webcam frame.

# Files 
- Classifier.py : Main script for the image 
- Keras_model.h5 : The trained Machine Learning model itself
- Labels.txt : Class labels used by the model

# How to run
- Install the required packages
- Run the script

  --- Using Bash :
  Python predict_webcam.py
