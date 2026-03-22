from tensorflow.keras.models import load_model
import cv2
import numpy as np

# load the model
model = load_model('keras_model.h5', compile=False)
# Load the labels, removed all the numbers from the list
class_names = [line.split()[1] for line in open('labels.txt', 'r').readlines()]
# Start the webcam
cam = cv2.VideoCapture(0)

# Process frames (using while loops)
while cam.isOpened():
# Take the webcam's image
    ret, img = cam.read()
# resize the image
    img = cv2.resize(img,(224, 224), interpolation = cv2.INTER_AREA)
# Flip the image vertically
    flipped_img = cv2.flip(img, 1)
# Showing the image in a window
    cv2.imshow('flipped img', flipped_img)
# Converts the image into the right format, data type and shape that the model will expect.
    img = np.asarray(img, dtype="float32").reshape(1, 224, 224, 3)
# Normalise the image
    img_normalised = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
# The model predicts
    prediction = model.predict(img_normalised)
    print(prediction)

# Get labels

# Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(2)
# 27 is the ASCII for the esc key on your keyboard
# This line stops the webcam when you press esc key
    if keyboard_input == 27:
        break

# 
cam.release()
cv2.destroyAllWindows()