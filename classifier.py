from tensorflow.keras.models import load_model
import cv2
import numpy as np

# load the model
model = load_model('keras_model.h5', compile=False)
# Load the labels, removed all the numbers from the list
class_names = [line.split()[1] for line in open('labels.txt', 'r').readlines()]
# Start the webcam
cam = cv2.VideoCapture(0)

# Keep the loop while the webcam is opened
while cam.isOpened():
# Read one frame and check if it works, if not loop breaks
    ret, img = cam.read()
    if not ret:
        break

# resize the frame to the size the model expects
    img = cv2.resize(img,(224, 224), interpolation = cv2.INTER_AREA)

# Flip the image vertically
    flipped_img = cv2.flip(img, 1)

# Converts the image into the right format, data type and shape that the model will expect.
    img = np.asarray(flipped_img, dtype="float32").reshape(1, 224, 224, 3)

# Normalise the image array for the model to improve prediction accuracy
    img = (img / 127.5) - 1

# The model makes predictions
    prediction = model.predict(img)
    print(prediction)

# Automate the prediction so that python finds the biggest number in the index
    index = np.argmax(prediction[0])

# Show the results of the prediction with the correct label
    print(class_names[index])

# Need to put label on the image (webcam), slightly changed the positioning of the text away from the centre of the frame.
    cv2.putText(flipped_img, class_names[index], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show the image on the webcam which is separate from what the model sees.
    cv2.imshow('Image', flipped_img)

# Need to keep the image on the screen.
    keyboard_input = cv2.waitKey(1)

# Need to press esc to exit the loop
    if keyboard_input == 27:
        break

# Release the webcam
cam.release()

# Completely close all opened windows.
cv2.destroyAllWindows()