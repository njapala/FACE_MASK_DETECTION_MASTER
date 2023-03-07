import numpy as np
import keras
import tensorflow
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import cv2
import datetime
import matplotlib.pyplot as plt

#To test for individual images

mymodel=load_model('mymodel.h5')
img_path = 'C://Nithish Japala//AI//Project//FaceMaskDetector-master//WIN_20230303_13_55_42_Pro.jpg'
test_image = image.load_img(img_path, target_size=(150, 150))
test_image
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
output = mymodel.predict(test_image)[0][0]

# Classify the output
if output == 1:
    result = 'NoMask'
else:
    result = 'Mask'

# Load the input image again
img = cv2.imread(img_path)

# Detect faces in the input image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# For each detected face, draw a bounding box and label it with the classification result
for (x, y, w, h) in faces:
    # Draw bounding box
    if result == 'NoMask':
        color = (0, 0, 255) # Red color for no mask
    else:
        color = (0, 255, 0) # Green color for mask
        
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    
    # Label the classification result
    cv2.putText(img, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Display the output image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Save the output image with the classification result
cv2.imwrite(f"output_{result}.jpg", img)