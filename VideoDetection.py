import numpy as np 
import keras
import tensorflow
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam
#Importing image module to preprocess the image data.
from tensorflow.keras.preprocessing import image
#Importing OpenCV, a computer vision library, to access the camera and perform real-time face detection.
import cv2
import datetime
import matplotlib.pyplot as plt

# Building model to classify between mask and no mask:
#Initializing a sequential model.
seqModel=Sequential()
# Adds a 2D convolutional layer with 32 filters, each of size 3x3, and a ReLU activation function. 
seqModel.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
#Adds a max pooling layer with a default pool size of 2x2. This layer reduces the dimensionality of the previous convolutional layer's output, making the model more efficient.
seqModel.add(MaxPooling2D() )
#: Adds another 2D convolutional layer with the same specifications as the first.
seqModel.add(Conv2D(32,(3,3),activation='relu'))
#Adds another max pooling layer.
seqModel.add(MaxPooling2D() )
# Adds another 2D convolutional layer with the same specifications as the previous two.
seqModel.add(Conv2D(32,(3,3),activation='relu'))
#Adds another max pooling layer.
seqModel.add(MaxPooling2D() )
#Flattens the output from the previous layer into a 1D array, which can then be fed into a fully connected neural network layer.
seqModel.add(Flatten())
#Adds a fully connected neural network layer with 100 neurons and a ReLU activation function.
seqModel.add(Dense(100,activation='relu'))
# Adds another fully connected neural network layer with 1 neuron and a sigmoid activation function. This is the output layer of the model and predicts a binary output (0 or 1).
seqModel.add(Dense(1,activation='sigmoid'))

seqModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Importing ImageDataGenerator to preprocess the image data.
from keras.preprocessing.image import ImageDataGenerator
#Creating an instance of the ImageDataGenerator class to apply data augmentation techniques like rescaling, shearing, zooming, and horizontal flipping to the training data.
traindata = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#Creating another instance of ImageDataGenerator class to rescale the pixel values of the test data.
testdata = ImageDataGenerator(rescale=1./255)
#Generating train data and its labels by reading images from directory 'train'
train_set = traindata.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=16 ,
        class_mode='binary')
#Generating test data and its labels by reading images from directory 'test'
test_set = testdata.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')
#Fitting the model on the train data and validating it on the test data
model_saved=seqModel.fit_generator(
        train_set,
        epochs=20,
        validation_data=test_set,

        )

seqModel.save('mymodel.h5',model_saved)

# Plotting the training and validation accuracy and loss
plt.plot(model_saved.history['loss'], label='Training Loss')
plt.plot(model_saved.history['val_loss'], label='Validation Loss')
plt.plot(model_saved.history['accuracy'], label='Training Accuracy')
plt.plot(model_saved.history['val_accuracy'], label='Validation Accuracy')
# plotting accuracy and loss graph
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()


# IMPLEMENTING LIVE DETECTION OF FACE MASK
#Load the saved model
mymodel=load_model('mymodel.h5')
#Initialize the video capture object
capture=cv2.VideoCapture(0)
#Load the Haar Cascade classifier for face detection
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Enter the infinite loop to capture live video and perform face mask detection
while capture.isOpened():
    _,capturedImg=capture.read()
    # Detect faces in the captured frame using the Haar Cascade classifier
    face=faceCascade.detectMultiScale(capturedImg,1.1,4)
    ## Iterate over each face detected
    for(x,y,w,h) in face:
        # Extract the face region from the captured image
        face_img = capturedImg[y:y+h, x:x+w]
        # Save the face image to a temporary file
        cv2.imwrite('temp.jpg',face_img)
        ## Load the saved face image and resize it to the required input size for the model
        testimage=image.load_img('temp.jpg',target_size=(150,150,3))
        testimage=image.img_to_array(testimage)
        testimage=np.expand_dims(testimage,axis=0)
        #  Predict whether the person in the face image is wearing a mask or not using the loaded model
        pred=mymodel.predict(testimage)[0][0]
        # Draw a rectangle around the face and label it with the prediction result
        if pred==1:
            cv2.rectangle(capturedImg,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(capturedImg,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(capturedImg,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(capturedImg,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        # Add the current date and time to the image
        datet=str(datetime.datetime.now())
        cv2.putText(capturedImg,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
# Display the captured image with the face detection and mask prediction results
   
    cv2.imshow('img',capturedImg)
# Break the loop if the user presses the 'q' key
    if cv2.waitKey(1)==ord('q'):
        break
 #Release the video capture object and close all windows   
capture.release()
cv2.destroyAllWindows()