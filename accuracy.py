import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image


# Load the pre-trained model
model = load_model('mymodel.h5')

# Define the paths to the test dataset
test_dir = 'C://Nithish Japala//AI//Project//FaceMaskDetector-master//test'
mask_dir = os.path.join(test_dir, 'with_mask')
no_mask_dir = os.path.join(test_dir, 'without_mask')

# Load the test images and labels
mask_images = [image.load_img(os.path.join(mask_dir, f), target_size=(150, 150)) for f in os.listdir(mask_dir)]
no_mask_images = [image.load_img(os.path.join(no_mask_dir, f), target_size=(150, 150)) for f in os.listdir(no_mask_dir)]


mask_images = np.array([np.array(img) for img in mask_images])
no_mask_images = np.array([np.array(img) for img in no_mask_images])

# Combine images into one array
test_images = np.concatenate([mask_images, no_mask_images])

test_labels = np.array([1]*len(mask_images) + [0]*len(no_mask_images))


# Predict the labels for the test images using the pre-trained model
predicted_labels = model.predict(test_images)

# Convert the predicted labels to binary values (0 or 1)
predicted_labels = np.round(predicted_labels)

# Calculate the accuracy of the model
accuracy = np.sum(predicted_labels == test_labels) / len(test_labels)
print(predicted_labels)
print('Accuracy:', accuracy)
