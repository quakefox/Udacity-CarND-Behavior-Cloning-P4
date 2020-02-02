#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def parse_record_data():
    samples = []
    with open(datapath + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def preprocess_image(imgfilename):
    filename = imgfilename.split('/')[-1]
    filename = datapath + 'IMG/' + filename
    img = cv2.imread(filename)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = (img-128.0)/128
    #crop_img = img[60:140, 0:320]
    return img
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            #print('probe 1')
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = preprocess_image(batch_sample[0])
                left_image = preprocess_image(batch_sample[1])
                right_image = preprocess_image(batch_sample[2])
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            # trim image to only see section with road
            #augment images, turn ON for track 1, turn OFF for track 2
            #images, angles = augment_training_data(images, angles)
            images_array = np.array(images)
            angles_array = np.array(angles)
        
            X_train = images_array
            y_train = angles_array
            yield shuffle(X_train, y_train)
            
def augment_training_data(images, measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
    return augmented_images, augmented_measurements
            
            
            
#recorded data
#datapath = 'recording/Route1/'
datapath = 'recording/Route2/'

#angle correction
correction = 0.2
            
# Set our batch size
batch_size = 64

samples = parse_record_data()
print('sample length is ', len(samples))

# Split training and validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

print('train_generator shape:', train_generator)



# In[10]:



# Validation of images and Data
import matplotlib.pyplot as plt

def testplot_sample_image(sample):
    img = cv2.imread(sample)
    #center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (img-128.0)/128
    crop_img = img[60:139, 0:319]
    plt.figure(figsize=(10,10))
    plt.imshow(crop_img)
    return crop_img

test_sample = samples[1]

print(test_sample[0])

img1 = testplot_sample_image(test_sample[0])
img2 = testplot_sample_image(test_sample[1])
img3 = testplot_sample_image(test_sample[2])

a = []
a.append(img1)
a.append(img2)
a.append(img3)
a = np.array(a)
print('a.shape:', a.shape)

a = a.reshape(a.shape[0],79,319,1)
print('after reshape: a.shape:', a.shape)


# In[6]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import math
from keras.models import Model
import matplotlib.pyplot as plt

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20),(0,0))))
#model.add(Conv2D(24,5,5,input_shape=(70,320,3), subsample=(2,2), activation='relu'))
model.add(Conv2D(24,5,5,subsample=(2,2), activation='relu'))

#subsample=(2,2),activation='relu', 
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,                     steps_per_epoch=math.ceil(len(train_samples)/batch_size),                     validation_data=validation_generator,                     validation_steps=math.ceil(len(validation_samples)/batch_size),                     epochs=7, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


model.summary()

