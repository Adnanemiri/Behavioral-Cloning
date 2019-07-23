import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from keras.models import Sequential, Model
from keras.layers import Cropping2D, ELU
from keras import backend
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def crop(image):
    image = image[50:140,:]
    image = cv2.resize(image,(200,66), interpolation=cv2.INTER_AREA)
    return image

def preprocess_image(row_data):
    
    angle = row_data['steering']
    direction = np.random.choice(['center', 'left', 'right'])

    if direction == 'left':
        path_file = 'data' + row_data['left']
        angle += 0.25
    elif direction == 'right':
        path_file = 'data' + row_data['right']
        angle -= 0.25
        
    else:
        path_file = 'data' + row_data['center']
    
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = crop(image)
    
    flip = np.random.randint(2)
    if flip==0:
        image = cv2.flip(image,1)
        angle = -angle
    
    return image, angle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = samples.sample(frac=1)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]

            images = []
            angles = []
            for r in range(0,len(batch_samples)):
                
                row_data = batch_samples.iloc[r]
                image, angle = preprocess_image(row_data)

                images.append(image)
                angles.append(angle)
                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))            
