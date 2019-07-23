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
from utils import *

driving_log = pd.read_csv('data/driving_log.csv')
train_samples, validation_samples = train_test_split(driving_log, test_size=0.2)

# Set our batch size
batch_size= 64

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

def build_model():
    # Model construction
    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(66,200,3),
            output_shape=(66,200,3)))

    model.add(Conv2D(24, (5, 5),padding='valid',strides=(2,2)))
    model.add(ELU())
    model.add(Conv2D(36, (5, 5), padding='valid',strides=(2,2)))
    model.add(ELU())
    model.add(Conv2D(48, (5, 5), padding='valid',strides=(2,2)))
    model.add(ELU())
    model.add(Conv2D(64, (3, 3), padding='valid'))
    model.add(ELU())
    model.add(Conv2D(64, (3, 3), padding='valid'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))
    
    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer= adam)
    return model

model = build_model()
history = model.fit_generator(train_generator, 
                        steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=np.ceil(len(validation_samples)/batch_size),
                        epochs=10, verbose=1)

model.save('my_model.h5')
