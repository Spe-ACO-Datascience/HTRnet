# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:04:18 2022

@author: Emilie
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#from keras.engine import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers import Activation, Input
from keras import backend as k 
from keras import optimizers, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint 


x_train = X_train
x_test = X_test
'''
Dataset parameters
split dataset to train and test
'''
# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)
input_shape = (128, 128, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

n_samples =  x_train.shape[0]
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
batch_size = 128


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(x_train, y_train,batch_size=batch_size)
validation_generator = train_datagen.flow(x_train, y_train,batch_size=batch_size)
test_generator = test_datagen.flow(x_test)






'''
Parameters
'''
# Training Parameters
learning_rate = 0.001
epochs = 5 # small number of epoch to save time (it should be increased)

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

img_width, img_height = 128, 128


'''
Layers 
'''
# Creating a Sequential Model and adding the layers

conv_net_in = Input(shape=(img_width, img_height, 1))
# First 2D convolution Layer
# Convolution Layer with 32 filters and a kernel size of 5
conv_net = Conv2D(32, (5, 5))(conv_net_in)
conv_net = Activation("relu")(conv_net)
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
conv_net = MaxPooling2D()(conv_net)
# Second 2D convolution Layer
conv_net = Conv2D(64, (3, 3), padding="same")(conv_net)
conv_net = Activation("relu")(conv_net)
conv_net = MaxPooling2D()(conv_net)
# Flatten the data to a 1-D vector for the fully connected layer
conv_net = Flatten()(conv_net)

# first fully connected
conv_net = Dense(1024)(conv_net)
conv_net = Activation('relu')(conv_net)
conv_net = Dropout(dropout)(conv_net)

# Output
conv_net = Dense(10)(conv_net)
conv_net = Activation('softmax')(conv_net)


conv_model = Model(conv_net_in, conv_net)

conv_model.summary()



'''
Hyper parameter for the learning task
''' 
adam = tf.keras.optimizers.Adam(lr=learning_rate)

conv_model.compile(optimizer=adam, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="CNN_MNIST_best_weights.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)


'''
Fit modele
'''
conv_model.fit(
        train_generator,
        steps_per_epoch=n_samples/batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=100,
        callbacks=[checkpointer])




