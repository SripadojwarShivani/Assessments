#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:18:40 2022

@author: shivanisri
"""

import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import AveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
x_train.shape
y_train.shape
x_train[0]
classes = [0,1,2,3,4,5,6,7,8,9]
x_train = x_train/255
y_train = y_train/255
x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)
x_train_flattened.shape
model = Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape = (784,),activation='relu'))
model.add(AveragePooling2D(2,2))

model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape = (784,),activation='relu'))
model.add(AveragePooling2D(2,2))

model.add(Conv2D(filters=128,kernel_size=(3,3),input_shape = (784,),activation='relu'))
model.add(AveragePooling2D(2,2))
model.add(Dense(10, input_shape=(784,), activation='sigmoid'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=5)
model.evaluate(x_test_flattened, y_test)