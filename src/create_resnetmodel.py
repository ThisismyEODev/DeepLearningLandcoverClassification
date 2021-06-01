# -*- coding: utf-8 -*-
"""
@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import keras
from keras import models
from keras import layers
from keras.applications.resnet50 import ResNet50

def build_model(parameters, X_train, y_train, X_validation, y_validation, model_name):

    model_file = parameters.model_path / model_name
    checkpoint = keras.callbacks.ModelCheckpoint(filepath = model_file,
                                             monitor = 'val_loss',
                                             save_best_only = True)
    callback_list = [checkpoint]
    
    base_model_resnet = ResNet50(include_top = False, 
                             weights = parameters.w, 
                             input_shape = (parameters.img_size,
                                            parameters.img_size,
                                            parameters.n_bands), 
                             classes = y_train.shape[1])

    model_resnet = models.Sequential()
    model_resnet.add(base_model_resnet)
    model_resnet.add(layers.Flatten())

    model_resnet.add(layers.Dense(128, activation=('relu')))
    model_resnet.add(layers.Dense(64, activation=('relu')))
    model_resnet.add(layers.Dense(10, activation=('softmax')))

    return model_resnet

