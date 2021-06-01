# -*- coding: utf-8 -*-
"""
@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import keras
from keras import models
from keras import layers
from keras.applications.resnet50 import ResNet50
import tensorflow as tf

def build_model(parameters, X_train, y_train):

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

def compile_and_fit_model(parameters, model, 
                          X_train, y_train, 
                          X_test, y_test,
                          save_model):

    model.compile(optimizer = parameters.optimizer,
                  loss = parameters.loss_function,
                  metrics = parameters.model_metric)
    
    model_file = parameters.model_path / parameters.model_name
    checkpoint = keras.callbacks.ModelCheckpoint(filepath = model_file,
                                             monitor = 'val_loss',
                                             save_best_only = True)
    callback_list = [checkpoint]

    history = model.fit(X_train, y_train,
                 batch_size = parameters.batch_size,
                 epochs = parameters.epoch,
                 steps_per_epoch = parameters.steps_per_epoch,
                 validation_data=(X_test, y_test),
                 validation_steps = parameters.validation_steps,
                 callbacks = callback_list,
                 verbose=1)
    
    if save_model:
        filepath = model_file
        tf.keras.models.save_model(
            model, filepath, overwrite=True, 
            include_optimizer=True, save_format=None,
            signatures=None, options=None, save_traces=True)

    return history

def compile_and_fit_model_from_generator(parameters, model, 
                                         train_generator,
                                         X_train,
                                         test_generator,
                                         X_test,
                                         save_model):

    model.compile(optimizer = parameters.optimizer,
                  loss = parameters.loss_function,
                  metrics = parameters.model_metric)
    
    model_file = parameters.model_path / parameters.model_name
    checkpoint = keras.callbacks.ModelCheckpoint(filepath = model_file,
                                             monitor = 'val_loss',
                                             save_best_only = True)
    callback_list = [checkpoint]

    history = model.fit_generator(train_generator,
                                  batch_size = parameters.batch_size,
                                  epochs = parameters.epoch,
                                  steps_per_epoch = parameters.steps_per_epoch,
                                  validation_data = test_generator,
                                  validation_steps = parameters.validation_steps,
                                  callbacks = callback_list,
                                  verbose=1)
    
    if save_model:
        filepath = model_file
        tf.keras.models.save_model(
            model, filepath, overwrite=True, 
            include_optimizer=True, save_format=None,
            signatures=None, options=None, save_traces=True)

    return history


