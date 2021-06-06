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
from keras.utils  import plot_model
import IPython

def build_model(parameters, classes):
    """
    Generates the ResNet50 model

    Parameters
    ----------
    parameters: 
        Parameters set in the src/parameter_file.py
    classes:
        Array of strings
        Names of each label

    Returns
    -------
    model_resnet:
        tf.keras.sequential model instance
        Pre-trained ResNet50model

    """
    base_model_resnet = ResNet50(include_top = False, 
                             weights = parameters.w, 
                             input_shape = (parameters.img_size,
                                            parameters.img_size,
                                            parameters.n_bands), 
                             classes = len(classes))

    model_resnet = models.Sequential()
    model_resnet.add(base_model_resnet)
    model_resnet.add(layers.Flatten())

    model_resnet.add(layers.Dense(128, activation=('relu')))
    model_resnet.add(layers.Dense(64, activation=('relu')))
    model_resnet.add(layers.Dense(10, activation=('softmax')))

    print(model_resnet.summary())

    img_path = str(parameters.path / 'model_directory' / parameters.model_plot_name)
    plot_model(model_resnet, to_file = img_path, show_shapes=True)
    IPython.display.Image(img_path)
    
    if parameters.freeze_layers == True:
        for layer in model_resnet.layers:
            layer.trainable = False        

    return model_resnet

def compile_and_fit_model(parameters, model, 
                          X_train, y_train, 
                          X_validation, y_validation,
                          save_model):
    """
    Compiles and runs the ResNet50 model
    
    parameters:     input parameter which is automatically loaded into the main.py
                    file
    
    X_train:        Training images
                    Array of float of size (num training samples, width, height, number of bands)
    y_train:        Encoded labels                
                    Array of float of size (num training samples, number of classes)
    X_validation:   Validation images
                    Array of float of size 
                    (num validation samples, width, height, number of bands)
    y_validation:    Encoded labels                
                    Array of float of size (num validation samples, number of classes)
    save_model: Wether to save the model or not
                Bool
    
    Returns:
    
    history     Model training history
    """

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
                 validation_data=(X_validation, y_validation),
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
                                         test_generator,
                                         save_model):
    """
    Compiles and runs the ResNet50 model on generators
    
    parameters:         input parameter which is automatically loaded into the main.py
                        file
    train_generator:    training data generator
                        keras imagedatagenerator instance
    test_generator:     testing data generator
                        keras imagedatagenerator instance
    save_model:         Wether to save the model or not
                        Bool

    Returns:

    history     Model training history
    """
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


