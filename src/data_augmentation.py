# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:49:13 2021

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

from keras.preprocessing.image import ImageDataGenerator

def augment_data(parameters, X_train, y_train_enc, 
                 X_validation, y_validation_enc):
    """
    Performs data augmentation before model training

    Parameters
    ----------
    parameters: 
        Parameters set in the src/parameter_file.py
    X_train: 
        numpy array of size (perc_training, imagesize, imagesize, numofbands)
        Training data
    y_train_enc:
        numpy array of size (perc_training, num of labels)
        Encoded training labels
    X_validation:
        numpy array of size (perc_testing, imagesize, imagesize, numofbands)
        Validation data
    y_validation:
        numpy array of size (perc_testing, num of labels)
        Encoded validation labels

    Returns
    -------
    train_generator: 
        ImageDataGenerator Instance
        Training data
    validation_generator:
        ImageDataGenerator Instance
        Validation data

    """

    train_datagen = ImageDataGenerator(zoom_range = parameters.zoom, 
                                       rotation_range = parameters.rotation,
                                       width_shift_range = parameters.width_shift, 
                                       height_shift_range = parameters.height_shift, 
                                       shear_range = parameters.shear, 
                                       horizontal_flip = True, 
                                       fill_mode = parameters.fill)

    validation_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train,
        y_train_enc,
        batch_size = parameters.batch_size,
        seed = parameters.seed
        )
    
    validation_generator = validation_datagen.flow(
        X_validation,
        y_validation_enc,
        batch_size = parameters.batch_size,
        seed = parameters.seed
        )
    
    return train_generator, validation_generator