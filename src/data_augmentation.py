# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:49:13 2021

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

from keras.preprocessing.image import ImageDataGenerator

def augment_data(parameters, X_train, y_train, X_test, y_test):

    train_datagen = ImageDataGenerator(
        shear_range = parameters.shear,
        zoom_range = parameters.zoom,
        rotation_range = parameters.rotation,
        horizontal_flip = True,
        vertical_flip = True)

    test_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size = parameters.batch_size,
        seed = parameters.seed
        )
    
    test_generator = test_datagen.flow(
        X_test,
        y_test,
        batch_size = parameters.batch_size,
        seed = parameters.seed
        )
    
    return train_datagen, test_datagen