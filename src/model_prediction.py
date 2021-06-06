# -*- coding: utf-8 -*-
"""
@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def make_prediction(model, X_test, y_test, label_dictionary):
    """
    Make prediction based on pretrained model and test data
    
    Parameters
    ----------
    model:
        tf.keras.sequential model instance
        Pre-trained ResNet50model
    X_test:
        numpy array of size (perc_testing, imagesize, imagesize, numofbands)
        Testing data
    y_test:
        numpy array of size (perc_testing)
        Testing labels
    label_dictionary:
        dict
        Dictionary of labels

    Returns
    -------
    predicted_labels:
        numpy array of size (perc_testing)
        Predicted labels
    y_test_true : TYPE
        DESCRIPTION.
    y_test_pred : TYPE
        DESCRIPTION.
    y_pred_encoded : TYPE
        Binary-valued predicted labels

    """

    y_pred = model.predict(X_test)
    predicted_labels = [np.argmax(y) for y in y_pred]
    y_test_true = [label_dictionary[x] for x in y_test]
    y_test_pred = [label_dictionary[x] for x in predicted_labels]

    y_pred_encoded = tf.keras.utils.to_categorical(
        predicted_labels, num_classes=len(label_dictionary), dtype='float32'
        )
    return predicted_labels, y_test_true, y_test_pred, y_pred_encoded


def run_prediction_on_example_image(model, classes, X_test, index_example):
    """
    Runs the prediction on a random image based on a previously run model

    model:      keras deep learning model, previously trained on data
    classes:    classes involved in the classification 
                list of strings
    X_test:     Testing images
                Array of float of size 
                (num testing samples, width, height, number of bands)
    index_example:  random index
                    int
    """
    proba = model.predict(X_test[index_example])[0]
    idxs = np.argsort(proba)[::-1][:2]
    
    for (i, j) in enumerate(idxs):
        label = "{}: {:.2f}%".format(classes[j], proba[j] * 100)
        print(label)
    
    for c, p in zip(classes, proba):
        print("{}: {:.2f}%".format(c, p * 100))
    
    plt.imshow(X_test[index_example])
    plt.show()
    
    
