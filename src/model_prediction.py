# -*- coding: utf-8 -*-
"""
@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt


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
    
    
