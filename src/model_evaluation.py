# -*- coding: utf-8 -*-
"""

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_model_accuracy(model, history, epochs):
    """
    Plots the evolution of accuracy and loss with number of epochs

    Parameters
    ----------
    model:
        tf.keras.sequential model instance
        Pre-trained ResNet50model
    history : 
        History object
        Training history of the model
    epochs:
        int
        number of epochs


    """
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")



