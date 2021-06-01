# -*- coding: utf-8 -*-
"""

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

def evaluate_model_accuracy(parameters, model, history, test_imgs, test_labels):
    score = model.evaluate(test_imgs, test_labels, verbose=0)
    print('Test loss:', np.round(score[0],7))
    print('Test accuracy:', np.round(score[1],7), "\n")
    
    plt.style.use("ggplot")
    plt.figure()
    N = parameters.epochs
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    # plt.savefig(args["plot"])



