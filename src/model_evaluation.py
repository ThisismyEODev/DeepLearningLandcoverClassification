# -*- coding: utf-8 -*-
"""

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

def evaluate_model_accuracy(model, history, test_imgs, test_labels):
    score = model.evaluate(test_imgs, test_labels, verbose=0)
    print('Test loss:', np.round(score[0],7))
    print('Test accuracy:', np.round(score[1],7), "\n")
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


