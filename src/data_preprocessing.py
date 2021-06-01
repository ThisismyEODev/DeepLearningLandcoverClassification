# -*- coding: utf-8 -*-
"""

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""


import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

def input_data_preparation(data_foldername, parameters):
    
    path_to_imagery = parameters.path / 'inputdata' / data_foldername
    all_files_in_path = list(path_to_imagery.rglob("*"))
    paths_to_imgs = [x for x in all_files_in_path if\
                     parameters.data_format in x.suffix]

    all_files_in_path = None

    eurosat_imgs = np.zeros([len(paths_to_imgs), 
                             parameters.img_size, 
                             parameters.img_size, 
                             parameters.n_bands])
    i = 0
    for pic in paths_to_imgs:
        eurosat_imgs[i] = np.asarray(Image.open(pic)).astype(np.uint8)/255
        i += 1

    labels = []
    for path in paths_to_imgs:
        labels.append(str(path.parent.name))

    # Extract classes and create a label dictionary
    classes, encoded_labels = np.unique(labels, return_inverse=True)
    encoded_labels = [lab.astype(np.uint8) for lab in encoded_labels] 
    encoded_labels = np.asarray(encoded_labels)

    label_dictionary = dict(zip(np.unique(encoded_labels), classes))
    print("We have the following image content classes' \
          'and their respective encoding:", label_dictionary, "\n")
    num_classes = len(np.array(np.unique(classes)))
    print(f"We have a total of {num_classes} classes")

    labels = None
    paths_to_imgs = None

    if parameters.balanced == True:
        print("We want to make sure our class distribution is balanced")
        smallest_class = np.argmin(np.bincount(encoded_labels))
        smallest_class_ind = np.where(encoded_labels == smallest_class)[0]

        print("Pick from each class the same number of samples ")
        index_balanced = []
        for i in range(num_classes):
            tmp = shuffle(np.where(encoded_labels == i)[0], 
                      random_state=42)[0:smallest_class_ind.shape[0]]
            index_balanced.append(tmp)

        index_balanced = [item for sublist in index_balanced for item in sublist]
        print(f"We have a total of {len(index_balanced)} images to work with")

        index_balanced = shuffle(index_balanced, 
                                 random_state = parameters.seed)
        encoded_labels_balanced = encoded_labels[index_balanced]
        eurosat_imgs_balanced = eurosat_imgs[index_balanced]

        X_train, X_validation,\
            y_train, y_validation =\
                train_test_split(eurosat_imgs_balanced, 
                         encoded_labels_balanced, 
                         stratify = encoded_labels_balanced, 
                         train_size = .9, 
                         random_state = parameters.seed)

        _, X_test, _, y_test =\
            train_test_split(eurosat_imgs_balanced, 
                         encoded_labels_balanced, 
                         stratify = encoded_labels_balanced, 
                         test_size = .1, 
                         random_state = parameters.seed)

        y_train_encoded = tf.keras.utils.to_categorical(
            y_train, num_classes=None, dtype='float32')

        y_validation_encoded = tf.keras.utils.to_categorical(
            y_validation, num_classes=None, dtype='float32')

        y_test_encoded = tf.keras.utils.to_categorical(
            y_test, num_classes=None, dtype='float32')

    else:
        X_train, X_validation,\
            y_train, y_validation =\
                train_test_split(eurosat_imgs, 
                         encoded_labels, 
                         stratify = encoded_labels, 
                         train_size = parameters.training_size, 
                         random_state = parameters.seed)

        _, X_test, _, y_test =\
            train_test_split(eurosat_imgs_balanced, 
                         encoded_labels_balanced, 
                         stratify = encoded_labels_balanced, 
                         test_size = parameters.test_size, 
                         random_state = parameters.seed)

        y_train_encoded = tf.keras.utils.to_categorical(
            y_train, num_classes=None, dtype='float32')

        y_validation_encoded = tf.keras.utils.to_categorical(
            y_validation, num_classes=None, dtype='float32')

        y_test_encoded = tf.keras.utils.to_categorical(
            y_test, num_classes=None, dtype='float32')

    return (label_dictionary, classes, 
            X_train, y_train_encoded, 
            X_validation, y_validation_encoded,
            X_test, y_test_encoded)



