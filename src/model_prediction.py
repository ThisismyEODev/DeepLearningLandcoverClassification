# -*- coding: utf-8 -*-
"""
@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn import metrics
from scipy import interp

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


def predict_on_single_testimage(model, X_test, y_test, class_names):
    """
    Pics test image at random and plots it together with class name

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
    class_names:
        array of strings
        Names of the labels

    """
    
    length_test_data = len(X_test)
    indexes_test_data = np.arange(length_test_data)
    ind = np.random.choice(indexes_test_data, 1, replace=False)
    img = X_test[ind, :, :, :]
    prediction = model.predict_classes(img, verbose=0)

    plt.figure(figsize=(10,10))
    plt.imshow(img[0])
    plt.title(f"The Model thinks this is a {class_names[prediction[0]]}\
 - The test label is a {class_names[y_test[ind]]}", fontsize=15)
    plt.show()
    
def print_prediction_score(model, X_test, class_names):
    """
    Prints prediction score of randomly picked image

    Parameters
    ----------
    model:
        tf.keras.sequential model instance
        Pre-trained ResNet50model
    X_test:
        numpy array of size (perc_testing, imagesize, imagesize, numofbands)
        Testing data
    class_names:
        array of strings
        Names of the labels

    """
    length_test_data = len(X_test)
    indexes_test_data = np.arange(length_test_data)
    ind = np.random.choice(indexes_test_data, 1, replace=False)
    img = X_test[ind, :, :, :]

    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0])
    print("This image most likely belongs to the {} class with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score)))

def create_prediction_dataframe(y_test_true, y_test_pred, num_classes):
    """
    

    Parameters
    ----------
    y_test_true : TYPE
        DESCRIPTION.
    y_test_pred : TYPE
        DESCRIPTION.
    num_classes:
        int
        Number of classes

    Returns
    -------
    pred_df : 
        Pandas dataframe
        DESCRIPTION.

    """
    pred_df = pd.DataFrame({'y_true': y_test_true, 'y_pred': y_test_pred})
    pred_df['accurate_preds'] = pred_df.y_true == pred_df.y_pred
    pred_df = pred_df.groupby(['y_true']).sum().reset_index()
    pred_df['label_count'] = num_classes
    pred_df['class_acc'] = pred_df.accurate_preds / pred_df.label_count
    pred_df = pred_df.sort_values(by = 'class_acc').reset_index()
    pred_df['overall_acc'] = sum(pred_df.accurate_preds) / sum(pred_df.label_count)
    pred_df = pred_df.sort_values('y_true').reset_index(drop = True)
    return pred_df

def dataframe_of_accurate_and_nonaccurate_prediction(y_test_true, y_test_pred):
    """
    Parameters
    ----------
    y_test_true : TYPE
        DESCRIPTION.
    y_test_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    pred_df : 
        Pandas dataframe
        DESCRIPTION.
    """

    pred_df = pd.DataFrame({'y_true': y_test_true, 'y_pred': y_test_pred})
    pred_df['accurate_preds'] = pred_df.y_true == pred_df.y_pred
    pred_df = pred_df.sort_values('y_true')
    return pred_df

def plot_confusion_matrix(y_test_true, y_test_pred, classes):
    """
    Plots annotated confusion matrix    

    Parameters
    ----------
    y_test_true : TYPE
        DESCRIPTION.
    y_test_pred : TYPE
        DESCRIPTION.
    classes:
        Array of strings
        Names of classes


    """

    cm = metrics.confusion_matrix(y_test_true, y_test_pred)
    cmap = plt.cm.Blues
    fig, ax = plt.subplots(figsize=(7,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="title",
           ylabel='True label',
           xlabel='Predicted label')

def plot_model_roc_curve(model, y_test_encoded, y_pred_encoded, classes):
    """
    Compute ROC curve and ROC area for each class

    Parameters
    ----------
    model:
        tf.keras.sequential model instance
        Pre-trained ResNet50model
    y_test_encoded:
        binary valued numpy array of size (perc_testing, num_classes)
        Encoded testing labels
   y_pred_encoded:
        binary valued numpy array of size (perc_testing, num_classes)
        Encoded predicted labels
    classes:
        Array of strings
        Names of classes

    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = len(classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_encoded[:, i], 
                                              y_pred_encoded[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    ## Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_encoded.ravel(), 
                                                          y_pred_encoded.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(6, 4))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                      ''.format(roc_auc["micro"]), linewidth=3)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]), linewidth=3)

    for i, label in enumerate(classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                            ''.format(label, roc_auc[i]), 
                                            linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
