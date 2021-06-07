# -*- coding: utf-8 -*-
"""
@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_data_distribution_and_correlation(parameters, classes, X, y):
    """
    Generates some class distribution plots for basic data exploration

    Parameters
    ----------
    parameters: 
        Parameters set in the src/parameter_file.py
    classes:
        Array of strings
        Names of each label
    X:
        numpy array of size (numdata, imagesize, imagesize, numofbands)
        Numerical data
    y:
        numpy array of size (numdata)
        string data

    """
    distribution_per_class = []
    for i in range(len(classes)):
        ind = np.argwhere(y == i)
        ind = ind[0][0]
        img = X[ind]
        imgvals = img.flatten()
        distribution_per_class.append(imgvals)
        
    distribution_per_class = np.asarray(distribution_per_class)
    dataframe = pd.DataFrame(distribution_per_class.T, columns = classes)
    dataframe.to_csv(str(parameters.path /\
                         'inputdata' / 'per_class_distribution.csv'), sep = "\t")
    print(dataframe.describe())

    fig = plt.figure(figsize=(10,6))
    for col in classes:
        sns.distplot(dataframe[col], norm_hist=True, label=col)
    plt.xlabel("Normalized pixel values", fontsize=15)
    plt.ylabel("Number", fontsize=15)
    plt.legend()
    plt.savefig(str(parameters.path /'inputdata' /\
                    'normalized_pixel_distribution.png'))
    plt.show()

    fig = plt.figure(figsize=(15,10))
    corr = dataframe.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    plt.savefig(str(parameters.path /'inputdata' /\
                    'class_correlation.png'))
    plt.show()
