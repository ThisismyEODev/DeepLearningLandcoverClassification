# -*- coding: utf-8 -*-
"""

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import tensorflow_datasets as tfds

def retrieve_data(parameters):
    """
    Get data based on download True or False
    Loads data into folder location given in "parameters"

    parameters:     input parameter which is automatically loaded into the main.py
                    file
                
    Returns:
        
    data_frame:     Name of folder in which data are now located
                    str
    """
    if parameters.download == True:
        print("Download EuroSAT dataset", "\n")
        data_path = parameters.path / 'inputdata'
        train_data, validation_data, test_data = tfds.load(
            name="eurosat",
            split=[
                tfds.Split.TRAIN.subsplit\
                    (tfds.percent[:int(parameters.training_size*100)]),
                tfds.Split.TRAIN.subsplit\
                    (tfds.percent[int(parameters.training_size*100):]),
                'test'
                ],
            as_supervised=True,
            data_dir = data_path)
        data_foldername = "eurosat"

    else:
        print("We are working with a previously downloaded dataset", "\n")
        data_foldername = parameters.data_folder
    
    return data_foldername
