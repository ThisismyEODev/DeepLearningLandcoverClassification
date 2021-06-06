# -*- coding: utf-8 -*-
"""

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import tensorflow_datasets as tfds

def retrieve_datafolder_name(parameters):
    """
    Get data based on download True or False
    Loads data into folder location given in "parameters"
    Returns name of the folder where data is located

    Parameters
    ----------
    parameters: 
        Parameters set in the src/parameter_file.py

    Returns
    -------
    data_foldername: 
        str
        Folder name where data is located            

    """
    if parameters.download == True:
        print("Download EuroSAT dataset", "\n")
        data_path = parameters.path / 'inputdata'
        train_data, validation_data, test_data = tfds.load(
            name="eurosat",
            split=[
                tfds.Split.TRAIN.subsplit\
                    (tfds.percent[:int(parameters.perc_training*100)]),
                tfds.Split.TEST.subsplit\
                    (tfds.percent[int(parameters.perc_testing*100):]),
                'test'
                ],
            as_supervised=True,
            data_dir = data_path)
        data_foldername = "eurosat"

    else:
        print("We are working with a previously downloaded dataset", "\n")
        data_foldername = parameters.data_folder
    
    return data_foldername
