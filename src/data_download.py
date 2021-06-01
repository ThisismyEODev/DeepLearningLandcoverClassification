# -*- coding: utf-8 -*-
"""

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import tensorflow_datasets as tfds
import parameter_file as parameters


def retrieve_data(parameters):
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

def main() -> None:
    retrieve_data(parameters)


if __name__ == "__main__":
    main()

