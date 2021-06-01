# -*- coding: utf-8 -*-
"""
Main file of the deep learning based landcover classification challenge.

For more information, see project's GitLab repo:

    https://github.com/ThisismyEODev/DeepLearningLandcoverClassification.git


@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import argparse
import time

from loguru import logger

import parameter_file as parameters
from .folder_setup import setup_workspace
from .data_download import retrieve_data
from .data_preprocessing import input_data_preparation
from .data_exploration import plot_data_distribution_and_correlation

def main() -> None:

    """Run module from command line."""

    logger.add(f"logs/{time.strftime('%Y%m%d_%H%M%S')}.log", retention="10 days")
    logger.info("Starting landcover classification ...")
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="UP 42 Coding Challenge."
    )

    print("Setting up the workspace")
    setup_workspace()

    print("Access data and point to where it is located")
    data_foldername = retrieve_data()

    print("Split data")
    label_dictionary, classes,\
        X_train, y_train_encoded,\
            X_validation, y_validation_encoded,\
                X_test, y_test_encoded =\
                    input_data_preparation(data_foldername, parameters)
    
    print("Perform some basic data exploration")
    plot_data_distribution_and_correlation(classes, 
                                           X_train, y_train_encoded)
    plot_data_distribution_and_correlation(classes, 
                                           X_validation, y_validation_encoded)
    plot_data_distribution_and_correlation(classes, 
                                           X_test, y_test_encoded)
    


if __name__ == "__main__":
    main()

