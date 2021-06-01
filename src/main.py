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
from .data_preprocessing import input_data_preparation, encode_labels
from .data_exploration import plot_data_distribution_and_correlation
from .data_augmentation import augment_data
from .create_resnetmodel import (build_model, compile_and_fit_model, 
                                 compile_and_fit_model_from_generator)
from .model_evaluation import evaluate_model_accuracy

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
        X_train, y_train,\
            X_validation, y_validation,\
                X_test, y_test =\
                    input_data_preparation(data_foldername, parameters)
    
    print("Perform some basic data exploration")
    plot_data_distribution_and_correlation(classes, 
                                           X_train, y_train)
    plot_data_distribution_and_correlation(classes, 
                                           X_validation, y_validation)
    plot_data_distribution_and_correlation(classes, 
                                           X_test, y_test)

    print("Encode labels")
    y_train_encoded, y_test_encoded, y_validation_encoded=\
        encode_labels(y_train, y_test, y_validation)

    print("Build ResNet50 model with some extra layers")
    model = build_model(parameters, X_train, y_train_encoded)
    print(model.summary())
    
    if parameters.augment == False:
        print("We leave the data as is", "\n")

        print("Compile and fit model")
        history = compile_and_fit_model(parameters, model,
                                    X_train, y_train_encoded,
                                    X_validation, y_validation_encoded,
                                    save_model=parameters.save_model)

    elif parameters.augment == True:
        print("We apply some variations to the data", "\n")
        train_datagen, test_datagen = augment_data(parameters, 
                                                   X_train, 
                                                   y_train, 
                                                   X_test, 
                                                   y_test)

        history = compile_and_fit_model_from_generator(parameters, model,
                                                       train_datagen,
                                                       X_train,
                                                       test_datagen,
                                                       X_validation,
                                                       save_model=parameters.save_model)

    
    print("Plot accuracy")
    evaluate_model_accuracy(model, history, X_test, y_test)
    
if __name__ == "__main__":
    main()

