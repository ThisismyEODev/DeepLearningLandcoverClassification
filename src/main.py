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
import numpy as np

from loguru import logger

import parameter_file as parameters
from .folder_setup import setup_workspace
from .data_download import retrieve_datafolder_name
from .data_preprocessing import input_data_preparation, encode_labels
from .data_exploration import plot_data_distribution_and_correlation
from .data_augmentation import augment_data
from .create_resnetmodel import (build_model, compile_and_fit_model, 
                                 compile_and_fit_model_from_generator)
from .model_evaluation import evaluate_model_accuracy
from .model_prediction import run_prediction_on_example_image

def main() -> None:

    """Run module from command line."""

    logger.add(f"logs/{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger.info("Starting landcover classification ...")
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="UP 42 Coding Challenge."
    )

    print("Setting up the workspace")
    setup_workspace()

    print("Retrieve folder name where input data is located")
    data_foldername = retrieve_datafolder_name()

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
        encode_labels(classes, y_train, y_test, y_validation)

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
        train_datagen, validation_datagen = augment_data(parameters, 
                                                   X_train, 
                                                   y_train, 
                                                   X_test, 
                                                   y_test)

        history = compile_and_fit_model_from_generator(parameters, model,
                                                       train_datagen,
                                                       validation_datagen,
                                                       save_model=parameters.save_model)

    
    print("Print and plot accuracy")
    accuracy = model.evaluate(X_test, y_test_encoded)
    evaluate_model_accuracy(parameters, model, history, X_test, y_test_encoded)
    
    print("Predict class of a random sample of the images")
    length_test_data = len(X_test)
    indexes_test_data = np.arange(length_test_data)
    ind = np.random.choice(indexes_test_data, 1, replace=False)
    run_prediction_on_example_image(model, classes, X_test, ind)

    # Report finishing module.
    logger.info(
        f"\n\nchange_detection_data_analysis finished in"
        f" {(time.time() - start_time)/60:.1f} minutes.\n"
    )
    

if __name__ == "__main__":
    main()

