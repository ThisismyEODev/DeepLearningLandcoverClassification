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
from sklearn import metrics
from loguru import logger

import parameter_file as parameters
from folder_setup import setup_workspace
from data_download import retrieve_datafolder_name
from data_preprocessing import input_data_preparation, encode_labels
from data_exploration import plot_data_distribution_and_correlation
from data_augmentation import augment_data
from create_resnetmodel import (build_model, compile_and_fit_model, 
                                 compile_and_fit_model_from_generator)
from model_evaluation import plot_model_accuracy
from model_prediction import (make_prediction, 
                               predict_on_single_testimage_and_score,
                               create_prediction_dataframe,
                               dataframe_of_accurate_and_nonaccurate_prediction,
                               plot_confusion_matrix,
                               plot_model_roc_curve)

def main() -> None:

    """Run module from command line."""

    logger.add(f"logs/{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger.info("Starting landcover classification ...")
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="UP 42 Coding Challenge."
    )

    print("Setting up the workspace")
    setup_workspace(parameters)

    print("Retrieve folder name where input data is located")
    data_foldername = retrieve_datafolder_name(parameters)

    print("Split data")
    label_dictionary, classes,\
        X_train, y_train,\
            X_validation, y_validation,\
                X_test, y_test =\
                    input_data_preparation(data_foldername, parameters)
    
    print("Perform some basic data exploration")
    plot_data_distribution_and_correlation(parameters, classes, 
                                           X_train, y_train, "train")
    plot_data_distribution_and_correlation(parameters, classes, 
                                           X_validation, y_validation, "validation")
    plot_data_distribution_and_correlation(parameters, classes, 
                                           X_test, y_test, "test")

    if parameters.step == "Load_and_data_explo":
        return

    elif parameters.step == "Model_training":

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

            train_generator, validation_generator = augment_data(parameters, 
                                                   X_train, 
                                                   y_train_encoded, 
                                                   X_validation, 
                                                   y_validation_encoded)

            history = compile_and_fit_model_from_generator(parameters, model,
                                                           train_generator,
                                                           validation_generator,
                                                           save_model=parameters.save_model)
        return

    elif parameters.step == "Full_pipeline":
    
        print("Print and plot accuracy")
        accuracy = model.evaluate(X_test, y_test_encoded)
        plot_model_accuracy(model, history, parameters.epoch)
    
        print("Run prediction")
        predicted_labels, y_test_true, y_test_pred, y_pred_encoded = \
                make_prediction(parameters,model, X_test, y_test, 
                                label_dictionary)

        print('Accuracy:', np.round(metrics.accuracy_score(y_test_true, 
                                                           y_test_pred), 4))
        print('Precision:', np.round(metrics.precision_score(y_test_true, 
                                                             y_test_pred,
                                                             average='weighted'), 
                                                             4))
        print('Recall:', np.round(metrics.recall_score(y_test_true, 
                                                       y_test_pred,
                                                       average='weighted'), 4))
        print('F1 Score:', np.round(metrics.f1_score(y_test_true, y_test_pred,
                                               average='weighted') ,4))

        predict_on_single_testimage_and_score(parameters, model, X_test, y_test, 
                                              classes)
        predict_df = create_prediction_dataframe(parameters, 
                                                 y_test_true, y_test_pred, 
                                                 len(classes))
        print(predict_df)

        accuracy_of_predict_df =\
                dataframe_of_accurate_and_nonaccurate_prediction(parameters,
                                                                 y_test_true, 
                                                                 y_test_pred)

        plot_confusion_matrix(parameters, y_test_true, y_test_pred, classes)
        plot_model_roc_curve(parameters, model, y_test_encoded, y_pred_encoded, 
                             classes)


    logger.info(
        f"\n\nchange_detection_data_analysis finished in"
        f" {(time.time() - start_time)/60:.1f} minutes.\n"
    )
    

if __name__ == "__main__":
    main()

