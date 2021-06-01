# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:37:48 2021

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

####### FIXED PARAMETERS - Please to not change #######

input_folder_names = dict()
input_folder_names["RGB"] = "2750"
input_folder_names["all"] = "tif"

input_format = dict()
input_format["RGB"] = ".jpg"
input_format["all"] = ".tif"


####### PARAMETER CONFIGURATION 

# Path to working directory
path = "C:/UP42_Challenge"

##### Data access Setup ###########

# Set to True, if you want to download the datasets via the tensorflow_datasets package
download = False

# Switch between RGB or all, depending on if you want to use only RGB or all spectral bands
spectral_bands = "RGB"
data_folder = input_folder_names[spectral_bands]
data_format = input_format[spectral_bands]

##### Train Test Validation Setup ###########
# This is the percentage of training data per label!

seed = 42
training_size = .7
testing_size = .3


##### RESNET50 Model Setup ###########
w = 'imagenet' # Weights for the ResNet50 Model

epochs = 10
batch_size = 32
target_size = (224, 224)
learning_rate = 0.01 # 0.01, 0.001, 0.0001

# Change here if you want to test other optimizers
model_metric = ['accuracy']
loss_function = 'categorical_crossentropy'

