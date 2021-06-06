# -*- coding: utf-8 -*-
"""
This is the configuration file for running the module

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

import tensorflow as tf
from keras.optimizers import Adam, SGD

####### FIXED PARAMETERS - Please to not change #######

input_folder_names = dict()
input_folder_names["RGB"] = "2750"
input_folder_names["all"] = "tif"

input_format = dict()
input_format["RGB"] = ".jpg"
input_format["all"] = ".tif"

number_of_bands = dict()
number_of_bands["RGB"] = 3
number_of_bands["all"] = 13

####### PARAMETER CONFIGURATION 

# Path to working directory
path = "C:/UP42_Challenge"

##### Data access Setup ###########

download = False

# Switch between RGB or all, depending on if you want to use only RGB or all spectral bands
spectral_bands = "RGB"
data_folder = input_folder_names[spectral_bands]
data_format = input_format[spectral_bands]
n_bands = number_of_bands[spectral_bands]

# This is the percentage of training data per label!
seed = 42
perc_training = .9
perc_testing = .1
img_size = 64
image_size = (img_size, img_size, n_bands)

# Set to True if you want to work with balanced classes
balance = False

# Set to True if you want to work augment your data before you run the model
augment = False
shear = 0.2
zoom = 0.2
rotation = 30
width_shift = 0.2
height_shift = 0.2
fill = 'nearest'

##### RESNET50 Model Setup ###########
w = 'imagenet' # Weights for the ResNet50 Model
freeze_layers = True

epochs = 10
batch_size = 32
target_size = (224, 224)
learning_rate = 0.001 # 0.01, 0.001, 0.0001

optimizer_name = "adam"
if optimizer_name == "adam":
    optimizer = Adam(lr=learning_rate)
elif optimizer_name == "sgd":
    optimizer = SGD(lr=learning_rate, momentum=.9, nesterov=False)

loss = "not_sparse"
if loss == "sparse":
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
elif loss == "not_sparse":
    loss_function = tf.keras.losses.CategoricalCrossentropy()

model_metric = ['accuracy']

# parameters for saving results and inbetween steps
run_number = 1

callback_file_name = f"Callbacks_ResNet50_{spectral_bands}_trainsize_{1-perc_training}\
_augmentation_{augment}_balanced_{balance}_num_epochs_{epochs}\_optimizer_{optimizer_name}_run_{run_number}.h5"

model_file_name = f"Model_ResNet50_{spectral_bands}_trainsize_{1-perc_training}\
_augmentation_{augment}_balanced_{balance}_num_epochs_{epochs}\_optimizer_{optimizer_name}_run_{run_number}.h5"

model_plot_name = f"Model_ResNet50_{spectral_bands}_trainsize_{1-perc_training}\
_augmentation_{augment}_balanced_{balance}_num_epochs_{epochs}\_optimizer_{optimizer_name}_run_{run_number}.png"

