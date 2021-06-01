# -*- coding: utf-8 -*-
"""
This is the configuration file for running the module

@author: ThisismyEODev
@email: nastasja.development@gmail.com
"""

from keras.optimizers import RMSprop, Adam, SGD

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

# Set to True, if you want to download the datasets via the tensorflow_datasets package
download = False

# Switch between RGB or all, depending on if you want to use only RGB or all spectral bands
spectral_bands = "RGB"
data_folder = input_folder_names[spectral_bands]
data_format = input_format[spectral_bands]

img_size = 64
n_bands = number_of_bands[spectral_bands]


##### Train Test Validation Setup ###########

seed = 42
training_size = .7
testing_size = .3
balanced = True

augment = False
shear = 0.2
zoom = 0.2
rotation = 30


##### RESNET50 Model Setup ###########

w = 'imagenet' # Weights for the ResNet50 Model

epochs = 10
batch_size = 32
target_size = (224, 224)
learning_rate = 0.01 # 0.01, 0.001, 0.0001

# Change here if you want to test other optimizers
optimizer_name = "sgd"
if optimizer_name == "sgd":
    optimizer = SGD()
elif optimizer_name == "adam":
    optimizer = Adam()
elif optimizer_name == "rms":
    optimizer = RMSprop()

model_metric = ['accuracy']
loss_function = 'categorical_crossentropy'

save_model = False
model_name = f"Resnet50_{spectral_bands}_{optimizer_name}.h5"