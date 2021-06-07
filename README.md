# DeepLearningLandcoverClassification
This is the solution of a coding challenge
The module is a simple implementation of a landcover classification using Transfer Learning via ResNet50.

## Usage

### Cloning This Repo

```bash
git clone https://github.com/ThisismyEODev/DeepLearningLandcoverClassification.git
```

### Running the code

To run the code, simply go to the src directory and run "python main.py".
This should load all the configurations set in the "parameter_file.py" file.
Note that the "path" parameter should point to the directory in which the eurosat data is stored
(see Known Issues below).
The parameter file contains, e.g. 
- the steps which should be run (0: load data, split data, get some first level statistics; 
1: build model and run training; 2: run prediction and read out some accuracy measures 
and comparison of predicted vs. true label)
- The percentage of training and validation / testing data to build and test the model
- ResNet50 model parameters, e.g. optimizer, learning rate, etc.
If any other of the parameters should be different, simply change them in the "parameter_file.py".


## Known Issues

- I had issues with downloading the data directly via tensorflow_dataset. 
The module currently only runs on previously downloaded files.
Therefore, the path variable in the "parameter_file.py" should be set to the
path where the data is stored

- Running the environment yml file on my laptop, a condaenvexception is printed as pip fails


## Possible TODOs

- Fix environment creation error (Pip)
- Try out other layer configurations / data augmentation to improve test accuracy
- Direct download from tensorflow_dataset
- Apply unit testing
- Increase function / input flexibility
- Implement functionality for dealing with all S2 bands
- Dockerize
- Optimize data read / model training
- Apply grid search for hyperparameter tuning




