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
If any other of the parameters should be different, simply change them in the "parameter_file.py".


## Known Issues

- I had issues with downloading the data directly via tensorflow_dataset. 
The module currently only runs on previously downloaded files.
Therefore, the path variable in the "parameter_file.py" should be set to the
path where the data is stored

- Another issue is that when running *proba = model_resnet.predict(X_test_balanced[1, :, :, :])[0]*
there is an error message concerning the shape of what is expected vs. what is put in. The root cause
of this error still has to be found.

## TODOs

- Fix data download issue
- Fix expected prediction shape error
- Apply unit testing
- Increase function / input flexibility
- Test augmentation since it hasn't been tested yet
- Implement functionality for dealing with all S2 bands
- Dockerize
- If possible, optimize data read
- If possible, optimize model running
- Apply grid search for hyperparameter tuning




