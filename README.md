# 2025-26-MEng-Research-Project-Team12
## ----- Setup -----
### 1) install pytorch on GPU (windows)
- ref: https://www.youtube.com/watch?v=d_jBX7OrptI

### 2) you can also install all libraries I had ('binary_test.yml')
- anaconda prompt -> 'conda env create -f binary_test.yml'

### 3) files you can run
- 'train_model.ipynb': model training
- 'test_model.ipynb': predict Gibbs E and enthalpy at 298K -> calculate Gibbs E at different T

## ----- To-do list -----
### 1) complete the 'test_model.ipynb' to calculate Gibbs E in different temperature
### 2) hyperparameter optimization (HPO)
- optimize the hyperparameters of the models to minimize its loss
- you can find hyperparameters by searching 'hyperparameter' in 'train_model.ipynb'
### 3) train ML model using 'BinarySolvGH-QM.csv' --> test the model using 'Data and predictions of solvation free energies in binary solvents (BinarySolv-Exp).csv'
