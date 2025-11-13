# 2025-26-MEng-Research-Project-Team12
## ----- Setup -----
### 1) install pytorch on GPU (windows)
- ref: https://www.youtube.com/watch?v=d_jBX7OrptI

### 2) you can also install all libraries I had ('binary_test.yml')
- anaconda prompt -> 'conda env create -f binary_test.yml'

### 3) files you can run
- 'preprocessing/data_custom.ipynb': data preprocessing
- 'train_model.ipynb': model training
- 'test_model.ipynb': predict Gibbs E and enthalpy at 298K -> calculate Gibbs E at different T

## ----- To-do list -----
### 1) download datasets
- Train dataset: BinarySolvGH-QM.csv
- - ref: https://zenodo.org/records/14238055
- Final test dataset: Data and predictions of solvation free energies in binary solvents (BinarySolv-Exp).csv
- - ref: https://doi.org/10.1016/j.cej.2025.162232
### 2) create 'data' folder
- After creating 'data' folder, move the all datasets to this path
### 3) data preprocessing
### 4) complete the 'test_model.ipynb' to calculate Gibbs E in different temperature
### 5) hyperparameter optimization (HPO)
- optimize the hyperparameters of the models to minimize its loss
- you can find hyperparameters by searching 'hyperparameter' in 'train_model.ipynb'
### 6) train ML model using 'BinarySolvGH-QM.csv' --> test the model using 'Data and predictions of solvation free energies in binary solvents (BinarySolv-Exp).csv'
