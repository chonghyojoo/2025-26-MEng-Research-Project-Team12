# 2025-26-MEng-Research-Project-Team12
## ----- Setup -----
### 1) Install pytorch on GPU (windows)
- ref: https://www.youtube.com/watch?v=d_jBX7OrptI

### 2) You can also install all libraries I had ('binary_test.yml')
- anaconda prompt -> 'conda env create -f binary_test.yml'

### 3) Files you can run
- 'preprocessing/data_custom.ipynb': data preprocessing
- 'train_model.ipynb': model training
- 'test_model.ipynb': predict Gibbs E and enthalpy at 298K -> calculate Gibbs E at different T

### 4) Download datasets
- Train dataset: BinarySolvGH-QM.csv
- - ref: https://zenodo.org/records/14238055
- Final test dataset: Data and predictions of solvation free energies in binary solvents (BinarySolv-Exp).csv
- - ref: https://doi.org/10.1016/j.cej.2025.162232
### 5) Create 'data' folder
- After creating 'data' folder, move the all datasets to this path

## ----- To-do list -----
### 1) Data preprocessing
- Run 'data_custom.ipynb' to preprocess the train and test datasets
- The preprocessed datasets should be saved in the same path with 'train_model.ipynb' and 'test_model.ipynb'
### 2) Complete the 'test_model.ipynb' to calculate Gibbs E in different temperature
- You can predict Gibbs E and enthalpy at 298K via 'test_model.ipynb'
- Complete the final code to predict Gibbs E at different temperature
### 3) Check the pipline
- Run 'train_model.ipynb' using the preprocessed 'BinarySolvGH-QM.csv' with default hyperparameters
- Run 'test_model.ipynb' using the preprocessed 'Data and predictions of solvation free energies in binary solvents (BinarySolv-Exp).csv'
### 4) Hyperparameter optimization (HPO)
- Optimize the hyperparameters of the models to minimize its loss
- You can find the main hyperparameters by searching 'hyperparameter' in 'train_model.ipynb'
