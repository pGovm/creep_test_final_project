# creep_test_final_project

## Dataset
The dataset for this project can be located using this [Link to Dataset](https://zenodo.org/records/10143352). Scroll down and download the zip file that is provided to access the dataset.
This dataset contains 20 test cases of material that went under a Full-Notch Creep Test(FNCT) to predict the material's elongation behaviors.

## Project Description
This project was to take the above dataset and train it on 5 different architectures (CNN, vanilla RNN, LSTM, GRU, and encoder-only Transformer) to do a comparative analysis between them to see which one would be able to predict the output more accurately. We use the same hyperparameters between all the models to ensure a fair comparison between them. This repository contains 3 main scripts:
- **final_project_data_preprocessing_function_7_29_2026.py**: This python script handles the data preprocessing of the dataset
- **architecture.py**: This file contains the architecture of all 5 models as well as the hyperparameters used for each of them.
- **training_testing_function.py**: This file handles the training of all 5 models and runs evaluations on them using the Root Mean Squared Error, Mean Absolute Error, and Coefficient of Determination.

The remaining scripts are main functions calling each individual architecture for training and testing on the dataset. Finally, the creep_data folder contains the results of all the trained models in the form of:
- Training and Validation loss curves
- A CSV file that saves the evaluation results of all the models
- The weights of each trained file.
