# Machine Learning Project 1: Find the Higgs boson

This directory contains all code necessary to run our tests and get the model that we submitted on aicrowd. 

## Contributors

* Florian Grötschla
* Adrien Bertaud
* Maximilian Wessendorf

## Usage

To run the code that generates our predictions, first copy the the train and test datasets to /data/train.csv and /data/test.csv. It then suffices to run `run.py`, the predictions are then stored in result/result.csv.
The codebase is splitted into several files with different contents:

* run.py: To get our model
* compare_methods.ipynb: jupyter notebook that contains code we used for the evaluation
* evaluation.py: functions to do the evaluation, such as splitting the data, the cross validation and computing different metrics
* implementations.py: contains the 6 functions we had to implement and all functions that are necessary to execute them

