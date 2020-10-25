# Machine Learning Project 1: Find the Higgs boson

This directory contains all code necessary to rerun our tests and get the model that we submitted on [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).
## Contributors

* Florian Grötschla
* Adrien Bertaud
* Maximilian Wessendorf

## Usage

To run the code that generates our predictions, first copy the the train and test datasets to /data/train.csv and /data/test.csv. It then suffices to run `run.py`, the predictions are then stored in result/result.csv.
The codebase is splitted into several files with different contents:

* **run.py**: to get our model
* **compare_methods.ipynb**: Jupyter Notebook that contains code we used to run each method with various parameters
* **plot_lin_reg_GD.ipynb**: Jupyter Notebook to plot results with linear regression results using gradient descent
* **implementations.py**: contains the 6 methods we had to implement and all functions that are necessary to execute them
* **evaluation.py**: functions to do the evaluation, such as splitting the data, the cross validation and computing different metrics



