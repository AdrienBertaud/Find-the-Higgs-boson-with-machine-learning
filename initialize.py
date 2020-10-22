# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from split_data import split_data
from proj1_helpers import *
from implementations import *
import datetime

def initialize():

    DATA_TRAIN_PATH = 'data/train500.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    y = np.expand_dims(y, axis=1)

    split_ratio = 0.5
    x_tr, x_te, y_tr, y_te = split_data(tX, y, split_ratio)


    # x_tr = np.array([[1.00368, 3.58310, 7.71850],
    #                 [7.66620, 4.15970, 5.76100]])

    # y_tr = np.array([[ 1.],
    #                  [-1.]])

    # y_te = 0
    # x_te = 0

    print("x_tr = ", x_tr.shape)
    num_variables = x_tr.shape[1]

    # Initialization
    w_initial = np.array(np.zeros(num_variables))
    print("w = ", w_initial.shape)
    w_initial = np.expand_dims(w_initial, axis=1)
    print("w = ", w_initial.shape)

    # Standardize data
    x_tr = standardize(x_tr)

    return x_tr, x_te, y_tr, y_te, w_initial

#initialize()