import numpy as np
from split_data import split_data
from proj1_helpers import load_csv_data
from implementations import standardize, remove_rows_with_faulty_values

def initialize(data_train_path='data/train.csv', split_ratio = 0.8, data_test_path='data/test.csv', remove_rows=False):

    y, tX, ids = load_csv_data(data_train_path)
    y = np.expand_dims(y, axis=1)
    
    _, tX_test, ids_test = load_csv_data(data_test_path)
    
    if (remove_rows):
        tX = remove_rows_with_faulty_values(tX)
    
    # Standardize data
    x_tr, tX_test = standardize(tX, tX_test)

    x_tr, x_te, y_tr, y_te = split_data(tX, y, split_ratio)

    num_variables = x_tr.shape[1]

    w_initial = np.array(np.zeros(num_variables))
    w_initial = np.expand_dims(w_initial, axis=1)
    print("w = ", w_initial.shape)

    return x_tr, x_te, y_tr, y_te, tX_test, ids_test, w_initial
