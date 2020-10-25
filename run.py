# -*- coding: utf-8 -*-
import numpy as np
from proj1_helpers import *
from implementations import *

# LOAD LABELED DATA
DATA_TRAIN_PATH = 'data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# LOAD UNLABELED DATA
DATA_TEST_PATH = 'data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# PREPROCESSING

## Polynomial expansion
tX = build_poly(tX, 14)
tX_test = build_poly(tX_test, 14)

## Standardization
tX, tX_test = standardize(tX, tX_test)

## Adding column of ones
tX = np.hstack((tX, np.ones((tX.shape[0], 1))))
tX_test = np.hstack((tX_test, np.ones((tX_test.shape[0], 1))))

# FITTING MODEL
w, loss = least_squares(y, tX)

# EXPORT PREDICTIONS
OUTPUT_PATH = 'results/result.csv'
y_pred = predict_labels(w, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
