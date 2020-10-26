# -*- coding: utf-8 -*-
import numpy as np

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])
    split_index = int(ratio * x.shape[0])
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    train_x = x[train_indices]
    test_x = x[test_indices]
    train_y = y[train_indices]
    test_y = y[test_indices]
    return train_x, test_x, train_y, test_y