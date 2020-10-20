# -*- coding: utf-8 -*-
import numpy as np
from proj1_helpers import *

"""
Calculates the classification accuracy for a trained model on a given test set.
"""
def calculate_accuracy(w, x_test, y_test):
    y_pred = predict_labels(w, x_test)
    total_number_predictions = y_pred.size
    numer_correct_predictions = 0
    for i in range(y_pred.size):
        if y_pred[i] == y_test[i]:
            numer_correct_predictions += 1
    accuracy = numer_correct_predictions / total_number_predictions
    return accuracy

def calculate_f1_score(w, x_test, y_test):
    y_pred = predict_labels(w, x_test)
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for i in range(y_pred.size):
        if y_pred[i] == 1 and y_test[i] == 1:
            true_positives += 1
        if y_pred[i] == -1 and y_test[i] == -1:
            true_negatives += 1
        if y_pred[i] == 1 and y_test[i] == -1:
            false_positives += 1
        if y_pred[i] == -1 and y_test[i] == 1:
            false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score =  (2 * precision * recall) / (precision + recall)
    return f1_score

"""
split the dataset based on the split ratio. If ratio is 0.8
you will have 80% of your data set dedicated to training
and the rest dedicated to testing
"""
def split_data(x, y, ratio, seed=1):
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

"""
Group the dataset into k groups of training and test data.
Use this as a basis for a k-fold cross-validation.
"""
def group_data(x, y, groups, seed=1):
    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])
    split_indices = np.array_split(indices, groups)
    x_split = []
    y_split = []
    for split in split_indices:
        x_split.append(x[split])
        y_split.append(y[split])
    return x_split, y_split