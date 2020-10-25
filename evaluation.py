# -*- coding: utf-8 -*-
import numpy as np
from proj1_helpers import *
from implementations import *
import random

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
    return f1_score, precision, recall, true_positives, false_positives, true_negatives, false_negatives

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


def equalize_true_false(x, y):
    mask_y_false = np.squeeze(np.array(y <= 0))
    mask_y_true = np.squeeze(np.array(y > 0))

    y_false = y[mask_y_false]
    y_true = np.squeeze(y[mask_y_true])

    x_true = x[mask_y_true,:]
    x_false = x[mask_y_false,:]

    diff = len(y_false)-len(y_true)

    indexes_to_remove = random.sample(range(1, len(y_false)), diff)

    x_false_equal = np.array(np.delete(x_false, indexes_to_remove,0))

    y_false_equal = np.array(np.delete(y_false, indexes_to_remove))

    x_equal = np.concatenate((x_true, x_false_equal))
    y_equal = np.concatenate((y_true, y_false_equal))

    p = np.random.permutation(len(x_equal))
    x_permut = x_equal[p]
    y_permut = y_equal[p]

    return x_permut, y_permut

"""
Execute a k-fold cross validation on a given dataset with a given method and given parameters.
"""
def cross_val(tX, y, equalize = false, splits, poly_degree, method, **kwargs):
    # create dict with results for run
    cv_result = {}
    cv_result['method'] = 'least_squares_GD'
    cv_result['parameters'] = kwargs
    cv_result['train_losses'] = []
    cv_result['test_losses'] = []
    cv_result['accuracies'] = []
    cv_result['f1_scores'] = []
    cv_result['precisions'] = []
    cv_result['recalls'] = []
    cv_result['confusion_matrices'] = []

    x_split, y_split = group_data(tX, y, splits)
    for i in range(len(x_split)):
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        cv_run_result = {}
        for j in range(len(x_split)):
            if i == j:
                x_test.append(x_split[i])
                y_test.append(y_split[i])
            else:
                x_train.append(x_split[i])
                y_train.append(y_split[i])

        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)

        if equalize == True:
            x_train, y_train = equalize_true_false(x_train, y_train)

        x_train = build_poly(x_train, poly_degree)
        x_test = build_poly(x_test, poly_degree)

        x_train, x_test = standardize(x_train, x_test)
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

        w, loss_train = method(y_train, x_train, **kwargs)
        loss_test = compute_loss(y_test, x_test, w)

        cv_result['train_losses'].append(loss_train)
        cv_result['test_losses'].append(loss_test)
        cv_result['accuracies'].append(calculate_accuracy(w, x_test, y_test))
        f1_score, precision, recall, tp, fp, tn, fn = calculate_f1_score(w, x_test, y_test)
        cv_result['f1_scores'].append(f1_score)
        cv_result['precisions'].append(precision)
        cv_result['recalls'].append(recall)
        confusion_matrix = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        cv_result['confusion_matrices'].append(confusion_matrix)

    cv_result['mean_train_loss'] = np.array(cv_result['train_losses']).mean()
    cv_result['std_train_loss'] = np.array(cv_result['train_losses']).std()
    cv_result['mean_test_loss'] = np.array(cv_result['test_losses']).mean()
    cv_result['std_test_loss'] = np.array(cv_result['test_losses']).std()
    cv_result['mean_accuracy'] = np.array(cv_result['accuracies']).mean()
    cv_result['std_accuracy'] = np.array(cv_result['accuracies']).std()
    cv_result['mean_f1_score'] = np.array(cv_result['f1_scores']).mean()
    cv_result['std_f1_score'] = np.array(cv_result['f1_scores']).std()
    cv_result['mean_precision'] = np.array(cv_result['precisions']).mean()
    cv_result['std_precision'] = np.array(cv_result['precisions']).std()
    cv_result['mean_recall'] = np.array(cv_result['recalls']).mean()
    cv_result['std_recall'] = np.array(cv_result['recalls']).std()

    return cv_result