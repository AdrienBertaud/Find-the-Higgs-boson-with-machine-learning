# -*- coding: utf-8 -*-
import numpy as np

def standardize(tx):
    means = np.mean(tx, axis=0)
    derivation = np.std(tx, axis=0)
    tx = (tx - means) / derivation
    return tx

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return 1/(2*len(y)) * (e.dot(e))

def compute_gradient(y, tx, w):
    return -1/len(y) * np.transpose(tx).dot(y - tx.dot(w))

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w) 
        w = w - gamma * gradient
    loss = compute_loss(y, tx, w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=2, num_batches=1):
            gradient = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * gradient
    loss = compute_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    w = np.linalg.solve(np.transpose(tx).dot(tx) + (lambda_ * 2 * len(y) * np.identity(tx.shape[1])), np.transpose(tx).dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError