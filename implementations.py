# -*- coding: utf-8 -*-
import numpy as np

def standardize(tx):
    means = np.mean(tx, axis=0)

    derivation = np.std(tx, axis=0)

    mask_derivation_null = (derivation==0)

    if len(derivation[mask_derivation_null]) > 0 :
        print("Warning, some derivation are null : ", mask_derivation_null)
        derivation[mask_derivation_null]=1

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

def sigmoid(t):
    """apply sigmoid function on t."""
    threshold = 1e2

    max_t = max(t)
    min_t = min(t)

    if min_t < -threshold or max_t > threshold:
        #print("WARNING: risk of overflow in exp : max = ", max(t),"\tmin = ", min(t))

        t[t > threshold] = threshold
        t[t < -threshold] = -threshold
        #print("Limiting values: max 2 = ", max(t),"\tmin 2 = ", min(t))

    exp = np.exp(-t)
    #print("exp = ",exp.shape)

    sig = 1.0 / (1 + exp)
    return sig

def calculate_log_likelihood_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    threshold = 1e-9

    x_hat = tx.dot(w)
    pred = sigmoid(x_hat)

    mask_inf_to_min = (pred < threshold)
    mask_sup_to_max =(pred >= 1-threshold)

    if len(pred[mask_inf_to_min]) > 0 :
        pred[mask_inf_to_min] = threshold

    if len(pred[mask_sup_to_max]) > 0 :
        pred[mask_sup_to_max] = 1-threshold

    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    loss = -loss
    #print("loss = ", loss.shape, "\t", loss)
    #loss = np.squeeze(loss)
    return loss

def calculate_gradient_sigmoid(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    threshold = 1e-8
    losses = []

    for i in range(max_iters):

        loss = calculate_log_likelihood_loss(y, tx, w)
        print(loss)
        losses.append(loss)

        grad = calculate_gradient_sigmoid(y, tx, w)
        w -= gamma * grad

        # if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
        #     print("loss is not evolving, stopping the loop at iteration : ", i)
        #     break

        #print(calculate_log_likelihood_loss(y, tx, w))

    loss = calculate_log_likelihood_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError