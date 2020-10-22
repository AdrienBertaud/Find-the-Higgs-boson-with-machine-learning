import importlib
import initialize
importlib.reload(initialize)
import numpy as np
import matplotlib.pyplot as plt
from split_data import split_data
from proj1_helpers import *
from implementations import *
import datetime

from initialize import *

x_tr, x_te, y_tr, y_te, w_initial = initialize()

max_iters = 1000
gamma = 0.05

# Logistic regression
start_time = datetime.datetime.now()
print(y_tr.shape)
y_tr[y_tr==-1]=0
print(y_tr.shape)
w, loss = logistic_regression(y=y_tr, tx=x_tr, initial_w=w_initial, max_iters=max_iters, gamma=gamma)
print(loss)
end_time = datetime.datetime.now()
print('== Logistic regression ==')
print("loss: " + str(loss) + ', Time: ' + str((end_time - start_time).total_seconds()) + 's')