# -*- coding: utf-8 -*-
import numpy as np
import sys

sys.path.append("..") # Adds higher directory to python modules path.

from evaluation import equalize_true_false

x = np.array([[0, 3, 10],
            [-2, -9, -10],
            [0, -3, -10],
            [-999, -999, -999],
            [2, 9, 10]])

y = np.array([[1],
            [0],
            [0],
            [-1],
            [1]])

x_eq, y_eq = equalize_true_false(x,y)

print("x_eq : ", x_eq)
print("y_eq : ", y_eq)