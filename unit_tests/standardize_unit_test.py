# -*- coding: utf-8 -*-
import numpy as np
import sys

sys.path.append("..") # Adds higher directory to python modules path.

from implementations import standardize

def standardize_unit_test():

    x = np.array([[0, 3, 10],
                    [2, 9, 10],
                    [0, 3, 10],
                    [-999, -999, -999],
                    [2, 9, 10]])

    x = standardize(x)

    print(x)

    x_expected = np.array([[-1, -1,  0],
                           [ 1,  1,  0],
                           [-1, -1,  0],
                           [ 0,  0,  0],
                           [ 1,  1,  0]])

    if (x == x_expected).all():
        print("TEST OK :)")
    else:
        print("TEST KO :(")

standardize_unit_test()