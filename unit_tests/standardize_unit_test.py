# -*- coding: utf-8 -*-
import numpy as np
import sys

sys.path.append("..") # Adds higher directory to python modules path.

from implementations import get_standardization_values, apply_standardization

def standardize_unit_test():

    x = np.array([[0, 3, 10],
                    [2, 9, 10],
                    [0, 3, 10],
                    [-999, -999, -999],
                    [2, 9, 10]])

    means, derivations = get_standardization_values(x)
    x = apply_standardization(x, means, derivations)

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