# -*- coding: utf-8 -*-
import numpy as np
import sys

sys.path.append("..") # Adds higher directory to python modules path.

from implementations import standardize

x = np.array([[0, 3, 10],
            [2, 9, 10],
            [0, 3, 10],
            [-999, -999, -999],
            [2, 9, 10]])

x_2 = np.array([[2, 9, -999],
                [2, 9, 10],
                [0, 3, 10],
                [-999, -999, -999],
                [2, 9, 10]])

x_expected = np.array([[-1, -1,  0],
                   [ 1,  1,  0],
                   [-1, -1,  0],
                   [ 0,  0,  0],
                   [ 1,  1,  0]])

x_expected_2 = np.array([[1, 1,  0],
                       [ 1,  1,  0],
                       [-1, -1,  0],
                       [ 0,  0,  0],
                       [ 1,  1,  0]])

def result(value):
    if value == True:
        print("TEST OK :)")
    else:
        print("TEST KO :(")


def standardize_unit_test_method_1(x, x_expected):

    print("standardize_unit_test_method_1")
    means, derivations = get_standardization_values(x)
    x_std = apply_standardization(x, means, derivations)
    print("x_std = ", x_std)

    result((x_std == x_expected).all())


def standardize_unit_test_method_2(x, x_2, x_expected, x_expected_2):

    print("standardize_unit_test_method_2")
    x_std_1, x_std_2 = standardize(x,x_2)

    print("x_std_1 = ", x_std_1)
    print("x_std_2 = ", x_std_2)

    result((x_std_1 == x_expected).all())
    result((x_std_2 == x_expected_2).all())

standardize_unit_test_method_2(x, x_2, x_expected, x_expected_2)