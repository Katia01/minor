"""-----------------------------------------------------------------------------
Useful methods for permutation.
-----------------------------------------------------------------------------"""
import re
import os
import numpy as np
from itertools import permutations
#from . import utilities

def permut_value(value, reference_value):
    """
    Determine the value vector which is the more similar to the reference one when they have both the same length.

    - Input: Values vector
             Reference vector
    - Output: Ordered value vector
    """

    index_value = []
    difference_value = []
    v_difference_value = []
    length_value = len(value)
    j = 0

    perm = permutations(value,length_value)
    perm = list(perm)

    for i in list(perm):
        [int(k) for k in i]
        difference = np.abs(np.subtract(reference_value, i))
        s = int(np.sum(difference))
        v_difference_value.append(s)

    index_value = np.argmin(v_difference_value)

    for i in list(perm):
        if index_value == j:
            value = np.array(i)
            np.asarray(value)

        j = j+1

    return value

def permut_reference_value(value, reference_value):
    """
    Determine the reference vector which is the more similar to the value one when the reference vector is longer than the value one.

    - Input: Values vector
             Reference vector
    - Output: Ordered reference vector
    """

    index_value = []
    difference_value = []
    v_difference_value = []
    length_value = len(value)
    j = 0

    perm = permutations(reference_value,length_value)
    perm = list(perm)

    for i in list(perm):
        [int(k) for k in i]
        difference = np.abs(np.subtract(value, i))
        s = int(np.sum(difference))
        v_difference_value.append(s)

    index_value = np.argmin(v_difference_value)

    for i in list(perm):
        if index_value == j:
            reference_value = np.array(i)
            np.asarray(value)

        j = j+1

    return reference_value

def permut_value_add_zero(value, reference_value, difference_value):
    """
    Determine the value vector which is the more similar to the refencre one when the value vector is
    longer than the reference one. Add more 0 to the reference vector to have the same length as the value one.

    - Input: Values vector
             Reference vector
             Difference vector
    - Output: Ordered value vector
              Reference vector with the appropriate length
    """

    index_value = []
    difference_value = []
    v_difference_value = []
    value_bis = []
    length_reference_value = len(reference_value)
    value_bis = value
    j = 0

    perm = permutations(value,length_reference_value)
    perm = list(perm)

    for i in list(perm):
        [int(k) for k in i]
        difference = np.abs(np.subtract(reference_value, i))
        s = int(np.sum(difference))
        v_difference_value.append(s)

    index_value = np.argmin(v_difference_value)
    for i in list(perm):
        if index_value == j:
            value = np.array(i)
            np.asarray(value)

        j = j+1

    for i in range(len(value_bis)):

        if value_bis[i] not in value:
            value = np.append(value,value_bis[i])

        count_value = utilities.countX(value, value_bis[i])
        count_value_bis = utilities.countX(value_bis, value_bis[i])

        if count_value < count_value_bis:
            value = np.append(value,value_bis[i])

    zero = np.zeros(difference_value)
    reference_value = np.append(reference_value, zero)
    reference_value = reference_value.astype(int)

    return value, reference_value
