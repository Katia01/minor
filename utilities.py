"""-----------------------------------------------------------------------------
Created on October 2019

@author: Katia

Useful methods
-----------------------------------------------------------------------------"""
import re
import os
import numpy as np
from itertools import permutations

from . import permute

def sorted_aphanumeric(data):

    """
    Order in an ascending manner a list of pictures according to their name.

    - Input: List of pictures
    - Output: Ordered list of pictures
    """

    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

    return sorted(data, key = alphanum_key)

def init_references(number_subjects):

    """
    Initializes nul reference vectors with the length of the first value vectors.

    - Input: Number of subjects present on the picture
    - Output: Three nul reference vectors
    """

    reference_radius = [0] * number_subjects
    reference_center_x = [0] * number_subjects
    reference_center_y = [0] * number_subjects

    return reference_radius, reference_center_x, reference_center_y

def variation_prediction(radius, reference_radius, center_x, reference_center_x, center_y, reference_center_y):

    """
    Compute difference between value vectors and reference ones.

    - Input: Three values vectors
             Three reference vectors
    - Output: Three difference vectors
    """

    variation_radius = np.abs(np.subtract(radius, reference_radius))
    variation_center_x = np.abs(np.subtract(center_x, reference_center_x))
    variation_center_y = np.abs(np.subtract(center_y, reference_center_y))

    return variation_radius, variation_center_x, variation_center_y

def difference_prediction(radius, reference_radius, center_x, reference_center_x, center_y, reference_center_y):

    """
    Compute difference of size between value vectors and reference ones.

    - Input: Three values vectors
             Three reference vectors
    - Output: Three size difference vectors
    """

    difference_radius = len(radius)-len(reference_radius)
    difference_center_x = len(center_x)-len(reference_center_x)
    difference_center_y = len(center_y)-len(reference_center_y)

    return difference_radius, difference_center_x, difference_center_y

def countX(lst, x):

    """
    Count the number of time a specific value is present on a vector.

    - Input: Values vector
             Specific value
    - Output: Final count
    """

    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

def ordered_prediction(radius, reference_radius, center_x, reference_center_x, center_y, reference_center_y):

    """
    Order values to be sure to compare them with their specific reference.

    - Input: Three values vectors
              Three reference vectors
    - Output: Three ordered values vectors
    """
    radius = permute.permut_value(radius, reference_radius)
    center_x = permute.permut_value(center_x, reference_center_x)
    center_y = permute.permut_value(center_y, reference_center_y)

    return radius, center_x, center_y

def adjust_length(radius, reference_radius, difference_radius, center_x,  reference_center_x, difference_center_x,\
center_y, reference_center_y, difference_center_y):

    """
    Delete some values of the reference vectors if they are longer than the value vectors.
    - Input: Three values vectors
             Three reference vectors
             Three difference vectors
    - Output: Three reference vectors which have now the same length than the value vectors
    """

    reference_radius = permute.permut_reference_value(radius, reference_radius)
    reference_center_x = permute.permut_reference_value(center_x, reference_center_x)
    reference_center_y = permute.permut_reference_value(center_y, reference_center_y)

    return reference_radius, reference_center_x, reference_center_y

def adjust_length_reference(radius, reference_radius, difference_radius, center_x,  reference_center_x, \
difference_center_x, center_y, reference_center_y, difference_center_y):

    """
    Add some 0 to the reference vectors if they are shorter than the value vectors.
    - Input: Three values vectors
             Three reference vectors
             Three difference vectors
    - Output: Three reference vectors
              Three values vectors
    """

    radius, reference_radius = permute.permut_value_add_zero(radius, reference_radius, difference_radius)
    center_x, reference_center_x = permute.permut_value_add_zero(center_x, reference_center_x, difference_center_x)
    center_y, reference_center_y = permute.permut_value_add_zero(center_y, reference_center_y, difference_center_y)

    return  radius, reference_radius, center_x, reference_center_x, center_y, reference_center_y

def compare(value, reference_value, treshold, variation_value, batch_i):

    """
    Adapt values with references if the variations are significative.
    - Input: Values vector
             Reference vector
             Tolerated treshold
             Variation value
    - Output: Reference vector
              Value vector
    """

    for i in range(len(value)):
        if batch_i == 0:
            reference_value[i] = value[i]
        if batch_i != 0:
            if variation_value[i] < treshold:
                value[i] = reference_value[i]
            else:
                reference_value[i] = value[i]
    return value, reference_value
