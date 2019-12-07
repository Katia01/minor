"""
Created on November 2019

@author: Katia Schalk

Associate xy coordinates with the corresponding z coordinate.
"""

import numpy as np

def Return_xyz_Coordinates(right_xy_coordinate, left_xy_coordinate, array_number, z_limit):
    """
    Find the z missing coordinates for specific xy coordinates.

    - Input: .npy file containing z coordinates
             Left and right xy coordinates of the sepcific body part
    - Output: Two vector with the left and right xyz body coordinates
    """

    z_array = np.load("depth_values_2/array_" + str(array_number) + ".npy")

    right_array_coordinates = []
    left_array_coordinates = []

    right_z = z_array[right_xy_coordinate[1]][right_xy_coordinate[0]]
    left_z = z_array[left_xy_coordinate[1]][left_xy_coordinate[0]]

    if right_z == 0 or right_z >  z_limit:
        r = 1
        if left_xy_coordinate[0] - right_xy_coordinate[0] > 0:
            while right_z == 0 or right_z > z_limit:
                right_z = z_array[right_xy_coordinate[1]][right_xy_coordinate[0] + r]
                r = r + 1
        else:
            while right_z == 0 or right_z >  z_limit:
                right_z = z_array[right_xy_coordinate[1]][right_xy_coordinate[0] - r]
                r = r + 1

    if left_z == 0 or left_z > z_limit:
        l = 1
        if left_xy_coordinate[0]  - right_xy_coordinate[0] > 0:
            while left_z == 0 or left_z > z_limit:
                left_z = z_array[left_xy_coordinate[1]][left_xy_coordinate[0] - l]
                l = l + 1
        else:
            while left_z == 0 or left_z > z_limit:
                left_z = z_array[left_xy_coordinate[1]][left_xy_coordinate[0] + l]
                l = l + 1

    right_array_coordinates.append(right_xy_coordinate[0])
    right_array_coordinates.append(right_xy_coordinate[1])
    right_array_coordinates.append(right_z)
    left_array_coordinates.append(left_xy_coordinate[0])
    left_array_coordinates.append(left_xy_coordinate[1])
    left_array_coordinates.append(left_z)

    return right_array_coordinates, left_array_coordinates
