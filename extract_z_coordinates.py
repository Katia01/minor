"""
Created on November 2019

@author: Katia Schalk

Associate xy coordinates with the corresponding z coordinate.
"""

import numpy as np

def Return_xyz_Coordinates(right_xy_coordinate, left_xy_coordinate, array_number, z_limit, array_name):
    """
    Find the z missing coordinates for specific xy coordinates.

    - Input: Left and right xy coordinates of the sepcific body part
             The numero of the z coordinates array
             Abberant value of z
             Name of the .npy file containing all the z coordinates array
    - Output: Two vector with the left and right xyz body coordinates
    """

    #z_array = np.load("depth_values_2/array_" + str(array_number) + ".npy")
    z_array = np.load(array_name + str(array_number) + ".npy")

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

def Return_Center_xyz_Coordinates(xy_coordinates, array_number, z_limit, array_name):
    """
    Find the z missing coordinates for specific xy coordinates.

    - Input: xy coordinates of the sepcific body part
             The numero of the z coordinates array
             Abberant value of z
             Name of the .npy file containing all the z coordinates array
    - Output: One vector with xyz body coordinates
    """

    z_array = np.load(array_name + str(array_number) + ".npy")

    array_coordinates = []

    z = z_array[xy_coordinates[1]][xy_coordinates[0]]

    array_coordinates.append(xy_coordinates[0])
    array_coordinates.append(xy_coordinates[1])
    array_coordinates.append(z)

    return array_coordinates
