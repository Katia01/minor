"""
Created on November 2019

@author: Katia Schalk

Transform xy prediction coordinates from pixel to meter and compute distance.
"""

import math

def Convert_Pixel_To_Meter(xyz_coordinates):
    """
    Convert xy prediction coordinates from pixel to meter.

    - Input: Vector containing xyz body parts coordinates, with xy in pixel
    - Ouput: Vector containing xyz body parts coordinates, all in meter
    """

    # FOV of the 3D camera:  (H x V x D): 69.4° x 42.5° x 77° (+/- 3°)
    # It is useful to convert xy pixel coordinates into xy meter coordinates.
    alpha_deg = int(70/2)
    beta_deg = int(42/2)
    alpha_rad = math.radians(alpha_deg)
    beta_rad = math.radians(beta_deg)
    tan_alpha = math.tan(alpha_rad)
    tan_beta = math.tan(beta_rad)

    L_pixel = int(641/2)
    H_pixel = int(481/2)

    xyz_new_coordinates = []

    x_coordinate = xyz_coordinates[0]
    y_coordinate = xyz_coordinates[1]

    # Convert z from mm to m
    z_new_coordinate = (xyz_coordinates[2] * 0.001)

    L_meter = tan_alpha * z_new_coordinate
    H_meter = tan_beta * z_new_coordinate

    x_new_coordinate = (x_coordinate * L_meter)/L_pixel
    y_new_coordinate =(y_coordinate * H_meter)/L_pixel

    xyz_new_coordinates.append(round(x_new_coordinate,2))
    xyz_new_coordinates.append(round(y_new_coordinate,2))
    xyz_new_coordinates.append(round(z_new_coordinate,2))

    return xyz_new_coordinates

def Compute_Norm(xyz_right_coordinates, xyz_left_coordinates):
    """
    Compute the norm between the right and the left body parts.

    - Input: Two vector containing xyz right and left body parts coordinates
    - Ouput: The norm between the left and the right point
    """

    xR = xyz_right_coordinates[0]
    yR = xyz_right_coordinates[1]
    zR = xyz_right_coordinates[2]

    xL = xyz_left_coordinates[0]
    yL = xyz_left_coordinates[1]
    zL = xyz_left_coordinates[2]

    norm = math.sqrt((xR - xL) ** 2 + (yR - yL) ** 2 + (zR - zL) ** 2)

    return norm
