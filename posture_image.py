"""
Created on November 2019

@author: Katia Schalk

Compute different movement caracteristics.
"""

import math
import numpy as np
from . import from_pixel_to_meter

def Compute_Knee_Angle (xyz_THI, xyz_KNE, xyz_ANK):
    """
    Compute knee angle.

    - Input: A vector containing xyz hip coordinates (in meter)
             A vector containing xyz knee coordinates (in meter)
             A vector containing xyz ankle coordinates (in meter)
    - Output: A knee angle (in degree)
    """

    a = from_pixel_to_meter.Compute_Norm(xyz_THI, xyz_KNE)
    b = from_pixel_to_meter.Compute_Norm(xyz_KNE, xyz_ANK)
    norm_ab = a * b

    vector_THI_KNE = np.subtract(xyz_THI, xyz_KNE)
    vector_ANK_KNE = np.subtract(xyz_ANK, xyz_KNE)

    scalar = np.dot(vector_THI_KNE, vector_ANK_KNE)

    teta = math.acos(scalar/norm_ab)
    knee_angle = round(math.degrees(teta),1)

    if  math.isnan(knee_angle):
        knee_angle = 0.0

    return knee_angle
