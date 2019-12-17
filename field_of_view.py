"""
Created on December 2019

@author: Katia Schalk

Correct the xy shift caused by the different field of view.
"""
import math
import cv2
import numpy as np

def Correct_Shift(right_xy, left_xy):
    """
    Correct the xy shift caused by the different field of view, to find the true z.

    - Input: A vector containg the right xy coordinates
             A vector contraining the left xy coordinates
    - Output: A vector containing the correct right xy coordinates
              A vector containing the correct left xy coordinates
    """
    right_xy_correct = []
    left_xy_correct = []

    right_x = right_xy[0]
    right_y = right_xy[1]
    left_x = left_xy[0]
    left_y = left_xy[1]

    right_xy_correct.append(right_x)
    right_xy_correct.append(right_y)
    left_xy_correct.append(left_x)
    left_xy_correct.append(left_y)

    return right_xy_correct, left_xy_correct

def Rescale_Depth_Image(array_name, array_number):

    alpha_rad_x = math.radians(34.7)
    alpha_rad_y = math.radians(21.25)
    beta_rad_x = math.radians(32.5)
    beta_rad_y = math.radians(20)

    depth_rescale = [[0 for col in range(640)] for row in range(480)]
    z_array = np.load(array_name + str(array_number) + ".npy")


    for v in range(len(z_array[0])):#column (640)
        for h in range(len(z_array)):#line (480)

            v1 = int(320 + (v - 320) * (math.tan(beta_rad_x)/math.tan(alpha_rad_x))+27)
            h1 = int(240 + (h - 240) * (math.tan(beta_rad_y)/math.tan(alpha_rad_y))+13)

            if h ==0 and v == 0:
                print(h1)
                print(v1)

            depth_rescale[h1][v1] = z_array[h][v]

    np.save(array_name + str(array_number) + "_rescale.npy", depth_rescale)

    return depth_rescale
