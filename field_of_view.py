"""
Created on December 2019

@author: Katia Schalk

Correct the xy shift caused by the different field of view.
"""
import math

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

    alpha_rad_x = math.radians(35)
    alpha_rad_y = math.radians(21.25)
    beta_rad_x = math.radians(32.5)
    beta_rad_y = math.radians(20)

    if right_xy[0]< 320:
        right_x = int(right_xy[0]*(math.tan(beta_rad_x)/math.tan(alpha_rad_x)))
    else:
        right_x = int(320 + (right_xy[0]-320)*(math.tan(beta_rad_x)/math.tan(alpha_rad_x)))

    if right_xy[1]<240:
        right_y = int(right_xy[1]*(math.tan(beta_rad_y)/math.tan(alpha_rad_y)))
    else:
        right_y = int(240 + (right_xy[1]-240)*(math.tan(beta_rad_y)/math.tan(alpha_rad_y)))

    if left_xy[0]<320:
        left_x = int(left_xy[0]*(math.tan(beta_rad_x)/math.tan(alpha_rad_x)))
    else:
        left_x = int(320 + (left_xy[0]-320)*(math.tan(beta_rad_x)/math.tan(alpha_rad_x)))

    if left_xy[1]<240:
        left_y = int(left_xy[1]*(math.tan(beta_rad_y)/math.tan(alpha_rad_y)))
    else:
        left_y = int(240 + (left_xy[1]-240)*(math.tan(beta_rad_y)/math.tan(alpha_rad_y)))

    right_xy_correct.append(right_x)
    right_xy_correct.append(right_y)
    left_xy_correct.append(left_x)
    left_xy_correct.append(left_y)

    return right_xy_correct, left_xy_correct
