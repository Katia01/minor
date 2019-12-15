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

    right_x = right_xy[0] - 27
    right_y = right_xy[1] - 13
    left_x = left_xy[0] - 27
    left_y = left_xy[1] - 13

    right_xy_correct.append(right_x)
    right_xy_correct.append(right_y)
    left_xy_correct.append(left_x)
    left_xy_correct.append(left_y)

    return right_xy_correct, left_xy_correct
