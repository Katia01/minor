"""
Created on December 2019

@author: Katia Schalk

Correct the xy shift caused by the different field of view.
"""

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

    right_x = int((((right_xy[0]-320)*32.5)/35)+320)
    right_y = int((((right_xy[1]-240)*20)/21)+240)
    left_x = int((((left_xy[0]-320)*32.5)/35)+320)
    left_y = int((((left_xy[1]-240)*20)/21)+240)

    right_xy_correct.append(right_x)
    right_xy_correct.append(right_y)
    left_xy_correct.append(left_x)
    left_xy_correct.append(left_y)

    return right_xy_correct, left_xy_correct
