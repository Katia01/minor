"""
Created on November 2019

@author: Katia Schalk

Identify which subject is the patient thanks to the green stickers place on his chest and on his back.
"""

import cv2
import numpy as np

def Identify_Patient_Coordinates(image_filename):
    """
    Use a green filter to determine the position of the green sticker.

    - Input: The original picture
    - Output: A vector containing the xy coordinates of the green sticker
    """

    img = cv2.imread(image_filename)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 52, 72), (90, 255,255))

    xy_patient_coordinates = []
    x_coordinates = []
    y_coordinates = []

    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    for i in range (481-7):
        for j in range (641-8):
            if mask[i][j] == 255 and mask[i+7][j+8] == 255:
                x_coordinates.append(j)
                y_coordinates.append(i)

    if not x_coordinates:
        for i in range (481-4):
            for j in range (641-5):
                if mask[i][j] == 255 and mask[i+4][j+5] == 255:
                    x_coordinates.append(j)
                    y_coordinates.append(i)

    xy_patient_coordinates.append(np.median(x_coordinates))
    xy_patient_coordinates.append(np.median(y_coordinates))

    return xy_patient_coordinates

def Determine_Patient_index(right_x, right_y, left_x, left_y, xy_reference_patient):
    """
    Determine the index corresponding to the patient.

    - Input: 4 vectors with the different coordinates of all the subject
             1 vector containing the xy coordinates of the green sticker
    - Output: The index of the patient
    """

    difference_x = []
    difference_y = []

    for subject in range(len(right_x)):
        difference_x.append(abs(right_x[subject] - xy_reference_patient[0]) + abs(left_x[subject] - xy_reference_patient[0]))

    index = np.argmin(difference_x)

    return index

def Select_Patient_Coordinates(right_x, right_y, left_x, left_y, index):
    """
    Exctract xy patient coordinates using the index.

    - Input: 4 vectors with the different coordinates of all the subject
             The index of the patient
    - Output: 2 vectors with the left and right xy coordinates of the patient
    """

    xy_right_patient_coordinates = []
    xy_left_patient_coordinates = []

    xy_right_patient_coordinates.append(right_x[index])
    xy_right_patient_coordinates.append(right_y[index])
    xy_left_patient_coordinates.append(left_x[index])
    xy_left_patient_coordinates.append(left_y[index])

    return xy_right_patient_coordinates, xy_left_patient_coordinates
