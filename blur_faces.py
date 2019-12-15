"""-----------------------------------------------------------------------------
Created on September 2019

@author: Katia Schalk

Blur faces on a given image.
-----------------------------------------------------------------------------"""

import cv2
import numpy as np
import json

from . import return_coordinates

def Return_Radius(array_xyz):
    """
    Return subjects face radius & the number of subject present on the predicted image.

    - Input: .json file containing xy prediction coordinates
    - Output: A vector containing the subjects face Return_Radius
              The number of subject
    """
    print(array_xyz[2]/1000)
    if 0 < array_xyz[2]/1000 <= 0.5:
        radius = 70
    else:
        if 0.5 < array_xyz[2]/1000 <= 1.5:
            radius = 65
        else:
            if  1.5 < array_xyz[2]/1000 <= 2:
                radius = 60
            else:
                if  1.5 < array_xyz[2]/1000 <= 2:
                    radius = 55
                else:
                    if 2 < array_xyz[2]/1000 <= 2.5:
                        radius = 50
                    else:
                        if 2.5 < array_xyz[2]/1000 <= 3:
                            radius = 40
                        else:
                            if 3 < array_xyz[2]/1000 <= 3.5:
                                radius = 35
                            else:
                                if 3.5 < array_xyz[2]/1000 <= 4:
                                    radius = 30
                                else:
                                    if 4 < array_xyz[2]/1000 <= 4.5:
                                        radius = 25
                                    else:
                                        if 4.5 < array_xyz[2]/1000 <= 5.5:
                                            radius = 20
                                        else:
                                            if 5.5 < array_xyz[2]/1000 <= 6.5:
                                                radius = 17
                                            else:
                                                if 5.5 < array_xyz[2]/1000 <= 7.5:
                                                    radius = 15
                                                else:
                                                    if 7.5 < array_xyz[2]/1000:
                                                        radius = 10
    return radius

def Return_Circle_Center(value_x, value_y):
    """
    Return subjects face center.

    - Input: .json file containing xy prediction coordinates
    - Output: Two vector, one with all the x center coordinates and one with all the y center coordinates
    """

    return value_x, value_y

def Blur_Face (image_filename, coordinate_filename, picture_size, number_subjects, radius, center_x, center_y):
    """
    Blur faces on the original image.

    - Input: Original picture
              .json file containing xy prediction coordinates
              Picture size
              Number of subjects on the picture
              Subjects face radius
              Subjects face center
    - Output: A new image .blur.png which is the original one with all the faces blurred.
    """

    img = cv2.imread(image_filename)
    out = img
    blurred_img = cv2.GaussianBlur(img, (23, 23), 11)

    mask = np.zeros(picture_size, dtype=np.uint8)

    for subject in range(number_subjects):

        mask = cv2.circle(mask, (center_x[subject], center_y[subject]), radius[subject],(255, 255, 255), -1)

        out = np.where(mask!=np.array([255, 255, 255]), out, blurred_img)

        cv2.imwrite(image_filename + ".blur.png", out)
