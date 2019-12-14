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

    if 0 < array_xyz[3] <= 1
        radius = 35
    else:
        if 1 < array_xyz[3] <= 2:
            radius = 30
        else:
            if 2 < array_xyz[3] <= 4:
                radius = 25
            else:
                    if 4 < array_xyz[3] <= 6:
                        radius = 20
                    else:
                        if 6 < array_xyz[3] <= 8:
                            radius = 15
                        else:
                            if 8 < array_xyz[3]:
                                radius = 10
    return radius

def Return_Circle_Center(array_xyz):
    """
    Return subjects face center.

    - Input: .json file containing xy prediction coordinates
    - Output: Two vector, one with all the x center coordinates and one with all the y center coordinates
    """
    center_x = []
    center_y = []

    center_x = array_xyz[1]
    center_y = array_xyz[2]

    return center_x, center_y

def Blur_Face (image_filename, coordinate_filename, picture_size, radius, center_x, center_y):
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
    mask = cv2.circle(mask, (center_x, center_y), radius,(255, 255, 255), -1)
    out = np.where(mask!=np.array([255, 255, 255]), out, blurred_img)
    cv2.imwrite(image_filename + ".blur.png", out)
