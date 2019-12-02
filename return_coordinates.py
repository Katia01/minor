"""-----------------------------------------------------------------------------
Created on September 2019

@author: Katia Schalk

Return coordinates of specific body parts using the openpifpaf Neural-Net predictions & the 3D intel camera.
-----------------------------------------------------------------------------"""

import json
import numpy as np
import statistics

def Return_Nose_Coordinates(coordinate_filename):
    """
    Return xy nose coordinates.

    - Input: .json file containing xy prediction coordinates
    - Output: Two vector, one with all the x nose coordinates and one with all the y nose coordinates
    """

    nose_x = []
    nose_y = []

    with open(coordinate_filename, 'r') as f:
        distros_dict = json.load(f)

        number_subjects =  len(distros_dict)

        for subject in range(number_subjects):
            nose_x.append(int(distros_dict[subject]['keypoints'][0]))
            nose_y.append(int(distros_dict[subject]['keypoints'][1]))

        return nose_x, nose_y

def Return_Eyes_Coordinates(coordinate_filename):
    """
    Return xy eyes coordinates.

    - Input: .json file containing xy prediction coordinates
    - Output: Four vector (distinction between left and right)
              => One with all the x eyes coordinates and one with all the y eyes coordinates
    """

    right_eye_x = []
    right_eye_y = []
    left_eye_x = []
    left_eye_y = []

    with open(coordinate_filename, 'r') as f:
        distros_dict = json.load(f)

        number_subjects =  len(distros_dict)

        for subject in range(number_subjects):
            right_eye_x.append(int(distros_dict[subject]['keypoints'][2*3]))
            right_eye_y.append(int(distros_dict[subject]['keypoints'][2*3+1]))
            left_eye_x.append(int(distros_dict[subject]['keypoints'][1*3]))
            left_eye_y.append(int(distros_dict[subject]['keypoints'][1*3+1]))

        return right_eye_x, right_eye_y, left_eye_x, left_eye_y

def Return_Ears_Coordinates(coordinate_filename):
    """
    Return xy ears coordinates.

    - Input: .json file containing xy prediction coordinates
    - Output: Four vector (distinction between left and right)
              => One with all the x ears coordinates and one with all the y ears coordinates
    """

    right_ear_x = []
    right_ear_y = []
    left_ear_x = []
    left_ear_y = []

    with open(coordinate_filename, 'r') as f:
        distros_dict = json.load(f)

        number_subjects =  len(distros_dict)

        for subject in range(number_subjects):
            right_ear_x.append(int(distros_dict[subject]['keypoints'][4*3]))
            right_ear_y.append(int(distros_dict[subject]['keypoints'][4*3+1]))
            left_ear_x.append(int(distros_dict[subject]['keypoints'][3*3]))
            left_ear_y.append(int(distros_dict[subject]['keypoints'][3*3+1]))

    return right_ear_x, right_ear_y, left_ear_x, left_ear_y

def Return_Shoulders_Coordinates(coordinate_filename):
    """
    Return xy shoulders coordinates.

    - Input: .json file containing xy prediction coordinates
    - Output: Four vector (distinction between left and right)
              => One with all the x shoulders coordinates and one with all the y shoulders coordinates.
    """

    right_shoulder_x = []
    right_shoulder_y = []
    left_shoulder_x = []
    left_shoulder_y = []

    with open(coordinate_filename, 'r') as f:
        distros_dict = json.load(f)

    number_subjects =  len(distros_dict)

    for subject in range(number_subjects):
        right_shoulder_x.append(int(distros_dict[subject]['keypoints'][3*6]))
        right_shoulder_y.append(int(distros_dict[subject]['keypoints'][3*6+1]))
        left_shoulder_x.append(int(distros_dict[subject]['keypoints'][3*5]))
        left_shoulder_y.append(int(distros_dict[subject]['keypoints'][3*5+1]))

    return right_shoulder_x, right_shoulder_y, left_shoulder_x, left_shoulder_y

def Return_Hips_Coordinates(coordinate_filename):
    """
    Return xy hips coordinates.

    - Input: .json file containing xy prediction coordinates
    - Output: Four vector (distinction between left and right)
              => One with all the x hips coordinates and one with all the y hips coordinates.
    """

    right_hip_x = []
    right_hip_y = []
    left_hip_x = []
    left_hip_y = []

    with open(coordinate_filename, 'r') as f:
        distros_dict = json.load(f)

    number_subjects =  len(distros_dict)

    for subject in range(number_subjects):
        right_hip_x.append(int(distros_dict[subject]['keypoints'][3*12]))
        right_hip_y.append(int(distros_dict[subject]['keypoints'][3*12+1]))
        left_hip_x.append(int(distros_dict[subject]['keypoints'][3*11]))
        left_hip_y.append(int(distros_dict[subject]['keypoints'][3*11+1]))

    return right_hip_x, right_hip_y, left_hip_x, left_hip_y

def Return_Knees_Coordinates(coordinate_filename):
    """
    Return xy knees coordinates.

    - Input: .json file containing xy prediction coordinates
    - Output: Four vector (distinction between left and right)
              => One with all the x hips coordinates and one with all the y knees coordinates.
    """

    RKNE_x = []
    RKNE_y = []
    LKNE_x = []
    LKNE_y = []

    with open(coordinate_filename, 'r') as f:
        distros_dict = json.load(f)

    number_subjects =  len(distros_dict)

    for subject in range(number_subjects):
        RKNE_x.append(int(distros_dict[subject]['keypoints'][3*14]))
        RKNE_y.append(int(distros_dict[subject]['keypoints'][3*14+1]))
        LKNE_x.append(int(distros_dict[subject]['keypoints'][3*13]))
        LKNE_y.append(int(distros_dict[subject]['keypoints'][3*13+1]))

    return RKNE_x, RKNE_y, LKNE_x, LKNE_y

def Return_Ankles_Coordinates(coordinate_filename):
        """
        Return xy ankles coordinates.

        - Input: .json file containing xy prediction coordinates
        - Output: Four vector (distinction between left and right)
                  => One with all the x hips coordinates and one with all the y ankles coordinates.
        """

        RANK_x = []
        RANK_y = []
        LANK_x = []
        LANK_y = []

        with open(coordinate_filename, 'r') as f:
            distros_dict = json.load(f)

        number_subjects =  len(distros_dict)

        for subject in range(number_subjects):
            RANK_x.append(int(distros_dict[subject]['keypoints'][3*16]))
            RANK_y.append(int(distros_dict[subject]['keypoints'][3*16+1]))
            LANK_x.append(int(distros_dict[subject]['keypoints'][3*15]))
            LANK_y.append(int(distros_dict[subject]['keypoints'][3*15+1]))

        return RANK_x, RANK_y, LANK_x, LANK_y
