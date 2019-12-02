"""-----------------------------------------------------------------------------
Created on September 2019

@author: Katia Schalk

Blur faces on a given image.
-----------------------------------------------------------------------------"""

import cv2
import numpy as np
import json

from . import return_coordinates

def Return_Radius(coordinate_filename):
    """
    Return subjects face radius & the number of subject present on the predicted image.

    - Input: .json file containing xy prediction coordinates
    - Output: A vector containing the subjects face Return_Radius
              The number of subject
    """
    radius = []

    nose_x, nose_y = return_coordinates.Return_Nose_Coordinates(coordinate_filename)
    right_eye_x, right_eye_y, left_eye_x, left_eye_y = return_coordinates.Return_Eyes_Coordinates(coordinate_filename)
    right_ear_x, right_ear_y, left_ear_x, left_ear_y = return_coordinates.Return_Ears_Coordinates(coordinate_filename)
    right_shoulder_x, right_shoulder_y, left_shoulder_x, left_shoulder_y = return_coordinates.Return_Shoulders_Coordinates(coordinate_filename)

    number_subjects = len(nose_x)

    for subject in range(number_subjects):

        if right_eye_x[subject] != 0 and left_eye_x[subject] != 0 and nose_x[subject] != 0 and right_ear_x[subject] !=0 and left_ear_x[subject] ==0:
            distance_eye_ear = np.abs(right_eye_x[subject] - right_ear_x[subject])
            print('eye-ear')
            print(distance_eye_ear)
            if distance_eye_ear >= 15:
                radius.append(int(2*distance_eye_ear))
                print('a')
            else:
                if distance_eye_ear >= 12.5 and  distance_eye_ear< 15:
                    radius.append(30)
                    print('ab')
                else:
                    if distance_eye_ear >= 9 and  distance_eye_ear< 12.5:
                        radius.append(28)
                        print('abb')
                    else:
                        if distance_eye_ear >= 4 and  distance_eye_ear< 9:
                            radius.append(26)
                            print('abbb')
                        else:
                            if distance_eye_ear >= 2 and  distance_eye_ear< 4:
                                radius.append(23)
                                print('abbbb')
                            else:
                                radius.append(20)
                                print('abbbbb')


        if right_eye_x[subject] != 0 and left_eye_x[subject] != 0 and nose_x[subject] != 0 and right_ear_x[subject] ==0 and left_ear_x[subject] !=0:
            distance_eye_ear = np.abs(left_eye_x[subject] - left_ear_x[subject])
            print('eye-ear')
            print(distance_eye_ear)
            if distance_eye_ear >= 15:
                radius.append(int(2*distance_eye_ear))
                print('b')
            else:
                if distance_eye_ear >= 12.5 and  distance_eye_ear< 15:
                    radius.append(30)
                    print('bb')
                else:
                    if distance_eye_ear >= 9 and  distance_eye_ear< 12.5:
                        radius.append(28)
                        print('bbb')
                    else:
                        if distance_eye_ear >= 4 and  distance_eye_ear< 9:
                            radius.append(26)
                            print('bbbb')
                        else:
                            if distance_eye_ear >= 2 and  distance_eye_ear< 4:
                                radius.append(23)
                                print('bbbbb')
                            else:
                                radius.append(20)
                                print('bbbbbb')

        #--------------- Seen from the front
        if right_eye_x[subject] != 0 and left_eye_x[subject] != 0 and nose_x[subject] != 0 and right_ear_x[subject] !=0 and left_ear_x[subject] !=0:
            distance_eye_eye = np.abs(left_eye_x[subject] - right_eye_x[subject])
            print('eye-eye')
            print(distance_eye_eye)
            if distance_eye_eye >= 15:
                radius.append(int(2*distance_eye_eye))
                print('c')
            else:
                if distance_eye_eye>=12.5 and  distance_eye_eye < 15:
                    radius.append(30)
                    print('cb')
                else:
                    if distance_eye_eye>=9 and  distance_eye_eye < 12.5:
                        radius.append(28)
                        print('cbb')
                    else:
                        if distance_eye_eye >= 4 and  distance_eye_eye< 9:
                            radius.append(26)
                            print('cbbb')
                        else:
                            if distance_eye_eye >= 2 and  distance_eye_eye< 4:
                                radius.append(23)
                                print('cbbbb')
                            else:
                                radius.append(20)
                                print('cbbbbb')

        #--------------- Seen from the front
        if right_eye_x[subject] != 0 and left_eye_x[subject] != 0 and nose_x[subject] != 0 and right_ear_x[subject] ==0 and left_ear_x[subject] ==0:
            distance_eye_eye = np.abs(left_eye_x[subject] - right_eye_x[subject])
            print('eye-eye')
            print(distance_eye_eye)
            if distance_eye_eye >= 15:
                radius.append(int(2*distance_eye_eye))
                print('d')
            else:
                if distance_eye_eye>=12.5 and  distance_eye_eye < 15:
                    radius.append(30)
                    print('db')
                else:
                    if distance_eye_eye>=9 and  distance_eye_eye < 12.5:
                        radius.append(28)
                        print('dbb')
                    else:
                        if distance_eye_eye >= 4 and  distance_eye_eye< 9:
                            radius.append(26)
                            print('dbbb')
                        else:
                            if distance_eye_eye >= 2 and  distance_eye_eye< 4:
                                radius.append(23)
                                print('dbbbb')
                            else:
                                radius.append(20)
                                print('dbbbbb')

        #--------------- Seen from the right side without nose
        if right_eye_x[subject]!=0 and left_eye_x[subject]==0 and  right_ear_x[subject] !=0:
            distance_eye_ear = np.abs(right_eye_x[subject] - right_ear_x[subject])
            print('eye-ear')
            print(distance_eye_ear)
            if distance_eye_ear >= 15:
                radius.append(int(2*distance_eye_ear))
                print('e')
            else:
                if distance_eye_ear >= 12.5 and  distance_eye_ear< 15:
                    radius.append(30)
                    print('eb')
                else:
                    if distance_eye_ear >= 9 and  distance_eye_ear< 12.5:
                        radius.append(28)
                        print('ebb')
                    else:
                        if distance_eye_ear >= 4 and  distance_eye_ear< 9:
                            radius.append(26)
                            print('ebbb')
                        else:
                            if distance_eye_ear >= 2 and  distance_eye_ear< 4:
                                radius.append(23)
                                print('ebbbb')
                            else:
                                radius.append(20)
                                print('ebbbbb')

        #--------------- Seen from the left side without nose
        else:
            if left_eye_x[subject]!=0 and right_eye_x[subject]==0 and left_ear_x[subject]!=0:
                distance_eye_ear = np.abs(left_ear_x[subject] - left_eye_x[subject])
                print('eye-ear')
                print(distance_eye_ear)
                if distance_eye_ear >= 15:
                    radius.append(int(2*distance_eye_ear))
                    print('z')
                else:
                    if distance_eye_ear >= 12.5 and  distance_eye_ear< 15:
                        radius.append(30)
                        print('zbb')
                    else:
                        if distance_eye_ear >= 9 and  distance_eye_ear< 12.5:
                            radius.append(28)
                            print('zbbb')
                        else:
                            if distance_eye_ear >= 4 and  distance_eye_ear< 9:
                                radius.append(26)
                                print('zbbbb')
                            else:
                                if distance_eye_ear >= 2 and  distance_eye_ear< 4:
                                    radius.append(23)
                                    print('zbbbbb')
                                else:
                                    radius.append(20)
                                    print('zbbbbbb')

        #--------------- Seen from behind with two ears
            else:
                if nose_x[subject] == 0 and right_ear_x[subject] != 0 and left_ear_x[subject] != 0:
                    distance_ear_ear = np.abs(right_ear_x[subject] - left_ear_x[subject])
                    print('ear-ear')
                    print(distance_ear_ear)
                    if distance_ear_ear >= 25:
                        radius.append(int(1.5*distance_ear_ear))
                        print('f')
                    else:
                        if distance_ear_ear >= 18 and  distance_ear_ear < 25:
                            radius.append(28)
                            print('fbis')
                        else:
                            if distance_ear_ear >= 12 and  distance_ear_ear < 18:
                                radius.append(26)
                                print('fbis')
                            else:
                                if distance_ear_ear >= 6 and  distance_ear_ear< 12:
                                    radius.append(24)
                                    print('fbisdd')
                                else:
                                    radius.append(20)
                                    print('fbisb')

        #--------------- Seen from behind with one ear
                else:
                    if nose_x[subject] == 0 and right_ear_x[subject] != 0 and left_ear_x[subject] ==0:
                        if right_shoulder_x[subject] !=0:
                            if right_ear_y[subject]-right_shoulder_y[subject] >= 17:
                                radius.append(int(np.abs(1.1*(right_ear_y[subject]-right_shoulder_y[subject]))))
                            else:
                                if 12 <= right_ear_y[subject]-right_shoulder_y[subject] < 17:
                                    radius.append(int(np.abs(1.5*(right_ear_y[subject]-right_shoulder_y[subject]))))
                                else:
                                    radius.append(int(np.abs(1.8*(right_ear_y[subject]-right_shoulder_y[subject]))))
                        else:
                            radius.append(28)
                            print('w')
                    else:
                        if nose_x[subject] == 0 and right_ear_x[subject] == 0 and left_ear_x[subject] !=0:
                            if left_shoulder_x[subject] !=0:
                                if left_ear_y[subject]-left_shoulder_y[subject] >= 17:
                                    radius.append(int(np.abs(1.1*(left_ear_y[subject]-left_shoulder_y[subject]))))
                                else:
                                    if 12 <= left_ear_y[subject]-left_shoulder_y[subject] < 17:
                                        radius.append(int(np.abs(1.5*(left_ear_y[subject]-left_shoulder_y[subject]))))
                                    else:
                                        radius.append(int(np.abs(2*(left_ear_y[subject]-left_shoulder_y[subject]))))
                            else:
                                radius.append(28)
                                print('h')

    number_subjects = len(radius)

    return radius, number_subjects

def Return_Circle_Center(coordinate_filename):
    """
    Return subjects face center.

    - Input: .json file containing xy prediction coordinates
    - Output: Two vector, one with all the x center coordinates and one with all the y center coordinates
    """
    center_x = []
    center_y = []

    nose_x, nose_y = return_coordinates.Return_Nose_Coordinates(coordinate_filename)
    right_eye_x, right_eye_y, left_eye_x, left_eye_y = return_coordinates.Return_Eyes_Coordinates(coordinate_filename)
    right_ear_x, right_ear_y, left_ear_x, left_ear_y = return_coordinates.Return_Ears_Coordinates(coordinate_filename)

    number_subjects = len(nose_x)

    for subject in range(number_subjects):

        #--------------- Seen from the front
        if right_eye_x[subject] != 0 and left_eye_x[subject] != 0 and nose_x[subject] != 0 and right_ear_x[subject] != 0 and left_ear_x[subject] != 0:
            center_x.append(nose_x[subject])
            center_y.append(nose_y[subject])
            print('1')
        else:
            #--------------- Seen from the front with only the right ear
            if right_eye_x[subject] != 0 and left_eye_x[subject] != 0 and nose_x[subject] != 0 and right_ear_x[subject] != 0 and left_ear_x[subject] == 0:
                center_x.append(int(right_ear_x[subject]+ 0.3 * np.abs(nose_x[subject] - right_ear_x[subject])))
                if nose_y[subject] > right_ear_y[subject]:
                    center_y.append(int(right_ear_y[subject]+ 0.10 * np.abs(nose_y[subject] - right_ear_y[subject])))
                    print('2')
                else:
                    center_y.append(int(nose_y[subject]+ 0.10 * np.abs(nose_y[subject] - right_ear_y[subject])))
                    print('2bis')
            else:
                #--------------- Seen from the front with only the left ear
                if right_eye_x[subject] != 0 and left_eye_x[subject] != 0 and nose_x[subject] != 0 and right_ear_x[subject] == 0 and left_ear_x[subject] != 0:
                    center_x.append(int(nose_x[subject]+ 0.4 * np.abs(nose_x[subject] - left_ear_x[subject])))
                    if nose_y[subject] >= left_ear_y[subject]:
                        center_y.append(int(left_ear_y[subject]+ 0.10 * np.abs(nose_y[subject] - left_ear_y[subject])))
                        print('3')
                    else:
                        center_y.append(int(nose_y[subject]+ 0.10 * np.abs(nose_y[subject] - left_ear_y[subject])))
                        print('3bis')
                else:
                    #--------------- Seen from the left side
                    if right_ear_x[subject] == 0 and left_eye_x[subject] != 0 and nose_x[subject] != 0:
                        if left_ear_x[subject] != 0:
                            center_x.append(int(nose_x[subject] + 0.4 * np.abs(left_ear_x[subject]- nose_x[subject])))
                            if nose_y[subject] >= left_ear_y[subject]:
                                center_y.append(int(left_ear_y[subject] + 0.10 * np.abs(left_ear_y[subject] - nose_y[subject])))
                                print('4')
                            else:
                                center_y.append(int(nose_y[subject] + 0.10 * np.abs(left_ear_y[subject]- nose_y[subject])))
                                print('4bis')
                        else:
                            center_x.append(int(nose_x[subject]))
                            center_y.append(int(nose_y[subject]))

                    else:
                        if right_ear_x[subject] == 0 and left_eye_x[subject] != 0 and nose_x[subject] == 0:
                            center_x.append(left_ear_x[subject])
                            center_y.append(left_ear_y[subject])
                            print('5')
                        else:
                            #--------------- Seen from the right side
                            if right_ear_x[subject] != 0 and left_eye_x[subject] == 0 and nose_x[subject] != 0:
                                if right_ear_x[subject] != 0:
                                    center_x.append(int(right_ear_x[subject] + 0.3 * np.abs(right_ear_x[subject] - nose_x[subject])))
                                    if nose_y[subject] >= right_ear_y[subject]:
                                        center_y.append(int(right_ear_y[subject] + 0.10 * np.abs(right_ear_y[subject] - nose_y[subject])))
                                        print('6')
                                    else:
                                        center_y.append(int(nose_y[subject] + 0.10 * np.abs(right_ear_y[subject] - nose_y[subject])))
                                        print('6bis')
                                else:
                                    center_x.append(int(nose_x[subject]))
                                    center_y.append(int(nose_y[subject]))
                            else:
                                if right_ear_x[subject] == 0 and left_eye_x[subject] != 0 and nose_x[subject] == 0:
                                    center_x.append(right_ear_x[subject])
                                    center_y.append(right_ear_y[subject])
                                    print('7')
                                else:
                                    #--------------- Seen from behind with two ears
                                    if nose_x[subject] == 0 and right_ear_x[subject] != 0 and left_ear_x[subject] != 0:
                                        if right_ear_x[subject]>left_ear_x[subject]:
                                            center_x.append(int(left_ear_x[subject]+0.4*np.abs(right_ear_x[subject]-left_ear_x[subject])))
                                            print('8')
                                        else:
                                            center_x.append(int(right_ear_x[subject]+0.4*np.abs(left_ear_x[subject]-right_ear_x[subject])))
                                            print('9')
                                        center_y.append(int(0.5*(right_ear_y[subject]+left_ear_y[subject])))
                                        print(int(0.5*(right_ear_y[subject]+left_ear_y[subject])))
                                    else:
                                        #--------------- Seen from behind with only the right ear
                                        if nose_x[subject] == 0 and right_ear_x[subject] != 0 and left_ear_x[subject] ==0:
                                            center_x.append(right_ear_x[subject])
                                            center_y.append(right_ear_y[subject])
                                            print('10')
                                        else:
                                            #--------------- Seen from behind with only the left ear
                                            if nose_x[subject] == 0 and right_ear_x[subject] == 0 and left_ear_x[subject] !=0:
                                                center_x.append(left_ear_x[subject])
                                                center_y.append(left_ear_y[subject])
                                                print('11')
                                            else:
                                                if nose_x[subject] != 0 and right_eye_x[subject] != 0 and left_eye_x[subject]==0:
                                                    center_x.append(nose_x[subject])
                                                    center_y.append(nose_y[subject])
                                                    print('12')
                                                else:
                                                    if nose_x[subject] != 0 and left_eye_x[subject] != 0 and right_eye_x[subject]==0:
                                                        center_x.append(left_eye_x[subject])
                                                        center_y.append(left_eye_y[subject])
                                                        print('13')

    return center_x, center_y

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

    r, number_subjects = Return_Radius (coordinate_filename)

    for subject in range(number_subjects):

        mask = cv2.circle(mask, (center_x[subject], center_y[subject]), radius[subject],(255, 255, 255), -1)

        out = np.where(mask!=np.array([255, 255, 255]), out, blurred_img)

        cv2.imwrite(image_filename + ".blur.png", out)
