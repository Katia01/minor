"""
Created on November 2019

@author: Katia Schalk

Create a csv file where xyz values predicted or estimated by the Neural-Net are saved.
"""

import csv
import numpy as np

def Create_csv_File(filename1, filename2):
    """
    Creates an output file in .csv format.

    - Input: The name of of .csv output file to be created
    """

    with open(filename1, 'w') as csvFile:
        col1 = ['REAR_x', 'REAR_y', 'REAR_z', 'LEAR_x', 'LEAR_y', 'LEAR_z',\
               'RSHO_x', 'RSHO_y', 'RSHO_z', 'LSHO_x', 'LSHO_y', 'LSHO_z', \
               'RTHI_x', 'RTHI_y', 'RTHI_z', 'LTHI_x', 'LTHI_y', 'LTHI_z', \
               'RKNE_x', 'RKNE_y', 'RKNE_z', 'LKNE_x', 'LKNE_y','LKNE_z', \
               'RANK_x', 'RANK_y', 'RANK_z', 'LANK_x', 'LANK_y', 'LANK_z']

        writer = csv.writer(csvFile)
        writer.writerow(col1)
        csvFile.close()

    with open(filename2, 'w') as csvFile:
        col2 = ['Time', 'RKNE_angle', 'LKNE_angle']

        writer = csv.writer(csvFile)
        writer.writerow(col2)
        csvFile.close()

def Save_Coordinates_csv(xyz_REAR, xyz_LEAR, xyz_RSHO, xyz_LSHO, xyz_RTHI, xyz_LTHI, xyz_RKNE, xyz_LKNE, xyz_RANK, xyz_LANK, filename1):
    """
    Save the xyz coordinates in an already created .csv file.

    - Input: A vector containing the right xyz coordinates to save (in meter)
             A vector containing the left xyz coordinates to save (in meter)
             The name of the existing .csv file
    """
    EAR = [xyz_REAR[0], xyz_REAR[1], xyz_REAR[2], xyz_LEAR[0], xyz_LEAR[1], xyz_LEAR[2]]
    SHO = [xyz_RSHO[0], xyz_RSHO[1], xyz_RSHO[2], xyz_LSHO[0], xyz_LSHO[1], xyz_LSHO[2]]
    THI = [xyz_RTHI[0], xyz_RTHI[1], xyz_RTHI[2], xyz_LTHI[0], xyz_LTHI[1], xyz_LTHI[2]]
    KNE = [xyz_RKNE[0], xyz_RKNE[1], xyz_RKNE[2], xyz_LKNE[0], xyz_LKNE[1], xyz_LKNE[2]]
    ANK = [xyz_RANK[0], xyz_RANK[1], xyz_RANK[2], xyz_LANK[0], xyz_LANK[1], xyz_LANK[2]]

    row = np.concatenate((EAR, SHO, THI, KNE, ANK), axis=None)

    with open(filename1, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
        csvFile.close()

def Save_Angle_csv(RKNE_angle, LKNE_angle, filename2, time):
    """
    Save the knee angles in an already created .csv file.

    - Input: The right knee angle to save (in degree)
             The left knee angle to save (in degree)
             The name of the existing .csv file
    """

    row = [time, RKNE_angle, LKNE_angle]

    with open(filename2, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
        csvFile.close()
