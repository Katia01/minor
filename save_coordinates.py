"""
Created on November 2019

@author: Katia Schalk

Create csv files where different values predicted or estimated with the Neural-Net and the 3D camera are saved.
"""

import csv
import numpy as np

def Create_csv_File(filename1, filename2, filename3):
    """
    Creates an output file in .csv format.

    - Input: The name of the .csv output file to be created
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
        col2 = ['REAR_x', 'REAR_y', 'LEAR_x', 'LEAR_y',\
               'RSHO_x', 'RSHO_y', 'LSHO_x', 'LSHO_y', \
               'RTHI_x', 'RTHI_y', 'LTHI_x', 'LTHI_y', \
               'RKNE_x', 'RKNE_y', 'LKNE_x', 'LKNE_y', \
               'RANK_x', 'RANK_y', 'LANK_x', 'LANK_y']

        writer = csv.writer(csvFile)
        writer.writerow(col2)
        csvFile.close()

    with open(filename3, 'w') as csvFile:
        col3 = ['EAR_x', 'EAR_y','EAR_z', 'SHO_x', 'SHO_y', 'SHO_z' 'THI_x', 'THI_y', 'THI_z']

        writer = csv.writer(csvFile)
        writer.writerow(col3)
        csvFile.close()

def Save_xyz_Coordinates_csv(xyz_REAR, xyz_LEAR, xyz_RSHO, xyz_LSHO, xyz_RTHI, xyz_LTHI, xyz_RKNE, xyz_LKNE, xyz_RANK, xyz_LANK, filename1):
    """
    Save the xyz coordinates (in meter) in an already created .csv file.

    - Input: 5 vectors containing the right xyz coordinates to save (in meter)
             5 vectors containing the left xyz coordinates to save (in meter)
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

def Save_xy_Coordinates_csv(xy_REAR, xy_LEAR, xy_RSHO, xy_LSHO, xy_RTHI, xy_LTHI, xy_RKNE, xy_LKNE, xy_RANK, xy_LANK, filename2):
    """
    Save the xy coordinates (in pixel) in an already created .csv file.

    - Input: 5 vectors containing the right xy coordinates to save (in pixel)
             5 vectors containing the left xy coordinates to save (in pixel)
             The name of the existing .csv file
    """

    EAR = [xy_REAR[0], xy_REAR[1], xy_LEAR[0], xy_LEAR[1]]
    SHO = [xy_RSHO[0], xy_RSHO[1], xy_LSHO[0], xy_LSHO[1]]
    THI = [xy_RTHI[0], xy_RTHI[1], xy_LTHI[0], xy_LTHI[1]]
    KNE = [xy_RKNE[0], xy_RKNE[1], xy_LKNE[0], xy_LKNE[1]]
    ANK = [xy_RANK[0], xy_RANK[1], xy_LANK[0], xy_LANK[1]]

    row = np.concatenate((EAR, SHO, THI, KNE, ANK), axis=None)

    with open(filename2, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
        csvFile.close()

def Save_Center_xyz_Coordinates_csv(xyz_EAR, xyz_SHO, xyz_THI, filename3):
    """
    Save the xyz coordinates of the middle of two body parts (in meter) in an already created .csv file.

    - Input: 3 vectors containing the xyz coordinates to save (in meter)
             The name of the existing .csv file
    """

    EAR = [xyz_EAR[0], xyz_EAR[1], xyz_EAR[2]]
    SHO = [xyz_SHO[0], xyz_SHO[1], xyz_SHO[2]]
    THI = [xyz_THI[0], xyz_THI[1], xyz_THI[2]]

    row = np.concatenate((EAR, SHO, THI), axis=None)

    with open(filename3, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
        csvFile.close()
