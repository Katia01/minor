"""
Created on December 2019

@author: Katia Schalk

Make useful plots.
"""

import matplotlib.pyplot as plt
import csv

"""
Plot left and right knee angle according to time.
"""
for num_col in range(2):
    x = []
    y = []

    with open('KNE_angle.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(row[0])
            y.append(row[num_col+1])

    if num_col == 0:
        plt.figure(1)
        plt.plot(x,y, label='Loaded from file!')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [degree]')
        plt.title('Knee angle in function of time')
        plt.legend()
        plt.savefig('Right_knee_angle.png')

    else:
        plt.figure(2)
        plt.plot(x,y, label='Loaded from file!')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [degree]')
        plt.title('Knee angle in function of time')
        plt.legend()
        plt.savefig('Left_knee_angle.png')
