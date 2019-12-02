"""-----------------------------------------------------------------------------
Created on October 2019

@author: Katia Schalk

Create an executable file that applies all the command lines needed to blur faces
on all the new videos.
To create the executable file execute the following command on the terminal:
    pyinstaller blur_executable_all_files.py
-----------------------------------------------------------------------------"""

import subprocess, shlex
import os

def subprocess_cmd(command):

    """
    Allow the executable file to apply some terminal commands.
    - Input: Command to apply on the terminal
    """

    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()

def start_setup():

    """
    Create the executable file which allow the user to blur faces on all the new videos
    """

    for filename1 in os.listdir("/Users/KatiaSchalk/Desktop/GDPLogging"):
    #for filename1 in os.listdir("C:/GDPLogging/Patients"):
        if filename1.startswith("Patient_"):
                a=os.path.join(directory, filename1)
            for filename2 in os.listdir(a + "/Sessions")
                if filename2.startswith("Session_"):
                    for filename3 in os.listdir(os.path.join(directory, filename2)+ "/RecordingLogs"):
                        if filename3.endswith(".avi"):
                            file =  os.path.basename(filename3)
                            path = os.path.dirname(filename3)
                            if not filename4.endswith(file + ".blur.mp4"):

                                print ("The blurring process will start on " + file + ". It will take time, please wait.")
                                subprocess_cmd("cd " + path +"; mkdir " + file +".images; ffmpeg -i " + file + "\
                                -qscale:v 2 -vf scale=641:-1 -f image2 "+file+".images/%05d.jpg; python3 -m \
                                openpifpaf.blur_video --checkpoint resnet152 --glob \"" +file+ ".images/*[05].jpg\"; \
                                ffmpeg -framerate 24 -pattern_type glob -i "+file+".images/\'*.jpg.blur.png\' -vf scale=640:-2 \
                                -c:v libx264 -pix_fmt yuv420p "+file+".blur.mp4")
                                print ("Congratulations," + file + ".blur.mp4 succesfully created !")

start_setup()
