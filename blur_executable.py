"""-------------------------------------------------------------------------------
Created on October 2019

@author: Katia Schalk

 Create an executable file that applies all the command lines needed to blur faces on a video chosen by the user.
 To create the executable file execute the following command on the terminal and write the required paths:
    pyinstaller blur_executable.py
-------------------------------------------------------------------------------"""

import subprocess, shlex

def subprocess_cmd(command):

    """
    Allow the executable file to apply some terminal commands.
    - Input: Command to apply on the terminal
    """

    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print (proc_stdout)

def start_setup():

    """
    Create the executable file which allow the user to blur faces on the video of his choice.
    """

    yes = set(['Yes', 'yes','y', 'ye', ''])
    no = set(['No','no','n'])
    prompt = '> '

    print ("Hi, welcome to the blur faces application !" )
    print ("To get started, give the name of your video (with the extension).")
    filename = input(prompt)

    print("Now give the exact directory path where your video is registered.")
    path = input(prompt)

    confirm = input("> Do you want to continue and blur faces on the video " + filename + "? \
    If yes, the blurring process takes time, please wait. (Yes/No)").lower()

    if confirm in yes:
        subprocess_cmd("cd " + path +"; mkdir " + filename +".images; ffmpeg -i " + filename + "\
         -qscale:v 2 -vf scale=641:-1 -f image2 "+filename+".images/%05d.jpg; python3 -m \
         openpifpaf.blur_video --checkpoint resnet152 --glob \"" +filename+ ".images/*[05].jpg\"; \
         ffmpeg -framerate 24 -pattern_type glob -i "+filename+".images/\'*.jpg.blur.png\' -vf scale=640:-2 \
         -c:v libx264 -pix_fmt yuv420p "+filename+".pose.mp4")
        print ("Congratulations," + filename + ".pose.mp4 succesfully created !")
        return True

    elif confirm in no:
        print ("Ok, nothing will happen !")
        return False

    else:
       print ("Please answer yes or no at this question ! Let the application start again.")
       start_setup()

start_setup()
