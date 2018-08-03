# Motion Capture - Informal Guide To Installation and Use

Tommaso Monaco, 07/2018

## Overview

Middlebury College summer research project, *"A System for Capturing Datasets for Mobile Image Matching"*, employed Xsens motion capture system (MVN Full Body -A) to record human arm motion. The project aimed to replicate actual human movement on a robotic arm. This is a guide to the installantion, setup, and functionalities of the motion capture system.

### Prerequisites

Xsens motion capture system can interface with the user via several MVN programs. We used **MVN Animate Pro 2018**, which we obtained using the college-owned product and license keys. The license key comes in the form of a dongle; the product key is a string of number and letters. Both keys can be found in the backpack that comes with the motion capture kit.  
The software can be downloaded from the Xsens [website](https://www.xsens.com/software/mvn-animate/). During installation, it is important that the license key dongle is plugged in; the user will be prompted for the product key, too. A *window computer* or a *Linux computer* are required to install MVN Animate Pro 2018. **Important**: a firmware update might be requiried after installation.



A firmware updater can be found on the same webpage from which you get the software.

## How does it work?

### Prepare for Recording a session


The computer communicates with the trackers via a wireless device: either the tracker's station with the antenna or the wireless dongle. Connect *only* one of them to the computer - connecting more than one will cause the tracker-computer communication to fail. 
Before recording a session for data acquisition, the kit needs to be calibrated. To calibrate the kit, select the *New Recording Session* icon. Then, a configuration window pops up; click on the settings icon next to the session name. Here, you may be prompted for a firmware update. 
When all the hardware is detected, click ok and move to the kit calibration (calibration menu is on the left of the configuration window). The *prop* tracker is the most important tracker as its trajectory will be used to direct the robot motions afterward. Hence, it is important to choose carefully the orientation of the prop tracker during calibration. Calibration will *set* the reference frame of the trackers. Hence, the z-coordinate will point upwards with respect to the orientation of the prop during calibration. Of course, it is suggested to keep the tracker in such an orientation that can be easily maintained (like a smartphone) when recording a session.
After calibration, you're ready to record a session! Click on the recording (red) button, and start recording.  

### Other tips

* Remember to face in the direction of the red arrow displayed on the screen (the x-axis direction). Then, turn a few degrees to the right so that you are now facing slightly off the x-axis. That's necessary because of the robot arm reference frame, which is slightly tilted. If you want the robot arm to perform the desired motion straight ahead, then follow these instructions.
* You may be prompted with an error message "Unavailable packets." That happens because the frequency of the communication between the station and the trackers is too high, so some data packets sent by the trackers get lost. Thus, lower the frequency of communication to 40Hz or thereabouts.

## What comes next?

After a session has been recorded, you will be prompted to save the session file. Thus, retrieve the location of the saved session file, open it up and click on "File". Then, click "Export" in the following drop-down menu. There are many options at this point. We suggest to export it as a .MVNX file, from which all the motion data can be easily accessed.

## More?

Yes, there is much more. This is a simple and short guide. Additional tutorials can be found [here](https://tutorial.xsens.com/). The detailed MVN user manual can be found [here](https://xsens.com/download/usermanual/3DBM/MVN_User_Manual.pdf).


