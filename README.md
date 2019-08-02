# Robotic-Palm
The project goal was to design a working model of a human palm and to replicate the palm motion and controls using a video feed of a human palm using cameras.

**Poster** - [link](https://github.com/mchandak29/robotic-palm/blob/master/Design%20Project%20poster.pdf)

## Hardware Design
<p align="center">
    <img src="https://github.com/mchandak29/robotic-palm/blob/master/Hardware_Design.png", width="480">
</p>

## Working
Using pose estimation, first we detect all the joints and segments and generate a heat map. Using these positions, we calculate the joint angles with change in position and these angles are transmitted to an arduino via serial transfer. The
arduino, maps these angles to its corresponding motor angle. Now, as the motor rotates, the finger bends similar to a human finger in a direction depending upon the direction of rotation of the motor.

For in-depth explanation of the design and working, please read [here](https://github.com/mchandak29/robotic-palm/blob/master/EE304_ProjectReport.pdf)

## Software Visuals
<p align="center">
    <img src="https://github.com/mchandak29/robotic-palm/blob/master/cpm_hand.gif", width="480">
</p>

This is the **Tensorflow** implementation of [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release), one of the state-of-the-art models for **2D body and hand pose estimation**.

<p align="center">
    <img src="https://github.com/mchandak29/robotic-palm/blob/master/cpm_hand_with_tracker.gif", width="480">
</p>

Tracking support for single hand.

### With some additional features:
 - Easy multi-stage graph construction
 - Kalman filters for smooth pose estimation
 - Simple self-tracking module

## Results Compilation
<p align="center">
    <img src="https://github.com/mchandak29/robotic-palm/blob/master/Screenshot%20from%202019-08-02%2015-40-24.png", width="480">
</p>
