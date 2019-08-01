# robotic-palm
This project aims to imitate an actual palm in a 3d printed robotic palm using sensory camera input.

<p align="center">
    <img src="https://github.com/mchandak29/robotic-palm/blob/master/cpm_hand.gif", width="480">
</p>

This is the **Tensorflow** implementation of [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release), one of the state-of-the-art models for **2D body and hand pose estimation**.

<p align="center">
    <img src="https://github.com/mchandak29/robotic-palm/blob/master/cpm_hand_with_tracker.gif", width="480">
</p>

Tracking support for single hand.

## With some additional features:
 - Easy multi-stage graph construction
 - Kalman filters for smooth pose estimation
 - Simple self-tracking module

## Environments
 - Windows 10 / Ubuntu 16.04
 - Tensorflow 1.4.0
 - OpenCV 3.2
