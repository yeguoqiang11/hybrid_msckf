# hybrid_msckf
# 1. Introduction
hybrid_msckf is a project of integrating imu and camera，using msckf contain mono slam and stereo slam. the project use newest imu aligment technology.  

# 2. Prerequisites
we use at least C++14 and test it on ubuntu 16.04
## OpenCV
we use opencv to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **Required at least 3.4. Tested with OpenCV 3.4.11.**
## Eigen3
we use eigen3 to manipulate matrix and vector. Download and install instructions can be found at http://eigen.tuxfamily.org. **Tested with eigen 3.2.10**

# 3. Function

## stereo camera
the project support both mono and stereo camera.

## rolling shutter camera
we have switch about using rolling shutter camera. users can use by open use_rolling_shutter_calib_.

## delay time estimation between camera and imu.
using tradictional delay time estimation, users can open by use_td_calib_.

## extrinsic online calibration
translation and rotation between imu and camera are also estimated online. you can open use_extrinsic_calib_

other things you can get in this project are static initialization, dynamic initialization, attitude estimation by imu, FEJ and OC，also adding feature depth to state estimation.
