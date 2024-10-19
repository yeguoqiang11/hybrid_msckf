# hybrid_msckf
# 1. Introduction
hybrid_msckf is a project of integrating imu and camera，using ekf。it contain mono slam and stereo slam. the project use newest imu aligment technology.  the project also develop a pano slam.
we try many camera model include fisheye, pinhole and so on. imu and camera calibration is also included

# 2. Prerequisites
we use at least C++14 and test it on ubuntu 16.04
## OpenCV
we use opencv to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **Required at least 3.4. Tested with OpenCV 3.4.11.**
## Eigen3
we use eigen3 to manipulate matrix and vector. Download and install instructions can be found at http://eigen.tuxfamily.org. **Tested with eigen 3.2.10**
