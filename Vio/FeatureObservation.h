//
// Created by d on 2021/1/29.
//

#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#define NUM_CAMS

namespace featslam{

struct FeatureObservation {
    FeatureObservation() : stereoDepth(-1), isStereo(false) {}
    size_t id;
    size_t frameId;
    cv::Point2f pt0;
    cv::Point2f upt0;
    Eigen::Vector3d ray0;
    cv::Point2f pt1;
    cv::Point2f upt1;
    Eigen::Vector3d ray1;
    double stereoDepth;
    bool isStereo;
};

} // namespace inslam

