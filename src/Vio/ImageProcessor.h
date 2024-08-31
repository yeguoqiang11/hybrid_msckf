//
// Created by d on 2021/1/22.
//

#pragma once

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "Vio/Caimura.hpp"
#include "Vio/FeatureObservation.h"
#include "Vio/FeatureGrid.hpp"
#include "Utils/json.hpp"

namespace inslam {

class ImageProcessor {
public:
    ImageProcessor(const nlohmann::json &config, std::shared_ptr<Caimura> cam);

    bool Process(const cv::Mat &I, size_t frameId, double timestamp,
                      const cv::Mat &Ir = cv::Mat(),
                      std::shared_ptr<Caimura> rightCam = nullptr);

    std::vector<FeatureObservation> GetActiveFeatures();

    void DrawDebugInfo(cv::Mat &img);

    enum Status {
        NOT_INITIALIZED = 0,
        TRACKING = 1,
        LOST = 2
    };

    enum DetectorType {
        FAST = 0,
        GFTT = 1 // good features to track
    };

private:
    void Track();

    void AddNewFeatures(const cv::Mat &img);

    void StereoFlow(const std::vector<cv::Mat> &pyramid0, const std::vector<cv::Mat> &pyramid1,
                    const std::shared_ptr<Caimura> &cam0, const std::shared_ptr<Caimura> &cam1,
                    const std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts1,
                    std::vector<uchar> &status);

    void CalcStereo(const cv::Mat &Ir, std::shared_ptr<Caimura> &rightCam);

    void CreateImagePyramid(const cv::Mat &I, std::vector<cv::Mat> &pyramid);

    bool CheckEpipolar(const Eigen::Vector3d &ray1, const Eigen::Vector3d &ray2,
                       const Eigen::Matrix3d &E12, double angularThresh);

    void CreateMask();

    // Config parameters
    int maxCorners_;
    int radius_;
    DetectorType detectorType_;
    int fastThreshold_;
    int border_;
    int pyramidLevels_;
    bool useImu_;

    std::shared_ptr<Caimura> cam_;

    // Mask
    cv::Mat mask_;

    // Status
    Status status_;

    // current frame id
    size_t currFrameId_;

    // Image pyramids
    double prevTimestamp_;
    double currTimestamp_;
    std::vector<cv::Mat> prevPyramid_;
    std::vector<cv::Mat> currPyramid_;

    // Feature points
    std::vector<FeatureObservation> features_;
    size_t featureIdGenerator_ = 0;

    // Tracking statistics
    int numTrackedFeatures_ = 0;
    int numNewFeatures_ = 0;
    double trackingRatio_ = 0;

    // Uniform distribution
    std::shared_ptr<FeatureGrid> detectGrid_;
    std::shared_ptr<FeatureGrid> trackGrid_;
};

}//namespace inslam {
