#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "Vio/Initialize/Imu/Imu.h"

namespace hybrid_msckf {

struct Feature {
    int featureID;
    int imageID;
    cv::Point2f pt;
    cv::Point3f spherePt1;
    cv::Point3f spherePt2;
    double depth; //norm
};


struct MapPoint {
    cv::Point3f pt3d;
    std::vector<Feature> features;
    bool pt3dFlag = false;
};
typedef std::shared_ptr<MapPoint> MapPointPtr;


class Frame {
public:
    Frame();
    Frame(double time, const cv::Mat &imgL);
    //Frame &operator=(const Frame &f);
    Frame(const Frame &f);

    void SetPose(const cv::Mat &R, const cv::Mat &t);
    void SetPose(const cv::Vec6d &pose);
    void SetFeatures(const std::vector<Feature> &features);    
    void SetTime(const double &time);
    void SetFrameId(int frameId);
    void SetTraslationScale(double scale) { t_ *= scale; }

    // get
    const int& Width() const;
    const int& Height() const;
    const cv::Mat& R() const;
    const cv::Mat& T() const;
    const cv::Mat &Rvec() const;
    const double &Time() const { return time_; }
    const int &FrameId() const;
    const cv::Mat &Image() const;
    const std::vector<cv::Mat> &LeftPyramid() const { return pyramidL_; }
    const std::vector<Feature> &Features() const;
    std::vector<Feature> &Features();
    std::vector<int> &IDList();
    void GetPose(cv::Mat &R, cv::Mat &t) const;
    cv::Vec6d GetPose() const;

    void ImageRelease();
    int CountCommonFeatures(const Frame &refFrame) const;

public:
    VisualInertialState imu_state;

private:
    int width_;
    int height_;
    double time_;
    cv::Mat imgL_;
    std::vector<cv::Mat> pyramidL_;
    cv::Mat R_;
    cv::Mat rvec_;
    cv::Mat t_;
    std::vector<Feature> features_;
    std::vector<int> idList_;
    int frameId_;
};
typedef std::shared_ptr<Frame> FramePtr;


class Map {
public:
    bool Init();
    void Clear();
    void InsertKeyframe(const FramePtr &frame);
    void PopKeyframe(const FramePtr frame);
    const std::vector<FramePtr> &GetKeyFrames() const;

public:
    std::vector<MapPointPtr> mapPoints_;
    std::vector<FramePtr> keyframes_;
};
typedef std::shared_ptr<Map> MapPtr;


} // namespace hybrid_msckf
