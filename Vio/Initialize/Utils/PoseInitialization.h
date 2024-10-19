#pragma once
#include <opencv2/opencv.hpp>
#include "Vio/Initialize/InitTypes.hpp"

namespace featslam{
class PoseInitialization {
public:
    PoseInitialization(const std::shared_ptr<Map>& map);
    bool Run();
    bool RunStereo(std::vector<FramePtr> &initFrameList);

private:
    bool GoodPoseInit(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, const cv::Mat &R, const cv::Mat &t);

private:
    std::shared_ptr<Map> map_;
};
}
