#include "Vio/Initialize/InitTypes.hpp"

namespace hybrid_msckf {

Frame::Frame() {
    time_ = -1;
    R_ = cv::Mat::eye(3, 3, CV_64F);
    rvec_ = cv::Mat::zeros(3, 1, CV_64F);
    t_ = cv::Mat::zeros(3, 1, CV_64F);
    features_.clear();
}

Frame::Frame(double time, const cv::Mat &imgL) {
    time_ = time;
    frameId_ = -1;
    R_ = cv::Mat::eye(3, 3, CV_64F);
    rvec_ = cv::Mat::zeros(3, 1, CV_64F);
    t_ = cv::Mat::zeros(3, 1, CV_64F);
    features_.clear();
    width_ = imgL.cols;
    height_ = imgL.rows;
    imgL_ = imgL.clone();
    cv::buildOpticalFlowPyramid(imgL, pyramidL_, cv::Size(21, 21), 3, true,
                                cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
}

Frame::Frame(const Frame &f) {
    this->imu_state = f.imu_state;
    this->width_ = f.width_;
    this->height_ = f.height_;
    this->time_ = f.time_;
    if(!f.Image().empty()) {
        f.Image().copyTo(this->imgL_);
    }
    //std::vector<cv::Mat> imgRs;
    f.R_.copyTo(this->R_);
    f.t_.copyTo(this->t_);
    f.rvec_.copyTo(this->rvec_);
    this->features_ = f.features_;
    this->idList_ = f.idList_;
    this->frameId_ = f.frameId_;
}

const int& Frame::Width() const {
    return width_;
}

const int& Frame::Height() const {
    return height_;
}

const cv::Mat& Frame::R() const  {
    return R_;
}

const cv::Mat &Frame::Rvec() const {
    return rvec_;
}

const cv::Mat& Frame::T() const {
    return t_;
}

const int &Frame::FrameId() const {
    return frameId_;
}

const std::vector<Feature> &Frame::Features() const {
    return features_;
}

std::vector<Feature> &Frame::Features(){
    return features_;
}

const cv::Mat &Frame::Image() const {
    return imgL_;
}

std::vector<int> &Frame::IDList() {
    return idList_;
}

void Frame::SetPose(const cv::Mat &R, const cv::Mat &t) {
    R.copyTo(R_);
    t.copyTo(t_);
    cv::Rodrigues(R_, rvec_);
}

void Frame::SetPose(const cv::Vec6d &pose) {
    rvec_.at<double>(0, 0) = pose[0];
    rvec_.at<double>(1, 0) = pose[1];
    rvec_.at<double>(2, 0) = pose[2];
    t_.at<double>(0, 0) = pose[3];
    t_.at<double>(1, 0) = pose[4];
    t_.at<double>(2, 0) = pose[5];
    cv::Rodrigues(rvec_, R_);
}

/*
 * get pose
 * @param R: Rotation Matrix
 * @param t: translation vector
 */
void Frame::GetPose(cv::Mat &R, cv::Mat &t) const {
    R_.copyTo(R);
    t_.copyTo(t);
}

/*
 * get psoe in 6d vector
 * return: 6d vector
 */
cv::Vec6d Frame::GetPose() const {
    cv::Vec6d pose;
    pose[0] = rvec_.at<double>(0, 0);
    pose[1] = rvec_.at<double>(1, 0);
    pose[2] = rvec_.at<double>(2, 0);
    pose[3] = t_.at<double>(0, 0);
    pose[4] = t_.at<double>(1, 0);
    pose[5] = t_.at<double>(2, 0);
    return pose;
}

void Frame::SetFeatures(const std::vector<Feature> &features) {
    features_ = features;
}

void Frame::SetTime(const double &time) {
    time_ = time;
}

void Frame::SetFrameId(int frameId) {
    frameId_ = frameId;
}

void Frame::ImageRelease() {
    if(!imgL_.empty())imgL_.release();

    for(auto &img : pyramidL_) {
        if(!img.empty())img.release();
    }
};

int Frame::CountCommonFeatures(const Frame &refFrame) const {
    const std::vector<Feature> &features = refFrame.Features();
    int cnt = 0;
    for (size_t i1 = 0; i1 < features.size(); ++i1) {
        for (size_t i2 = 0; i2 < features_.size(); ++i2) {
            if (features[i1].featureID == features_[i2].featureID) {
                ++cnt;
                break;
            }
        }
    }
    return cnt;
}

/*  Map  */

bool Map::Init() {
    keyframes_.clear();
    mapPoints_.clear();
    return true;
}

void Map::Clear() {
    keyframes_.clear();
    mapPoints_.clear();
}

void Map::InsertKeyframe(const FramePtr &frame) {
    const std::vector<Feature> &features = frame->Features();
    if (features.size() == 0) {
        return;
    }
    // check observations size first.
    if (static_cast<int>(mapPoints_.size()) < features[features.size() - 1].featureID + 1) {
        int n = static_cast<int>(mapPoints_.size());
        mapPoints_.resize(features[features.size() - 1].featureID + 1);
        for (int i = n; i < static_cast<int>(mapPoints_.size()); ++i) {
            mapPoints_[i] = std::make_shared<MapPoint>();
        }
    }
    // assign objToObservation
    for (int i = 0; i < static_cast<int>(features.size()); ++i) {
        if (static_cast<int>(mapPoints_.size()) < features[i].featureID + 1) {
            int n = static_cast<int>(mapPoints_.size());
            mapPoints_.resize(features[i].featureID + 1);
            for (int i = n; i < static_cast<int>(mapPoints_.size()); ++i) {
                mapPoints_[i] = std::make_shared<MapPoint>();
            }
        }
        int featureID = features[i].featureID;
        mapPoints_[featureID]->features.push_back(features[i]);
    }
    keyframes_.push_back(frame);
}

void Map::PopKeyframe(const FramePtr frame) {
    const std::vector<Feature> &features = frame->Features();
    for (int i = 0; i < features.size(); ++i) {
        int featureID = features[i].featureID;
        if (mapPoints_.size() > featureID) {
            size_t n = mapPoints_[featureID]->features.size();
            if (n > 0) {
                mapPoints_[featureID]->features.resize(n - 1);
                if (n <= 2) {
                    // stereo mappoint
                    if(mapPoints_[featureID]->features[0].imageID == 0 && mapPoints_[featureID]->features[0].depth > 0) {
                        continue;
                    }
                    mapPoints_[featureID]->pt3dFlag = false;
                }
            }
        }
    }
    if (keyframes_.size() > 0) {
        keyframes_.resize(keyframes_.size() - 1);
    }
}


const std::vector<FramePtr> &Map::GetKeyFrames() const {
    return keyframes_;
}



} // namespace hybrid_msckf
