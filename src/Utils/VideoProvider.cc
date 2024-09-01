//
// Created by d on 6/8/20.
//

#include "Utils/VideoProvider.h"

using namespace std;

namespace hybrid_msckf {

VideoProvider::VideoProvider(const string &video_path,
                                          const string &times_path,
                                          double time_scale,
                                          const string &img_type)
    : videoPath_(video_path), timesPath_(times_path), timeScale_(time_scale),
      isVideo_(true), imgType_(img_type), frameIdx_(0), fps_(-1)
{
    // open video file
    if (EndsWith(video_path, ".avi") || EndsWith(video_path, ".mp4") ) {
        videoCap_.open(video_path);
        if (!videoCap_.isOpened()) {
            std::cerr << "can't open video file: " << video_path << "\n";
            exit(-1);
        }
        isVideo_ = true;
    } else {
        isVideo_ = false;
    }

    LoadTimestamps();
    totalFrames_ = static_cast<int>(timestamps_.size() );

    GetSize();
}


VideoProvider::VideoProvider(const string &video_path, double fps)
    : videoPath_(video_path), timesPath_(""), timeScale_(1.0),
      isVideo_(true), frameIdx_(0), fps_(fps)
{
    videoCap_.open(video_path);
    if (!videoCap_.isOpened()) {
        std::cerr << "can't open video file: " << video_path << "\n";
        exit(-1);
    }
    totalFrames_ = 0;

    GetSize();
}


void VideoProvider::GetSize() {
    cv::Mat img;
    double time;
    if (!Read(img, time)) {
        cerr << "can't get the first frame!" << endl;
        exit(-1);
    }
    width_ = img.cols;
    height_ = img.rows;
    Reset();
}


void VideoProvider::Reset() {
    if (isVideo_) {
        if (videoCap_.isOpened()) {
            videoCap_.release();
        }
        videoCap_.open(videoPath_);
        if (!videoCap_.isOpened()) {
            std::cerr << "can't open video file: " << videoPath_ << "\n";
            exit(-1);
        }
    }
    frameIdx_ = 0;
}


bool VideoProvider::Read(cv::Mat &img, double &timestamp) {
    if (frameIdx_ >= totalFrames_) {
        return false;
    }

    if (isVideo_) {
        videoCap_.read(img);
    } else {
        std::string img_path = videoPath_ + "/" + timeStrs_[frameIdx_] + imgType_;
        img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    }

    if (timestamps_.empty()) {
        timestamp = static_cast<double>(frameIdx_) / fps_;
    } else {
        timestamp = timestamps_.at(frameIdx_);
    }

    frameIdx_++;

    return !img.empty();
}


bool VideoProvider::Skip(int n) {
    if (isVideo_) {
        for (int i = 0; i < n; ++i) {
            if (!videoCap_.grab()) {
                cerr << "[VideoProvider::Skip] reaches video end!" << endl;
                break;
            }
        }
    }
    frameIdx_ = std::min(frameIdx_ + n, totalFrames_);
    return true;
}


void VideoProvider::LoadTimestamps() {
    std::ifstream ifs(timesPath_);
    if (!ifs.is_open()) {
        std::cerr << "can't open timestamp file: " << timesPath_ << "\n";
        exit(-1);
    }

    string aline, s;
    while (!ifs.eof()) {
        std::getline(ifs, aline);
        if (aline.empty()) {
            break;
        }
        std::stringstream ss(aline);
        ss >> s;

        timeStrs_.push_back(s);
        timestamps_.push_back(std::stod(s) * timeScale_);
    }
}

}//namespace hybrid_msckf {
