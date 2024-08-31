//
// Created by d on 6/8/20.
//

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <memory>

namespace inslam {

class VideoProvider {
public:
    VideoProvider(const std::string &video_path,
                        const std::string &times_path,
                        double time_scale = 1.0,
                        const std::string &img_type = ".png");

    VideoProvider(const std::string &video_path, double fps);

    bool Read(cv::Mat &img, double &timestamp);

    bool Skip(int n);

    inline int Width() {
        return width_;
    }

    inline int Height() {
        return height_;
    }

private:
    bool EndsWith(const std::string &str, const std::string &suffix) const {
        return str.size() > suffix.size() &&
                  str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    }

    void LoadTimestamps();

    void GetSize();

    void Reset();

    std::string videoPath_;
    std::string timesPath_;
    double timeScale_;
    std::string imgType_;
    bool isVideo_;
    double fps_;

    cv::VideoCapture videoCap_;
    std::vector<std::string> timeStrs_;
    std::vector<double> timestamps_;

    int frameIdx_;
    int totalFrames_;

    int width_;
    int height_;

};

}//namespace inslam {