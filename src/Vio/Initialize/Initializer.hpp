#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "Utils/json.hpp"
#include "Vio/Caimura.hpp"
#include "Vio/Initialize/InitTypes.hpp"
#include "Vio/Initialize/FrameHandler.hpp"

namespace inslam {

class Initializer
{
public:

    Initializer() = delete;
    Initializer(const nlohmann::json &config, std::shared_ptr<Caimura> leftCaim,
                std::shared_ptr<Caimura> rightCaim = nullptr);

    bool Run(const cv::Mat &imgL, double imgTime, std::vector<Eigen::Matrix<double, 7, 1>> &imuList,
             Eigen::Matrix<double, 7, 1> &qv, const cv::Mat &imgR = cv::Mat());
    bool RunVSlam(const cv::Mat &imgL, double imgTime, const cv::Mat &imgR = cv::Mat());
    bool RunCalibrateImu(const cv::Mat &imgL, double imgTime, std::vector<Eigen::Matrix<double, 7, 1>> &imuList,
                         const cv::Mat &imgR = cv::Mat());
    void SetImuParam(double acc_n, double gyro_n, double ba_n, double bg_n);
    void SetImuExtrinsic(const Eigen::Matrix3d &Rcb, const Eigen::Vector3d &tcb);
    void SetImuBa(const Eigen::Vector3d &ba);
    void SetImuBg(const Eigen::Vector3d &bg);
    void Reset();
    void GetCameraPose(cv::Mat &R, cv::Mat &t);
    std::vector<Eigen::Vector3d> GetMapPoint();
    void SetSaveSlamPoseFlag(bool flag);
    //format: r00 r01 r02 r10 r11 r12 r20 r21 r22 t0 t1 t2
    void SaveSlamPose(const std::string &path,
                      Eigen::Matrix4d &T_G_C0,
                      Eigen::Matrix4d &T_C_I);

private:

    std::shared_ptr<FrameHandler> frameHandler_;
    MapPtr map_;
};

} // namespace  inslam

