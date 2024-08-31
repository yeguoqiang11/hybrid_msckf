#include "Vio/Initialize/Initializer.hpp"

#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>

namespace inslam {
Initializer::Initializer(const nlohmann::json &config, std::shared_ptr<Caimura> leftCaim,
                         std::shared_ptr<Caimura> rightCaim) {
    map_ = std::make_shared<Map>();
    frameHandler_ = std::make_shared<FrameHandler>(map_, config, leftCaim, rightCaim);

    SetImuParam(0.1, 0.005, 0.005, 0.0001);
    Eigen::Matrix3d Rcb = leftCaim->Ric_;
    Eigen::Vector3d tcb = leftCaim->pic_;
    SetImuExtrinsic(Rcb, tcb);
    SetImuBa(leftCaim->ba_);
    SetImuBg(leftCaim->bg_);
}


bool Initializer::Run(const cv::Mat &imgL, double imgTime, std::vector<Eigen::Matrix<double, 7, 1>> &imuList,
                      Eigen::Matrix<double, 7, 1> &qv, const cv::Mat &imgR) {
    for(auto imu : imuList) {
        double gyro[3] = {imu(1), imu(2), imu(3)};

        imu.tail(3) /= 10;
        double acc[3] = {imu(4), imu(5), imu(6)};
        frameHandler_->InsertImu(gyro, acc, imu(0));
    }
    if(frameHandler_->Run(imgL, imgTime, imgR)) {
        qv = frameHandler_->GetQV();
        frameHandler_->Reset();
        return true;
    }

    return  false;
}


bool Initializer::RunVSlam(const cv::Mat &imgL, double imgTime, const cv::Mat &imgR) {
    return frameHandler_->Run(imgL, imgTime, imgR);
}


bool Initializer::RunCalibrateImu(const cv::Mat &imgL, double imgTime, std::vector<Eigen::Matrix<double, 7, 1>> &imuList,
                                  const cv::Mat &imgR) {
    frameHandler_->EnableCalibrateImu();
    for(auto imu : imuList) {
        double gyro[3] = {imu(1), imu(2), imu(3)};

        imu.tail(3) /= 10;
        double acc[3] = {imu(4), imu(5), imu(6)};
        frameHandler_->InsertImu(gyro, acc, imu(0));
    }
    return frameHandler_->Run(imgL, imgTime, imgR);
}


void Initializer::SetImuParam(double acc_n, double gyro_n, double ba_n, double bg_n) {
    frameHandler_->SetImuParam(acc_n, gyro_n, ba_n, bg_n);
}


void Initializer::SetImuExtrinsic(const Eigen::Matrix3d &Rcb, const Eigen::Vector3d &tcb) {
    frameHandler_->SetImuExtrinsic(Rcb, tcb);
}


void Initializer::SetImuBa(const Eigen::Vector3d &ba) {
    frameHandler_->SetInitBa(ba);
}


void Initializer::SetImuBg(const Eigen::Vector3d &bg) {
    frameHandler_->SetInitBg(bg);
}


void Initializer::Reset() {
    frameHandler_.reset();
    map_.reset();
}


void Initializer::GetCameraPose(cv::Mat &R, cv::Mat &t) {
    frameHandler_->GetCameraPose(R, t);
}


std::vector<Eigen::Vector3d> Initializer::GetMapPoint() {
    return frameHandler_->GetMapPoint();
}


void Initializer::SetSaveSlamPoseFlag(bool flag) {
    frameHandler_->SetSaveSlamPoseFlag(flag);
}


void Initializer::SaveSlamPose(const std::string &path,
                               Eigen::Matrix4d &T_G_C0,
                               Eigen::Matrix4d &T_C_I) {
    frameHandler_->SaveSlamPose(path, T_G_C0, T_C_I);
}

} // namespace inslam

