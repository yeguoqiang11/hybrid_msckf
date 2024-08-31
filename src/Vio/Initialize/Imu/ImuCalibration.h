#ifndef IMUCALIBRATION_H
#define IMUCALIBRATION_H
#include <assert.h>
#include <unordered_map>
#include <Eigen/Core>

#include "Vio/Initialize/Imu/Imu.h"
#include "Vio/Initialize/InitTypes.hpp"

namespace inslam {
class ImuCalibration {
public:
    ImuCalibration(MapPtr map, double angularResolution) : map_(map) {
        bg_.setZero();
        Rcb_.setIdentity();
        angularResolution_ = angularResolution;
        isSuccess_ = false;
        startCalibFrameNum_ = 500;
        calibDegThres_ = 25;  // degree
        alignTimeStep_ = 0.006;
        maxDeltaT_ = 5.0;
        minDeltaT_ = -5.0;
    }

    ~ImuCalibration() {}

    bool Calibrate();

    bool IsSuccess() { return isSuccess_; }

    bool Reset();

    Eigen::Matrix3d GetRotationExtrinsic() { return Rcb_; }

    void SetRotation(const Eigen::Matrix3d &R) {
        Rcb_ = R;
    }

    void SetBg(const Eigen::Vector3d &bg) {
        bg_ = bg;
    }

    void InsertFrame(FramePtr &frame) {
        frameDataset_.push_back(frame);
    }

    void InsertImu(const Eigen::Vector3d &gyro, const Eigen::Vector3d &acc, double timestamp) {
        Eigen::Matrix<double, 7, 1> imuData;
        imuData << timestamp, gyro - bg_, acc;
        if(!imuDataList_.empty() && timestamp - imuDataList_.back()(0) < 1e-7) {
            return;
        }
        imuDataList_.push_back(imuData);
    }

private:
    void CalibrateRotAndTime(double rawDeltaT,
                             std::vector<Eigen::Matrix3d> &RcjiList,
                             std::vector<double> &frameTimeList,
                             std::vector<Eigen::Matrix3d> &RImuList);

    bool ExtrinsicCalibration(std::vector<Eigen::Matrix3d> &RcList, std::vector<Eigen::Matrix3d> &RimuList,
                                              Eigen::Matrix3d &calibRcbResult, double &error);

    double CalibrateRawTime(std::vector<Eigen::Matrix3d> &RcjiList,
                            std::vector<double> &frameTimeList,
                            std::vector<Eigen::Matrix3d> &RImuList);

    void InterpolationImuPose(double dT,
                              std::vector<Eigen::Matrix3d> &RcjiList,
                              std::vector<double> &frameTimeList,
                              std::vector<Eigen::Matrix3d> &RImuList,
                              std::vector<Eigen::Matrix3d> &RcList,
                              std::vector<Eigen::Matrix3d> &RgList);

    bool CalibrationExRotation(std::vector<Eigen::Matrix3d> &Rc, std::vector<Eigen::Matrix3d> &Rimu,
                               Eigen::Matrix3d &calib_ric_result, double &error);


    void IntegrateGyroRK4(double dt, const Eigen::Vector3d &w0,
                        const Eigen::Vector3d &w1, Eigen::Vector4d &dq);

private:
    std::vector<FramePtr> frameDataset_;
    std::vector<Eigen::Matrix<double, 7, 1>> imuDataList_;
    double alignTimeStep_, maxDeltaT_, minDeltaT_;
    Eigen::Matrix3d Rcb_;
    double calibDegThres_;
    bool isSuccess_;
    Eigen::Vector3d bg_;
    int startCalibFrameNum_;
    double angularResolution_;
    MapPtr map_;
};

}  // namespace inslam
#endif
