#ifndef IMU_H
#define IMU_H
#include <iostream>
#include <vector>

#include <Eigen/Core>

namespace inslam {

struct Preintegrated {
    Preintegrated() { Initialize(); }

    Preintegrated(const Eigen::Vector3d &ba, const Eigen::Vector3d &bg, double an, double gn, double ban, double bgn, double input_acc_mag);

    void Initialize();

    void IntegrateNewImu(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro, const double dt);

    void RePreintegration();

    void InsertNewImu(const Preintegrated &imu);

    Eigen::Vector3d GetAvgAcc();

    double GetMoveAccStd();

    Eigen::Matrix3d GetRji() const { return Rji_; }
    Eigen::Vector3d GetVij() const { return Vij_; }
    Eigen::Vector3d GetPij() const { return Pij_; }
    Eigen::Matrix3d GetdRg() const { return dRg_; }
    Eigen::Matrix3d GetdPa() const { return dPa_; }
    Eigen::Matrix3d GetdPg() const { return dPg_; }
    Eigen::Matrix3d GetdVa() const { return dVa_; }
    Eigen::Matrix3d GetDvg() const { return dVg_; }
    Eigen::Matrix<double, 15, 15> GetCov() const { return cov_; }

public:
    double dT_, input_ratio_;
    Eigen::Vector3d Vij_, Pij_, ba_, bg_;
    Eigen::Matrix3d Rji_, dRg_, dVg_, dVa_, dPg_, dPa_;
    Eigen::Matrix<double, 6, 6> Nga_, Nbga_;
    Eigen::Matrix<double, 15, 15> cov_;

    struct ImuData {
        ImuData(Eigen::Vector3d acc, Eigen::Vector3d gyro, double dt) : acc_(acc), gyro_(gyro), dt_(dt) {}

        Eigen::Vector3d acc_, gyro_;
        double dt_;
    };
    std::vector<ImuData> raw_data_;
};

struct VisualInertialState {
    Preintegrated imu0;
    Eigen::Matrix3d Rwc;
    Eigen::Vector3d twc, Pj, Vj;
    double timestamp = -1, last_imu_time = -1;
    bool is_inertial_ok_ = false;
    int frame_id = -1;
    bool fixed = false;
};

}  // namespace inslam
#endif
