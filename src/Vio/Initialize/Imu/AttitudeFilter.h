//
// Created by d on 2021/3/18.
//

#pragma once

#include <iostream>
#include <Eigen/Dense>

namespace hybrid_msckf {

class AttitudeFilter {
public:
    AttitudeFilter(const Eigen::Vector3d &gyroBias, const Eigen::Vector3d &accBias);

    void FeedImu(double t, const Eigen::Vector3d &gyro, const Eigen::Vector3d &acc, bool useGravity = true);

    inline Eigen::Quaterniond GetQuat() {
        return q_;
    }

    Eigen::Vector3d GetEulerAngles();

private:
    // dt: s, gyro: rad/s, accel: m/(s^2)
    void Update(double dt, double gx, double gy, double gz, double ax, double ay, double az);

    // Integrate gyro data
    void Integrate(double dt, double gx, double gy, double gz);

    // IMU bias
    Eigen::Vector3d bg_;
    Eigen::Vector3d ba_;

    double beta_;

    double t_;
    Eigen::Vector3d gyro_;
    Eigen::Vector3d acc_;

    Eigen::Quaterniond q_;

    bool isInitialized_;
};

} // namespace hybrid_msckf