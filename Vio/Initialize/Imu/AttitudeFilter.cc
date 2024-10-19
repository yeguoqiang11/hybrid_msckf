//
// Created by d on 2021/3/18.
//

#include "AttitudeFilter.h"
#include "Utils/MathUtil.h"

using namespace std;

namespace featslam{

AttitudeFilter::AttitudeFilter(const Eigen::Vector3d &gyroBias, const Eigen::Vector3d &accBias)
        : bg_(gyroBias), ba_(accBias) {
    beta_ = 0.1;
    q_.setIdentity();
    isInitialized_ = false;
}


void AttitudeFilter::FeedImu(double t, const Eigen::Vector3d &gyro, const Eigen::Vector3d &acc, bool useGravity) {
    if (!isInitialized_) {
        Eigen::Vector3d a = acc - ba_;
        q_ = MathUtil::GetQfromA(a);
        isInitialized_ = true;
    } else {
        double dt = t - t_;
        Eigen::Vector3d w = (gyro + gyro_) * 0.5 - bg_;
        Eigen::Vector3d a = acc - ba_;
        if (useGravity) {
            Update(dt, w(0), w(1), w(2), a(0), a(1), a(2));
        } else {
            Integrate(dt, w(0), w(1), w(2));
        }
    }

    t_ = t;
    gyro_ = gyro;
    acc_ = acc;
}


void AttitudeFilter::Integrate(double dt, double gx, double gy, double gz) {
    Eigen::Quaterniond dq = MathUtil::VecToQuat(Eigen::Vector3d(gx, gy, gz) * dt);
    Eigen::Quaterniond newq = q_ * dq;
    q_ = newq;
}


void AttitudeFilter::Update(double dt, double gx, double gy, double gz, double ax, double ay, double az) {
    double q0 = q_.w(), q1 = q_.x(), q2 = q_.y(), q3 = q_.z();
    double s0, s1, s2, s3;
    double qDot1, qDot2, qDot3, qDot4;
    double _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2, _8q1, _8q2, q0q0, q1q1, q2q2, q3q3;
    double norm, recipNorm;

    // Rate of change of quaternion from gyroscope
    qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
    qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
    qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
    qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

    // Compute feedback only if accelerometer measurement valid
    // (avoids NaN in accelerometer normalisation)
    norm = sqrt(ax * ax + ay * ay + az * az);
    if (norm > 0.00001) {

        // Normalise accelerometer measurement
        recipNorm = 1.0 / norm;
        ax *= recipNorm;
        ay *= recipNorm;
        az *= recipNorm;

        // Auxiliary variables to avoid repeated arithmetic
        _2q0 = 2.0 * q0;
        _2q1 = 2.0 * q1;
        _2q2 = 2.0 * q2;
        _2q3 = 2.0 * q3;
        _4q0 = 4.0 * q0;
        _4q1 = 4.0 * q1;
        _4q2 = 4.0 * q2;
        _8q1 = 8.0 * q1;
        _8q2 = 8.0 * q2;
        q0q0 = q0 * q0;
        q1q1 = q1 * q1;
        q2q2 = q2 * q2;
        q3q3 = q3 * q3;

        // Gradient decent algorithm corrective step
        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
        s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
        s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay;

        norm = sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
        if (norm > 0.00001) {
            recipNorm = 1.0 / norm; // normalise step magnitude
            s0 *= recipNorm;
            s1 *= recipNorm;
            s2 *= recipNorm;
            s3 *= recipNorm;

            // Apply feedback step
            qDot1 -= beta_ * s0;
            qDot2 -= beta_ * s1;
            qDot3 -= beta_ * s2;
            qDot4 -= beta_ * s3;
        }
    }

    // Integrate rate of change of quaternion to yield quaternion
    q0 += qDot1 * dt;
    q1 += qDot2 * dt;
    q2 += qDot3 * dt;
    q3 += qDot4 * dt;

    // Normalize quaternion
    Eigen::Vector4d newq(q0, q1, q2, q3);
    newq.normalize();

    // Update q
    q_ = Eigen::Quaterniond(newq(0), newq(1), newq(2), newq(3));

}


Eigen::Vector3d AttitudeFilter::GetEulerAngles() {
    double q0 = q_.w(), q1 = q_.x(), q2 = q_.y(), q3 = q_.z();
    double roll = atan2(q0 * q1 + q2 * q3, 0.5 - q1 * q1 - q2 * q2);
    double pitch = asin(-2.0 * (q1 * q3 - q0 * q2));
    double yaw = atan2(q1 * q2 + q0 * q3, 0.5 - q2 * q2 - q3 * q3);
    return {roll, pitch, yaw};
}

} // namespace inslam