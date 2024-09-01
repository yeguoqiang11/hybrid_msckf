#include "Vio/Initialize/Imu/Imu.h"

namespace hybrid_msckf {

inline Eigen::Matrix3d RotationVector2Matrix3d(Eigen::Vector3d rvec) {
    double theta = rvec.norm();
    if (fabs(theta) < 1.0e-08) {
        if (theta < 0) {
            theta = -1.0e-08;
        } else {
            theta = 1.0e-08;
        }
    }
    Eigen::Vector3d nvec = rvec / theta;
    Eigen::Matrix3d n_hat;
    n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
    Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
    return R;
}

inline Eigen::Vector3d RotationMatrix2Vector3d(Eigen::Matrix3d R) {
    double cos_theta = (R.trace() - 1.0) * 0.5;  
    double theta = acos(cos_theta);
    Eigen::Matrix3d Right = (R - R.transpose()) * 0.5;
    Eigen::Vector3d rvec;
    rvec[0] = (Right(2, 1) - Right(1, 2)) * 0.5;
    rvec[1] = (Right(0, 2) - Right(2, 0)) * 0.5;
    rvec[2] = (Right(1, 0) - Right(0, 1)) * 0.5;
    if (cos_theta > 1.0 - 1.0e-07) {
        return Eigen::Vector3d(0, 0, 0);
    }
    if (cos_theta < 1.0e-07 - 1.0) {
        return 3.1415926 * rvec / rvec.norm();
    }
    rvec /= rvec.norm();

    return rvec * theta;
}

inline Eigen::Matrix3d Jr(Eigen::Vector3d rvec) {
    double angle = rvec.norm();

    if (angle < 1.0e-06) {
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Matrix3d r_hat;
    r_hat << 0.0, -rvec(2), rvec(1), rvec(2), 0.0, -rvec(0), -rvec(1), rvec(0), 0.0;

    Eigen::Matrix3d J = Eigen::Matrix3d::Identity() - r_hat * (1.0 - cos(angle)) / (angle * angle)
                        + r_hat * r_hat * (angle - sin(angle)) / (angle * angle * angle);
    return J;
}

Preintegrated::Preintegrated(const Eigen::Vector3d &ba,
                             const Eigen::Vector3d &bg,
                             double an,
                             double gn,
                             double ban,
                             double bgn,
                             double input_acc_mag) {
    Nga_.setIdentity();
    Nga_.block<3, 3>(0, 0) *= gn * gn;
    Nga_.block<3, 3>(3, 3) *= an * an;
    Nbga_.setIdentity();
    Nbga_.block<3, 3>(0, 0) *= bgn * bgn;
    Nbga_.block<3, 3>(3, 3) *= ban * ban;
    ba_ = ba;
    bg_ = bg;
    input_ratio_ = 9.8 / input_acc_mag;
    Rji_.setIdentity();
    Pij_.setZero();
    Vij_.setZero();
    dRg_.setZero();
    dVg_.setZero();
    dVa_.setZero();
    dPa_.setZero();
    dPg_.setZero();
    dT_ = 0;
    cov_.setZero();
    raw_data_.clear();
}

void Preintegrated::Initialize() {
    Rji_.setIdentity();
    Pij_.setZero();
    Vij_.setZero();
    dRg_.setZero();
    dVg_.setZero();
    dVa_.setZero();
    dPa_.setZero();
    dPg_.setZero();
    dT_ = 0;
    cov_.setZero();
}

// residual: theta, Vij, Pij;
// cov: theta, Vij, Pij, bg, ba;
void Preintegrated::IntegrateNewImu(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro, const double dt) {
    Eigen::Matrix<double, 9, 9> A;
    Eigen::Matrix<double, 9, 6> B;
    A.setIdentity();
    B.setZero();
    Eigen::Vector3d acc0 = (input_ratio_ * acc - ba_);
    Eigen::Vector3d gyro0 = gyro - bg_;

    Pij_ = Pij_ + Vij_ * dt + 0.5 * Rji_ * acc0 * dt * dt;
    Vij_ = Vij_ + Rji_ * acc0 * dt;

    Eigen::Matrix3d hat_acc;
    hat_acc << 0.0, -acc0(2), acc0(1), acc0(2), 0.0, -acc0(0), -acc0(1), acc0(0), 0.0;
    A.block<3, 3>(3, 0) = -Rji_ * hat_acc * dt;
    A.block<3, 3>(6, 0) = -0.5 * Rji_ * hat_acc * dt * dt;
    A.block<3, 3>(6, 3).setIdentity();
    A.block<3, 3>(6, 3) *= dt;
    B.block<3, 3>(3, 3) = Rji_ * dt;
    B.block<3, 3>(6, 3) = 0.5 * Rji_ * dt * dt;

    dPa_ = dPa_ + dVa_ * dt - 0.5 * Rji_ * dt * dt;
    dPg_ = dPg_ + dVg_ * dt - 0.5 * Rji_ * hat_acc * dt * dt * dRg_;
    dVa_ = dVa_ - Rji_ * dt;
    dVg_ = dVg_ - Rji_ * hat_acc * dt * dRg_;

    Eigen::Matrix3d dR = RotationVector2Matrix3d(gyro0 * dt);
    Eigen::Matrix3d Jrv = Jr(gyro0 * dt);
    Rji_ = Rji_ * dR;
    A.block<3, 3>(0, 0) = dR.transpose();
    B.block<3, 3>(0, 0) = Jrv * dt;

    cov_.block<9, 9>(0, 0) = A * cov_.block<9, 9>(0, 0) * A.transpose() + B * Nga_ * B.transpose();
    cov_.block<6, 6>(9, 9) += Nbga_ * dt * dt;

    dRg_ = dRg_ - dR.transpose() * Jrv * dt;

    dT_ += dt;
    raw_data_.push_back(ImuData(acc, gyro, dt));
}

void Preintegrated::RePreintegration() {
    Initialize();
    std::vector<ImuData> imu_data;
    imu_data.swap(raw_data_);

    for (int i = 0; i < static_cast<int>(imu_data.size()); i++) {
        ImuData &data = imu_data[i];
        IntegrateNewImu(data.acc_, data.gyro_, data.dt_);
    }
}

void Preintegrated::InsertNewImu(const Preintegrated &imu) {
    raw_data_.insert(raw_data_.end(), imu.raw_data_.begin(), imu.raw_data_.end());
}

Eigen::Vector3d Preintegrated::GetAvgAcc() {
    Eigen::Vector3d acc(0, 0, 0);
    for (int i = 0; i < static_cast<int>(raw_data_.size()); i++) {
        acc += raw_data_[i].acc_;
    }
    acc *= input_ratio_;
    acc /= static_cast<double>(raw_data_.size());
    return acc;
}

double Preintegrated::GetMoveAccStd() {
    double sum = 0;
    for (int i = 0; i < raw_data_.size(); i++) {
        double err = raw_data_[i].acc_.norm() * input_ratio_ - 9.8;
        sum += err * err;
    }
    sum = sqrt(sum / raw_data_.size());
    return sum;
}

}  // namespace hybrid_msckf
