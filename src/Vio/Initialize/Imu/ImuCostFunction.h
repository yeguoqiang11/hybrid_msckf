#ifndef IMUCOSTFUNCTION_H
#define IMUCOSTFUNCTION_H
#include <ceres/sized_cost_function.h>
#include <Eigen/Core>
#include <ceres/ceres.h>

#include "Vio/Initialize/Imu/Imu.h"

namespace inslam {
// parameters: [rvi, ti] [vi, bgi, bai] [rvj, tj] [vj, bgj, baj]
// residual: [dtheta dv dp dbg dba]
class ImuCost : public ceres::SizedCostFunction<15, 6, 9, 6, 9> {
public:
    ImuCost(Eigen::Matrix<double, 15, 15> sqrt_info, Eigen::Vector3d Gw, Eigen::Matrix3d Rcb, Eigen::Vector3d tcb, Preintegrated imu)
        : sqrt_info_(sqrt_info), Gw_(Gw), tcb_(tcb), Rcb_(Rcb), imu_(imu) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::VectorXd> posei(parameters[0], 6);
        Eigen::Map<const Eigen::VectorXd> vbi(parameters[1], 9);
        Eigen::Map<const Eigen::VectorXd> posej(parameters[2], 6);
        Eigen::Map<const Eigen::VectorXd> vbj(parameters[3], 9);

        Eigen::Matrix3d Rbiw = RotationVector2Matrix3d(posei.topRows(3));
        Eigen::Matrix3d Rbjw = RotationVector2Matrix3d(posej.topRows(3));

        Eigen::Vector3d dbg = vbi.segment<3>(3) - imu_.bg_;
        Eigen::Vector3d dba = vbi.segment<3>(6) - imu_.ba_;

        Eigen::Matrix3d corrected_Rji = imu_.Rji_ * RotationVector2Matrix3d(imu_.dRg_ * dbg);
        Eigen::Matrix3d err_R = corrected_Rji.transpose() * Rbiw.transpose() * Rbjw;
        Eigen::Vector3d err_rvec = RotationMatrix2Vector3d(err_R);

        Eigen::Vector3d corrected_Pij = imu_.Pij_ + imu_.dPa_ * dba + imu_.dPg_ * dbg;
        Eigen::Vector3d corrected_Vij = imu_.Vij_ + imu_.dVa_ * dba + imu_.dVg_ * dbg;

        const double &dt = imu_.dT_;
        Eigen::Vector3d dv = Rbiw.transpose() * (vbj.segment<3>(0) - vbi.segment<3>(0) + Gw_ * dt);
        Eigen::Vector3d dp = Rbiw.transpose() * (posej.segment<3>(3) - posei.segment<3>(3) - vbi.segment<3>(0) * dt + 0.5 * Gw_ * dt * dt);

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = err_rvec;
        residual.block<3, 1>(3, 0) = dv - corrected_Vij;
        residual.block<3, 1>(6, 0) = dp - corrected_Pij;
        residual.block<3, 1>(9, 0) = vbj.segment<3>(3) - vbi.segment<3>(3);
        residual.block<3, 1>(12, 0) = vbj.segment<3>(6) - vbi.segment<3>(6);
        // std::cout << "Pij: " << imu_.Pij_.transpose() << std::endl;
        // std::cout << "pj: " << posej.segment<3>(3).transpose() << std::endl;
        // std::cout << "pi: " << posei.segment<3>(3).transpose() << std::endl;
        // std::cout << "dp: " << dp.transpose() << std::endl;
        // // std::cout << "c Pij: " << corrected_Pij.transpose() << std::endl;
        // std::cout << "error: " << residual.segment(0, 15).transpose() << std::endl;
        // std::cout << "Rjw: " << posej.topRows(3).transpose() << std::endl;
        // std::cout << "Riw: " << posei.topRows(3).transpose() << std::endl;
        Eigen::Matrix3d Rji_hat = Rbiw.transpose() * Rbjw;
        // std::cout << "Rji: " << RotationMatrix2Vector3d(Rji_hat).transpose() << std::endl;
        // std::cout << "error: " << residual.segment<3>(0).transpose() << std::endl;
        // std::cout << "rvj: " << posej.topRows(3).transpose() << std::endl;

        Eigen::Vector3d dv1 = (vbj.segment<3>(0) - vbi.segment<3>(0) + Gw_ * dt);
        // std::cout << "Rbw: " << Rbiw << std::endl;
        // std::cout << "dv1: " << dv1.transpose() << std::endl;
        // std::cout << "dv: " << dv.transpose() << std::endl;
        // std::cout << "cvji: " << corrected_Vij.transpose() << std::endl;
        // std::cout << "residual: " << residual.segment(0, 9).transpose() << std::endl;
        residual = sqrt_info_ * residual;
        // std::cout << "residual norm: " << residual.norm() << std::endl;

        // std::cout << "dt: " << dt << std::endl;
        // std::cout << "sqrtinfo: \n" << sqrt_info_ << std::endl;

        Eigen::Matrix3d invJr = NewInvJr(err_rvec);
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacob_posei(jacobians[0]);
                jacob_posei.setZero();
                // jacob Rbiw
                jacob_posei.block<3, 3>(0, 0) = -invJr * Rbjw.transpose() * Rbiw;
                jacob_posei.block<3, 3>(3, 0) << 0.0, -dv(2), dv(1), dv(2), 0.0, -dv(0), -dv(1), dv(0), 0.0;
                jacob_posei.block<3, 3>(6, 0) << 0.0, -dp(2), dp(1), dp(2), 0.0, -dp(0), -dp(1), dp(0), 0.0;
                // jacob tbiw
                jacob_posei.block<3, 3>(6, 3) = -Rbiw.transpose();
                jacob_posei = sqrt_info_ * jacob_posei;
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacob_vbi(jacobians[1]);
                jacob_vbi.setZero();
                // jacob vi
                jacob_vbi.block<3, 3>(3, 0) = -Rbiw.transpose();
                jacob_vbi.block<3, 3>(6, 0) = -Rbiw.transpose() * dt;

                // jacob bgi
                jacob_vbi.block<3, 3>(0, 3) = -invJr * err_R.transpose() * Jr(imu_.dRg_ * dbg) * imu_.dRg_;
                jacob_vbi.block<3, 3>(3, 3) = -imu_.dVg_;
                jacob_vbi.block<3, 3>(6, 3) = -imu_.dPg_;
                jacob_vbi.block<3, 3>(9, 3) = -Eigen::Matrix3d::Identity();
                // jacob bai
                jacob_vbi.block<3, 3>(3, 6) = -imu_.dVa_;
                jacob_vbi.block<3, 3>(6, 6) = -imu_.dPa_;
                jacob_vbi.block<3, 3>(12, 6) = -Eigen::Matrix3d::Identity();
                jacob_vbi = sqrt_info_ * jacob_vbi;
            }

            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacob_posej(jacobians[2]);
                jacob_posej.setZero();
                // jacob Rbjw
                jacob_posej.block<3, 3>(0, 0) = invJr;
                // jacob tbjw
                jacob_posej.block<3, 3>(6, 3) = Rbiw.transpose();
                jacob_posej = sqrt_info_ * jacob_posej;
            }

            if (jacobians[3] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacob_vbj(jacobians[3]);
                jacob_vbj.setZero();
                // jacob vj
                jacob_vbj.block<3, 3>(3, 0) = Rbiw.transpose();

                // jacob bgj
                jacob_vbj.block<3, 3>(9, 3).setIdentity();
                // jacob baj
                jacob_vbj.block<3, 3>(12, 6).setIdentity();
                jacob_vbj = sqrt_info_ * jacob_vbj;
            }
        }

        return true;
    }

    inline Eigen::Matrix3d Jr(const Eigen::Vector3d &rvec) const {
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

    inline Eigen::Matrix3d NewInvJr(const Eigen::Vector3d rvec) const {
        double angle = rvec.norm();
        double angle2 = angle * angle;

        if (angle < 1.0e-05) {
            return Eigen::Matrix3d::Identity();
        }

        Eigen::Matrix3d r_hat;
        r_hat << 0.0, -rvec(2), rvec(1), rvec(2), 0.0, -rvec(0), -rvec(1), rvec(0), 0.0;
        Eigen::Matrix3d J_inv =
            Eigen::Matrix3d::Identity() + 0.5 * r_hat + r_hat * r_hat * (1.0 / angle2 - (1.0 + cos(angle)) / (2.0 * angle * sin(angle)));
        return J_inv;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (fabs(theta) < 1.0e-09) {
            if (theta < 0) {
                theta = -1.0e-09;
            } else {
                theta = 1.0e-09;
            }
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }

    inline Eigen::Vector3d RotationMatrix2Vector3d(const Eigen::Matrix3d R) const {
        Eigen::AngleAxisd rv(R);
        return rv.angle() * rv.axis();

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
            return 3.14159265357 * rvec / rvec.norm();
        }
        rvec /= rvec.norm();
        return rvec * theta;
    }

private:
    Eigen::Matrix<double, 15, 15> sqrt_info_;
    Eigen::Vector3d Gw_, tcb_;
    Eigen::Matrix3d Rcb_;
    Preintegrated imu_;
};

// Parameters [vi, vj, bg, ba, rgw, rvcb, scale]
class AligmentExtrinsicCost : public ceres::SizedCostFunction<9, 3, 3, 3, 3, 3, 3, 1> {
public:
    AligmentExtrinsicCost(Eigen::Matrix<double, 9, 9> sqrt_info, Eigen::Matrix3d Rciw, Eigen::Matrix3d Rcjw, Eigen::Vector3d vPi,
        Eigen::Vector3d vPj, Eigen::Vector3d Gw, Preintegrated imu) : sqrt_info_(sqrt_info), Rciw_(Rciw), Rcjw_(Rcjw),
        vPi_(vPi), vPj_(vPj), Gw_(Gw), imu_(imu) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> Vi(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> Vj(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> bg(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> ba(parameters[3]);
        Eigen::Map<const Eigen::Vector3d> rvgw(parameters[4]);
        Eigen::Map<const Eigen::Vector3d> rvcb(parameters[5]);
        const double &scale = parameters[6][0];
        Eigen::Matrix3d Rcb = RotationVector2Matrix3d(rvcb);

        Eigen::Vector3d dbg = bg - imu_.bg_;
        Eigen::Vector3d dba = ba - imu_.ba_;
        Eigen::Matrix3d Rji = imu_.Rji_ * RotationVector2Matrix3d(imu_.dRg_ * dbg);
        Eigen::Matrix3d hat_Rij = Rcb * Rcjw_.transpose() * Rciw_ * Rcb.transpose();
        Eigen::Matrix3d err_R = hat_Rij * Rji;
        Eigen::Vector3d err_rv = RotationMatrix2Vector3d(err_R);

        Eigen::Vector3d corrected_vij = imu_.Vij_ + imu_.dVa_ * dba + imu_.dVg_ * dbg;
        Eigen::Vector3d corrected_pij = imu_.Pij_ + imu_.dPa_ * dba + imu_.dPg_ * dbg;

        Eigen::Matrix3d Rgw = RotationVector2Matrix3d(rvgw);
        Eigen::Vector3d G = Rgw * Gw_;

        const double &dt = imu_.dT_;
        Eigen::Matrix3d Rbiw = Rciw_ * Rcb.transpose();
        Eigen::Vector3d dv = Rbiw.transpose() * (scale * Vj - scale * Vi + G * dt);
        Eigen::Vector3d dp = Rbiw.transpose() * (scale * vPj_ - scale * vPi_ - scale * Vi * dt + 0.5 * G * dt * dt);

        Eigen::Map<Eigen::VectorXd> residual(residuals, 9);
        residual.segment<3>(0) = err_rv;
        residual.segment<3>(3) = dv - corrected_vij;
        residual.segment<3>(6) = dp - corrected_pij;
        residual = sqrt_info_ * residual;
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_vi(jacobians[0]);
                jacob_vi.setZero();
                jacob_vi.block<3, 3>(3, 0) = -scale * Rbiw.transpose();
                jacob_vi.block<3, 3>(6, 0) = -scale * Rbiw.transpose() * dt;
                jacob_vi = sqrt_info_ * jacob_vi;
            }
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_vj(jacobians[1]);
                jacob_vj.setZero();
                jacob_vj.block<3, 3>(6, 0) = scale * Rbiw.transpose();
                jacob_vj = sqrt_info_ * jacob_vj;
            }
            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_bg(jacobians[2]);
                jacob_bg.setZero();
                jacob_bg.block<3, 3>(0, 0) = err_R * Jr(imu_.dRg_ * dbg) * imu_.dRg_;
                jacob_bg.block<3, 3>(3, 0) = -imu_.dVg_;
                jacob_bg.block<3, 3>(6, 0) = -imu_.dPg_;
                jacob_bg = sqrt_info_ * jacob_bg;
            }
            if (jacobians[3] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_ba(jacobians[3]);
                jacob_ba.setZero();
                jacob_ba.block<3, 3>(3, 0) = -imu_.dVa_;
                jacob_ba.block<3, 3>(6, 0) = -imu_.dPa_;
                jacob_ba = sqrt_info_ * jacob_ba;
            }
            if (jacobians[4] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_rgw(jacobians[4]);
                jacob_rgw.setZero();
                Eigen::Matrix3d rgw_Jr = Jr(rvgw);
                Eigen::Matrix3d hat_G;
                hat_G << 0.0, -Gw_(2), Gw_(1), Gw_(2), 0.0, -Gw_(0), -Gw_(1), Gw_(0), 0.0;
                jacob_rgw.block<3, 3>(3, 0) = -Rbiw.transpose() * Rgw * hat_G * rgw_Jr * dt;
                jacob_rgw.block<3, 3>(6, 0) = -0.5 * Rbiw.transpose() * Rgw * hat_G * rgw_Jr * dt * dt;
                jacob_rgw = sqrt_info_ * jacob_rgw;
            }
            if (jacobians[5] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_rcb(jacobians[5]);
                jacob_rcb.setZero();
                Eigen::Matrix3d rcb_Jr = Jr(rvcb);
                Eigen::Matrix3d Rcicj = Rcjw_.transpose() * Rciw_;
                jacob_rcb.block<3, 3>(0, 0) = Rji.transpose() * Rcb * (Rcicj.transpose() - Eigen::Matrix3d::Identity()) * rcb_Jr;
                Eigen::Matrix3d hat_dv;
                hat_dv << 0.0, -dv(2), dv(1), dv(2), 0.0, -dv(0), -dv(1), dv(0), 0.0;
                jacob_rcb.block<3, 3>(3, 0) = -hat_dv * Rcb * rcb_Jr;
                Eigen::Matrix3d hat_dp;
                hat_dp << 0.0, -dp(2), dp(1), dp(2), 0.0, -dp(0), -dp(1), dp(0), 0.0;
                jacob_rcb.block<3, 3>(6, 0) = -hat_dp * Rcb * rcb_Jr;
                jacob_rcb = sqrt_info_ * jacob_rcb;
            }
            if (jacobians[6] != NULL) {
                Eigen::Map<Eigen::VectorXd> jacob_s(jacobians[6], 9);
                jacob_s.setZero();
                jacob_s.segment<3>(3) = Rbiw.transpose() * (Vj - Vi);
                jacob_s.segment<3>(6) = Rbiw.transpose() * (vPj_ - vPi_ - Vi * dt);
                jacob_s = sqrt_info_ * jacob_s;
            }
        }


        return true;
    }

    inline Eigen::Matrix3d Jr(const Eigen::Vector3d &rvec) const {
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

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (fabs(theta) < 1.0e-06) {
            if (theta < 0) {
                theta = -1.0e-06;
            } else {
                theta = 1.0e-06;
            }
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }

    inline Eigen::Vector3d RotationMatrix2Vector3d(const Eigen::Matrix3d R) const {
        double theta = acos((R.trace() - 1.0) * 0.5);

        Eigen::Matrix3d Right = (R - R.transpose()) * 0.5;
        Eigen::Vector3d rvec;
        rvec[0] = (Right(2, 1) - Right(1, 2)) * 0.5;
        rvec[1] = (Right(0, 2) - Right(2, 0)) * 0.5;
        rvec[2] = (Right(1, 0) - Right(0, 1)) * 0.5;
        double rv_norm = rvec.norm();
        if (rv_norm < 1.0e-07) {
            return rvec;
        }

        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        if (fabs(sin_theta) < 1.0e-06 || fabs(cos_theta) > 1 - 1.0e-06) {
            return rvec;
        }
        rvec /= sin_theta;

        return rvec * theta;
    }

private:
    Eigen::Matrix<double, 9, 9> sqrt_info_;
    Eigen::Matrix3d Rciw_, Rcjw_;
    Eigen::Vector3d vPi_, vPj_, Gw_;
    Preintegrated imu_;

};

class AligmentCost : public ceres::SizedCostFunction<9, 3, 3, 3, 3, 3, 1> {
public:
    AligmentCost(Eigen::Matrix<double, 9, 9> sqrt_info, Eigen::Matrix3d Rcb, Eigen::Matrix3d Rbiw, Eigen::Matrix3d Rbjw,
        Eigen::Vector3d vPi, Eigen::Vector3d vPj, Eigen::Vector3d Gw, Eigen::Vector3d tcb, Preintegrated imu)
        : sqrt_info_(sqrt_info), Rcb_(Rcb), Rbiw_(Rbiw), Rbjw_(Rbjw), vPi_(vPi), vPj_(vPj), Gw_(Gw), tcb_(tcb), imu_(imu) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> Vi(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> Vj(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> bg(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> ba(parameters[3]);
        Eigen::Map<const Eigen::Vector3d> rgw(parameters[4]);
        const double &scale = parameters[5][0];

        Eigen::Vector3d dbg = bg - imu_.bg_;
        Eigen::Vector3d dba = ba - imu_.ba_;
        Eigen::Matrix3d Rji = imu_.Rji_ * RotationVector2Matrix3d(imu_.dRg_ * dbg);
        Eigen::Matrix3d err_R = Rbjw_.transpose() * Rbiw_ * Rji;
        Eigen::Vector3d err_rv = RotationMatrix2Vector3d(err_R);

        Eigen::Matrix3d Rgw = RotationVector2Matrix3d(rgw);
        Eigen::Vector3d G = Rgw * Gw_;

        Eigen::Vector3d corrected_vij = imu_.Vij_ + imu_.dVa_ * dba + imu_.dVg_ * dbg;
        Eigen::Vector3d corrected_pij = imu_.Pij_ + imu_.dPa_ * dba + imu_.dPg_ * dbg;

        const double &dt = imu_.dT_;
        Eigen::Vector3d dv = Rbiw_.transpose() * (scale * Vj - scale * Vi + G * dt);
        Eigen::Vector3d dp = Rbiw_.transpose() * (scale * vPj_ - scale * vPi_ - scale * Vi * dt + 0.5 * G * dt * dt);

        Eigen::Map<Eigen::VectorXd> residual(residuals, 9);
        residual.segment<3>(0) = err_rv;
        residual.segment<3>(3) = dv - corrected_vij;
        residual.segment<3>(6) = dp - corrected_pij;
        residual = sqrt_info_ * residual;
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_vi(jacobians[0]);
                jacob_vi.setZero();
                jacob_vi.block<3, 3>(3, 0) = -scale * Rbiw_.transpose();
                jacob_vi.block<3, 3>(6, 0) = -scale * Rbiw_.transpose() * dt;
                jacob_vi = sqrt_info_ * jacob_vi;
            }
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_vj(jacobians[1]);
                jacob_vj.setZero();
                jacob_vj.block<3, 3>(3, 0) = scale * Rbiw_.transpose();
                jacob_vj = sqrt_info_ * jacob_vj;
            }
            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_bg(jacobians[2]);
                jacob_bg.setZero();
                jacob_bg.block<3, 3>(0, 0) = err_R * Jr(imu_.dRg_ * dbg) * imu_.dRg_;
                jacob_bg.block<3, 3>(3, 0) = -imu_.dVg_;
                jacob_bg.block<3, 3>(6, 0) = -imu_.dPg_;
                jacob_bg = sqrt_info_ * jacob_bg;
            }
            if (jacobians[3] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_ba(jacobians[3]);
                jacob_ba.setZero();
                jacob_ba.block<3, 3>(3, 0) = -imu_.dVa_;
                jacob_ba.block<3, 3>(6, 0) = -imu_.dPa_;
                jacob_ba = sqrt_info_ * jacob_ba;
            }
            if (jacobians[4] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_rgw(jacobians[4]);
                jacob_rgw.setZero();
                Eigen::Matrix3d rgw_Jr = Jr(rgw);
                Eigen::Matrix3d hat_G;
                hat_G << 0.0, -Gw_(2), Gw_(1), Gw_(2), 0.0, -Gw_(0), -Gw_(1), Gw_(0), 0.0;
                jacob_rgw.block<3, 3>(3, 0) = -Rbiw_.transpose() * Rgw * hat_G * rgw_Jr * dt;
                jacob_rgw.block<3, 3>(6, 0) = -0.5 * Rbiw_.transpose() * Rgw * hat_G * rgw_Jr * dt * dt;
                jacob_rgw = sqrt_info_ * jacob_rgw;
            }
            if (jacobians[5] != NULL) {
                Eigen::Map<Eigen::VectorXd> jacob_s(jacobians[5], 9);
                jacob_s.setZero();
                jacob_s.segment<3>(3) = Rbiw_.transpose() * (Vj - Vi);
                jacob_s.segment<3>(6) = Rbiw_.transpose() * (vPj_ - vPi_ - Vi * dt);
                jacob_s = sqrt_info_ * jacob_s;
            }
        }
        return true;
    }
    inline Eigen::Matrix3d Jr(const Eigen::Vector3d &rvec) const {
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

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (fabs(theta) < 1.0e-06) {
            if (theta < 0) {
                theta = -1.0e-06;
            } else {
                theta = 1.0e-06;
            }
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }

    inline Eigen::Vector3d RotationMatrix2Vector3d(const Eigen::Matrix3d R) const {
        double theta = acos((R.trace() - 1.0) * 0.5);

        Eigen::Matrix3d Right = (R - R.transpose()) * 0.5;
        Eigen::Vector3d rvec;
        rvec[0] = (Right(2, 1) - Right(1, 2)) * 0.5;
        rvec[1] = (Right(0, 2) - Right(2, 0)) * 0.5;
        rvec[2] = (Right(1, 0) - Right(0, 1)) * 0.5;
        double rv_norm = rvec.norm();
        if (rv_norm < 1.0e-07) {
            return rvec;
        }

        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        if (fabs(sin_theta) < 1.0e-06 || fabs(cos_theta) > 1 - 1.0e-06) {
            return rvec;
        }
        rvec /= sin_theta;

        return rvec * theta;
    }
private:
    Eigen::Matrix<double, 9, 9> sqrt_info_;
    Eigen::Matrix3d Rcb_, Rbiw_, Rbjw_;
    Eigen::Vector3d vPi_, vPj_, Gw_, tcb_;
    Preintegrated imu_;
};

// residual: [dtheta, dv, dp]
// parameters: [vi, vj, ba, gw_dir, scale]
class ImuInitCost : public ceres::SizedCostFunction<9, 3, 3, 3, 3, 1> {
public:
    ImuInitCost(Eigen::Matrix<double, 9, 9> sqrt_info,
                Eigen::Vector3d vPi,
                Eigen::Vector3d vPj,
                Eigen::Vector3d Gw,
                Eigen::Vector3d tcb,
                Eigen::Vector3d bg,
                Eigen::Matrix3d Rcb,
                Eigen::Matrix3d Rbiw,
                Eigen::Matrix3d Rbjw,
                Preintegrated imu)
        : sqrt_info_(sqrt_info),
          visual_pi_(vPi),
          visual_pj_(vPj),
          Gw_(Gw),
          tcb_(tcb),
          bg_(bg),
          Rcb_(Rcb),
          Rbiw_(Rbiw),
          Rbjw_(Rbjw),
          imu_(imu) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> Vi(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> Vj(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> ba(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> rvgw(parameters[3]);
        double scale = parameters[4][0];
        Eigen::Vector3d dbg = bg_ - imu_.bg_;
        Eigen::Vector3d dba = ba - imu_.ba_;
        Eigen::Matrix3d Rji = imu_.Rji_ * RotationVector2Matrix3d(imu_.dRg_ * dbg);
        Eigen::Matrix3d err_R = Rji.transpose() * Rbiw_.transpose() * Rbjw_;
        Eigen::Vector3d err_rv = RotationMatrix2Vector3d(err_R);

        Eigen::Matrix3d Rgw = RotationVector2Matrix3d(rvgw);
        Eigen::Vector3d G = Rgw * Gw_;

        Eigen::Vector3d corrected_Vij = imu_.Vij_ + imu_.dVa_ * dba + imu_.dVg_ * dbg;
        Eigen::Vector3d corrected_Pij = imu_.Pij_ + imu_.dPa_ * dba + imu_.dPg_ * dbg;

        const double &dt = imu_.dT_;
        Eigen::Vector3d dv = Rbiw_.transpose() * (scale * Vj - scale * Vi + G * dt);
        Eigen::Vector3d dp = Rbiw_.transpose() * (scale * visual_pj_ - scale * visual_pi_ - scale * Vi * dt + 0.5 * G * dt * dt);

        Eigen::Map<Eigen::Matrix<double, 9, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = err_rv;
        residual.block<3, 1>(3, 0) = dv - corrected_Vij;
        residual.block<3, 1>(6, 0) = dp - corrected_Pij;
        residual = sqrt_info_ * residual;

        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_vi(jacobians[0]);
                jacob_vi.setZero();
                jacob_vi.block<3, 3>(3, 0) = -scale * Rbiw_.transpose();
                jacob_vi.block<3, 3>(6, 0) = -scale * Rbiw_.transpose() * dt;
                jacob_vi = sqrt_info_ * jacob_vi;
            }
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_vj(jacobians[1]);
                jacob_vj.setZero();
                jacob_vj.block<3, 3>(3, 0) = scale * Rbiw_.transpose();
                jacob_vj = sqrt_info_ * jacob_vj;
            }
            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_ba(jacobians[2]);
                jacob_ba.setZero();
                jacob_ba.block<3, 3>(3, 0) = -imu_.dVa_;
                jacob_ba.block<3, 3>(6, 0) = -imu_.dPa_;
                jacob_ba = sqrt_info_ * jacob_ba;
            }
            if (jacobians[3] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacob_rgw(jacobians[3]);
                jacob_rgw.setZero();
                Eigen::Matrix3d rgw_Jr = Jr(rvgw);
                Eigen::Matrix3d hat_G;
                hat_G << 0.0, -Gw_(2), Gw_(1), Gw_(2), 0.0, -Gw_(0), -Gw_(1), Gw_(0), 0.0;
                jacob_rgw.block<3, 3>(3, 0) = -Rbiw_.transpose() * Rgw * hat_G * rgw_Jr * dt;
                jacob_rgw.block<3, 3>(6, 0) = -0.5 * Rbiw_.transpose() * Rgw * hat_G * rgw_Jr * dt * dt;
                jacob_rgw = sqrt_info_ * jacob_rgw;
            }
            if (jacobians[4] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 9, 1>> jacob_s(jacobians[4]);
                jacob_s.setZero();
                jacob_s.block<3, 1>(3, 0) = Rbiw_.transpose() * (Vj - Vi);
                jacob_s.block<3, 1>(6, 0) = Rbiw_.transpose() * (visual_pj_ - visual_pi_ - Vi * dt);
                jacob_s = sqrt_info_ * jacob_s;
            }
        }
        return true;
    }

    inline Eigen::Matrix3d Jr(const Eigen::Vector3d &rvec) const {
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

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (fabs(theta) < 1.0e-06) {
            if (theta < 0) {
                theta = -1.0e-06;
            } else {
                theta = 1.0e-06;
            }
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }

    inline Eigen::Vector3d RotationMatrix2Vector3d(const Eigen::Matrix3d R) const {
        double theta = acos((R.trace() - 1.0) * 0.5);

        Eigen::Matrix3d Right = (R - R.transpose()) * 0.5;
        Eigen::Vector3d rvec;
        rvec[0] = (Right(2, 1) - Right(1, 2)) * 0.5;
        rvec[1] = (Right(0, 2) - Right(2, 0)) * 0.5;
        rvec[2] = (Right(1, 0) - Right(0, 1)) * 0.5;
        double rv_norm = rvec.norm();
        if (rv_norm < 1.0e-07) {
            return rvec;
        }

        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        if (fabs(sin_theta) < 1.0e-06 || fabs(cos_theta) > 1 - 1.0e-06) {
            return rvec;
        }
        rvec /= sin_theta;

        return rvec * theta;
    }

private:
    Eigen::Matrix<double, 9, 9> sqrt_info_;
    Eigen::Vector3d visual_pi_, visual_pj_, Gw_, tcb_, bg_;
    Eigen::Matrix3d Rcb_, Rbiw_, Rbjw_;
    Preintegrated imu_;
};

// imu_Rji = imu_Rji * exp(jacob * delta_bias)
// error = log(imu_Rji.transpose() * Rji)
class GyroBiasErr : public ceres::SizedCostFunction<3, 3> {
public:
    GyroBiasErr(Eigen::Matrix3d sqrt_info, Eigen::Matrix3d delta_Rji, Eigen::Matrix3d visual_Rji, Eigen::Matrix3d bias_jacob)
        : imu_Rji_(delta_Rji), visual_Rji_(visual_Rji), bias_jacob_(bias_jacob), sqrt_info_(sqrt_info) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        const double *bias = parameters[0];
        Eigen::Vector3d gyro_bias(bias[0], bias[1], bias[2]);
        Eigen::Matrix3d new_imu_Rji = imu_Rji_ * RotationVector2Matrix3d(bias_jacob_ * gyro_bias);
        Eigen::Matrix3d err_R = visual_Rji_.transpose() * new_imu_Rji;
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        //Eigen::Vector3d rvec = RotationMatrix2Vector3d(err_R);
        residual = sqrt_info_ * RotationMatrix2Vector3d(err_R);

        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacob(jacobians[0]);
                jacob = sqrt_info_ * err_R * Jr(bias_jacob_ * gyro_bias) * bias_jacob_;
            }
        }
        return true;
    }

    inline Eigen::Matrix3d Jr(const Eigen::Vector3d &rvec) const {
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

    inline Eigen::Matrix3d NewInvJr(const Eigen::Vector3d rvec) const {
        double angle = rvec.norm();
        double angle2 = angle * angle;

        if (angle < 1.0e-05) {
            return Eigen::Matrix3d::Identity();
        }

        Eigen::Matrix3d r_hat;
        r_hat << 0.0, -rvec(2), rvec(1), rvec(2), 0.0, -rvec(0), -rvec(1), rvec(0), 0.0;
        Eigen::Matrix3d J_inv =
            Eigen::Matrix3d::Identity() + 0.5 * r_hat + r_hat * r_hat * (1.0 / angle2 - (1.0 + cos(angle)) / (2.0 * angle * sin(angle)));
        return J_inv;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d &rvec) const {
        double theta = rvec.norm();
        if (fabs(theta) < 1.0e-06) {
            if (theta < 0) {
                theta = -1.0e-06;
            } else {
                theta = 1.0e-06;
            }
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }

    inline Eigen::Vector3d RotationMatrix2Vector3d(const Eigen::Matrix3d &R) const {
        double theta = acos((R.trace() - 1.0) * 0.5);

        Eigen::Matrix3d Right = (R - R.transpose()) * 0.5;
        Eigen::Vector3d rvec;
        rvec[0] = (Right(2, 1) - Right(1, 2)) * 0.5;
        rvec[1] = (Right(0, 2) - Right(2, 0)) * 0.5;
        rvec[2] = (Right(1, 0) - Right(0, 1)) * 0.5;

        double norm = rvec.norm();
        if (norm < 1.0e-06) {
            norm = 1.0e-06;
        }
        rvec /= norm;

        return rvec * theta;
    }

private:
    Eigen::Matrix3d imu_Rji_, visual_Rji_, bias_jacob_, sqrt_info_;
};

class PriorCost : public ceres::SizedCostFunction<3, 9> {
public:
    PriorCost(Eigen::Matrix3d sqrt_info, Eigen::Vector3d ba) : ba_(ba), sqrt_info_(sqrt_info) {}
    virtual bool Evaluate(double const *const *parameters, double* residuals, double** jacobians) const {
        Eigen::Map<const Eigen::Vector3d> baj(parameters[0] + 6);
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = baj - ba_;
        residual = sqrt_info_ * residual;
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> jacob(jacobians[0]);
                jacob.setZero();
                jacob.block<3, 3>(0, 6).setIdentity();
                jacob = sqrt_info_ * jacob;
            }
        }
        return true;
    }

private:
    Eigen::Vector3d ba_;
    Eigen::Matrix3d sqrt_info_;
};

class AttitudeError : public ceres::SizedCostFunction<3, 6, 6> {
public:
    AttitudeError(Eigen::Matrix3d sqrt_info, Preintegrated imu) : sqrt_info_(sqrt_info), imu_(imu) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rvi(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> rvj(parameters[1]);

        Eigen::Matrix3d Rbiw = RotationVector2Matrix3d(rvi);
        Eigen::Matrix3d Rbjw = RotationVector2Matrix3d(rvj);

        Eigen::Matrix3d Rji = Rbiw.transpose() * Rbjw;
        Eigen::Matrix3d dR = imu_.Rji_.transpose() * Rji;
        Eigen::Vector3d err_rv = RotationMatrix2Vector3d(dR);
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = sqrt_info_ * err_rv;

        Eigen::Matrix3d invJr = NewInvJr(err_rv);
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobi(jacobians[0]);
                jacobi.setZero();
                jacobi.block<3, 3>(0, 0) = -invJr * Rbjw.transpose() * Rbiw;
                jacobi = sqrt_info_ * jacobi;
            }
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobj(jacobians[1]);
                jacobj.setZero();
                jacobj.block<3, 3>(0, 0) = invJr;
                jacobj = sqrt_info_ * jacobj;
            }
        }
        
        return true;
    }


    inline Eigen::Matrix3d NewInvJr(const Eigen::Vector3d rvec) const {
        double angle = rvec.norm();
        double angle2 = angle * angle;

        if (angle < 1.0e-05) {
            return Eigen::Matrix3d::Identity();
        }

        Eigen::Matrix3d r_hat;
        r_hat << 0.0, -rvec(2), rvec(1), rvec(2), 0.0, -rvec(0), -rvec(1), rvec(0), 0.0;
        Eigen::Matrix3d J_inv =
            Eigen::Matrix3d::Identity() + 0.5 * r_hat + r_hat * r_hat * (1.0 / angle2 - (1.0 + cos(angle)) / (2.0 * angle * sin(angle)));
        return J_inv;
    }

    inline Eigen::Vector3d RotationMatrix2Vector3d(const Eigen::Matrix3d &R) const {
        double theta = acos((R.trace() - 1.0) * 0.5);

        Eigen::Matrix3d Right = (R - R.transpose()) * 0.5;
        Eigen::Vector3d rvec;
        rvec[0] = (Right(2, 1) - Right(1, 2)) * 0.5;
        rvec[1] = (Right(0, 2) - Right(2, 0)) * 0.5;
        rvec[2] = (Right(1, 0) - Right(0, 1)) * 0.5;

        double norm = rvec.norm();
        if (norm < 1.0e-07) {
            return rvec;
        }
        rvec /= norm;

        return rvec * theta;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d &rvec) const {
        double theta = rvec.norm();
        if (fabs(theta) < 1.0e-06) {
            if (theta < 0) {
                theta = -1.0e-06;
            } else {
                theta = 1.0e-06;
            }
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }
private:
    Eigen::Matrix3d sqrt_info_;
    Preintegrated imu_;
};
// parameters: [[Ri, ti], [vi, bgi, bai]], [[Rj, tj], [vj, bgj, baj]]
class Imu9DError : public ceres::SizedCostFunction<15, 6, 3, 6, 6, 3, 6> {
public:
    Imu9DError(Eigen::Matrix<double, 15, 15> sqrt_info, Eigen::Vector3d Gw, Preintegrated imu)
        : sqrt_info_(sqrt_info), Gw_(Gw), imu_(imu) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::VectorXd> posei(parameters[0], 6);
        Eigen::Map<const Eigen::Vector3d> vi(parameters[1]);
        Eigen::Map<const Eigen::VectorXd> bi(parameters[2], 6);
        Eigen::Map<const Eigen::VectorXd> posej(parameters[3], 6);
        Eigen::Map<const Eigen::Vector3d> vj(parameters[4]);
        Eigen::Map<const Eigen::VectorXd> bj(parameters[5], 6);

        Eigen::Matrix3d Rbiw = RotationVector2Matrix3d(posei.topRows(3));
        Eigen::Matrix3d Rbjw = RotationVector2Matrix3d(posej.topRows(3));

        Eigen::Vector3d dbg = bi.segment<3>(0) - imu_.bg_;
        Eigen::Vector3d dba = bi.segment<3>(3) - imu_.ba_;

        Eigen::Matrix3d corrected_Rji = imu_.Rji_ * RotationVector2Matrix3d(imu_.dRg_ * dbg);
        Eigen::Matrix3d err_R = corrected_Rji.transpose() * Rbiw.transpose() * Rbjw;
        Eigen::Vector3d err_rv = RotationMatrix2Vector3d(err_R);

        Eigen::Vector3d corrected_Pij = imu_.Pij_ + imu_.dPa_ * dba + imu_.dPg_ * dbg;
        Eigen::Vector3d corrected_Vij = imu_.Vij_ + imu_.dVa_ * dba + imu_.dVg_ * dbg;

        const double &dt = imu_.dT_;
        Eigen::Vector3d dv = Rbiw.transpose() * (vj - vi + Gw_ * dt);
        Eigen::Vector3d dp = Rbiw.transpose() * (posej.segment<3>(3) - posei.segment<3>(3) - vi * dt + 0.5 * Gw_ * dt * dt);

        Eigen::Map<Eigen::VectorXd> residual(residuals, 15);
        residual.segment<3>(0) = err_rv;
        residual.segment<3>(3) = dv - corrected_Vij;
        residual.segment<3>(6) = dp - corrected_Pij;
        residual.segment<3>(9) = bj.segment<3>(0) - bi.segment<3>(0);
        residual.segment<3>(12) = bj.segment<3>(3) - bi.segment<3>(3);
        residual = sqrt_info_ * residual;

        Eigen::Matrix3d invJr = NewInvJr(err_rv);
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacob_posei(jacobians[0]);
                jacob_posei.setZero();
                // jacob Rbiw
                jacob_posei.block<3, 3>(0, 0) = -invJr * Rbjw.transpose() * Rbiw;
                jacob_posei.block<3, 3>(3, 0) << 0.0, -dv(2), dv(1), dv(2), 0.0, -dv(0), -dv(1), dv(0), 0.0;
                jacob_posei.block<3, 3>(6, 0) << 0.0, -dp(2), dp(1), dp(2), 0.0, -dp(0), -dp(1), dp(0), 0.0;

                // jacob tbiw
                jacob_posei.block<3, 3>(6, 3) = -Rbiw.transpose();
                jacob_posei = sqrt_info_ * jacob_posei;
            }
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacob_vi(jacobians[1]);
                jacob_vi.setZero();
                jacob_vi.block<3, 3>(3, 0) = -Rbiw.transpose();
                jacob_vi.block<3, 3>(6, 0) = -Rbiw.transpose() * dt;
                jacob_vi = sqrt_info_ * jacob_vi;
            }
            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacob_bi(jacobians[2]);
                jacob_bi.setZero();
                // jacob bgi
                jacob_bi.block<3, 3>(0, 0) = -invJr * err_R.transpose() * Jr(imu_.dRg_ * dbg) * imu_.dRg_;
                jacob_bi.block<3, 3>(3, 0) = -imu_.dVg_;
                jacob_bi.block<3, 3>(6, 0) = -imu_.dPg_;
                jacob_bi.block<3, 3>(9, 0) = -Eigen::Matrix3d::Identity();
                // jacob bai
                jacob_bi.block<3, 3>(3, 3) = -imu_.dVa_;
                jacob_bi.block<3, 3>(6, 3) = -imu_.dPa_;
                jacob_bi.block<3, 3>(12, 3) = -Eigen::Matrix3d::Identity();
                jacob_bi = sqrt_info_ * jacob_bi;
            }
            if (jacobians[3] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacob_posej(jacobians[3]);
                jacob_posej.setZero();
                // jacob Rbjw
                jacob_posej.block<3, 3>(0, 0) = invJr;
                // jacob tbjw
                jacob_posej.block<3, 3>(6, 3) = Rbiw.transpose();
                jacob_posej = sqrt_info_ * jacob_posej;
            }
            if (jacobians[4] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacob_vj(jacobians[4]);
                jacob_vj.setZero();
                // jacob vj
                jacob_vj.block<3, 3>(3, 0) = Rbiw.transpose();
                jacob_vj = sqrt_info_ * jacob_vj;
            }
            if (jacobians[5] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacob_bj(jacobians[5]);
                jacob_bj.setZero();
                // jacob bgj
                jacob_bj.block<3, 3>(9, 0).setIdentity();
                // jacob baj
                jacob_bj.block<3, 3>(12, 3).setIdentity();
                jacob_bj = sqrt_info_ * jacob_bj;
            }
        }
        return true;
    }

    inline Eigen::Matrix3d Jr(const Eigen::Vector3d &rvec) const {
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

    inline Eigen::Matrix3d NewInvJr(const Eigen::Vector3d rvec) const {
        double angle = rvec.norm();
        double angle2 = angle * angle;

        if (angle < 1.0e-05) {
            return Eigen::Matrix3d::Identity();
        }

        Eigen::Matrix3d r_hat;
        r_hat << 0.0, -rvec(2), rvec(1), rvec(2), 0.0, -rvec(0), -rvec(1), rvec(0), 0.0;
        Eigen::Matrix3d J_inv =
            Eigen::Matrix3d::Identity() + 0.5 * r_hat + r_hat * r_hat * (1.0 / angle2 - (1.0 + cos(angle)) / (2.0 * angle * sin(angle)));
        return J_inv;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (fabs(theta) < 1.0e-06) {
            if (theta < 0) {
                theta = -1.0e-06;
            } else {
                theta = 1.0e-06;
            }
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }

    inline Eigen::Vector3d RotationMatrix2Vector3d(const Eigen::Matrix3d R) const {
        double cos_theta = (R.trace() - 1.0) * 0.5;

        double theta = acos(cos_theta);
        Eigen::Matrix3d Right = (R - R.transpose()) * 0.5;
        Eigen::Vector3d rvec;
        rvec[0] = (Right(2, 1) - Right(1, 2)) * 0.5;
        rvec[1] = (Right(0, 2) - Right(2, 0)) * 0.5;
        rvec[2] = (Right(1, 0) - Right(0, 1)) * 0.5;
        if (fabs(cos_theta) > (1.0 - 1.0e-06)) {
            return rvec;
        }
        double sin_theta = sin(theta);
        if (fabs(sin_theta) < 1.0e-06 || fabs(cos_theta) > 1 - 1.0e-06) {
            return rvec;
        }
        rvec /= sin_theta;

        return rvec * theta;
    }

private:
    Eigen::Matrix<double, 15, 15> sqrt_info_;
    Eigen::Vector3d Gw_;
    Preintegrated imu_;
};

class BiasError : public ceres::SizedCostFunction<6, 6, 6> {
public:
    BiasError(Eigen::Matrix<double, 6, 6> sqrt_info) : sqrt_info_(sqrt_info) {}
    virtual bool Evaluate(double const *const *parameters, double* residuals, double **jacobians) const {
        Eigen::Map<const Eigen::VectorXd> vbi(parameters[0], 6);
        Eigen::Map<const Eigen::VectorXd> vbj(parameters[1], 6);

        Eigen::Map<Eigen::VectorXd> residual(residuals, 6);
        residual.segment<3>(0) = vbj.segment<3>(0) - vbi.segment<3>(0);
        residual.segment<3>(3) = vbj.segment<3>(3) - vbi.segment<3>(3);
        residual = sqrt_info_ * residual;
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jacob_vbi(jacobians[0]);
                jacob_vbi.setZero();
                // jacob bgi
                jacob_vbi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
                // jacob bai
                jacob_vbi.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
                jacob_vbi = sqrt_info_ * jacob_vbi;
            }
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jacob_vbj(jacobians[1]);
                jacob_vbj.setZero();
                // jacob bgj
                jacob_vbj.block<3, 3>(0, 0).setIdentity();
                // jacob baj
                jacob_vbj.block<3, 3>(0, 3).setIdentity();
                jacob_vbj = sqrt_info_ * jacob_vbj;
            }
        }
        return true;
    }
private:
    Eigen::Matrix<double, 6, 6> sqrt_info_;
};

}  // namespace inslam
#endif
