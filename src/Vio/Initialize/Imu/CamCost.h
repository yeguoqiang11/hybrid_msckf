#ifndef INERTIALCOST_H
#define INERTIALCOST_H
#include <assert.h>
#include <ceres/sized_cost_function.h>
#include <Eigen/src/Core/Matrix.h>
#include <ceres/ceres.h>
#include "Vio/Initialize/Imu/Imu.h"

namespace inslam {
// parameter: [R, t]
// exp(theta0^) * exp(delta^) = exp((theta0 + invJr(theta0) * delta)^)
class PerturbationPoseParameterization : public ceres::LocalParameterization {
    virtual bool Plus(double const *x, double const *delta, double *x_plus_delta) const {
        Eigen::Map<const Eigen::Vector3d> theta(x);
        Eigen::Map<const Eigen::Vector3d> delta_theta(delta);

        Eigen::Map<Eigen::Vector3d> result(x_plus_delta);
        result = theta + NewInvJr(theta) * delta_theta;  //RotationMatrix2Vector3d(rotation);
        x_plus_delta[3] = x[3] + delta[3];
        x_plus_delta[4] = x[4] + delta[4];
        x_plus_delta[5] = x[5] + delta[5];
        return true;
    }
    virtual bool ComputeJacobian(double const *x, double *jacobian) const {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jacob(jacobian);
        jacob.setIdentity();
        return true;
    }

    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }

    inline Eigen::Matrix3d InvJr(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (theta < 1.0e-07) {
            return Eigen::Matrix3d::Identity();
        }
        Eigen::Vector3d a = rvec / theta;
        double cos_theta = cos(theta / 2);
        double sin_theta = sin(theta / 2);
        double tmp = 0.5 * theta * cos_theta / sin_theta;
        Eigen::Matrix3d a_hat;
        a_hat << 0.0, -a(2), a(1), a(2), 0.0, -a(0), -a(1), a(0), 0.0;
        return tmp * Eigen::Matrix3d::Identity() + (1.0 - tmp) * a * a.transpose() + a_hat * theta / 2.0;
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
        if (theta < 1.0e-08) {
            theta = 1.0e-08;
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

        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        if (fabs(sin_theta) < 1.0e-08 || fabs(cos_theta) > 1) {
            return rvec;
        }
        rvec /= sin_theta;

        return rvec * theta;
    }
};
// parameters: [Rbiw, tbiw] [point_w] [Rcb, tcb];
class ProjectionExtrinsicCost : public ceres::SizedCostFunction<3, 6, 3, 6> {
public:
    ProjectionExtrinsicCost(Eigen::Matrix3d sqrt_info, Eigen::Vector3d obs)
        : sqrt_info_(sqrt_info), obs_(obs) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rvbw(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tbw(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> ptw(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> rvcb(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> tcb(parameters[2] + 3);

        Eigen::Matrix3d Rbw = RotationVector2Matrix3d(rvbw);
        Eigen::Matrix3d Rcb = RotationVector2Matrix3d(rvcb);
        Eigen::Matrix3d Rwc = (Rbw * Rcb).transpose();
        Eigen::Vector3d tcw = (tbw + Rbw * tcb);
        Eigen::Vector3d twc = -Rwc * tcw;
        Eigen::Vector3d ptc = Rwc * ptw  + twc;
        double pt_len = ptc.norm();
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = sqrt_info_ * ((ptc / pt_len) - obs_);

        double len3 = pt_len * pt_len * pt_len;
        double x = ptc(0), y = ptc(1), z = ptc(2);
        double len_inv = 1.0 / pt_len;
        Eigen::Matrix3d jacobp;
        jacobp << len_inv - x * x / len3, -x * y / len3, -x * z / len3, -x * y / len3, len_inv - y * y / len3, -y * z / len3, -x * z / len3,
            -y * z / len3, len_inv - z * z / len3;
        jacobp = sqrt_info_ * jacobp;
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacob_pose(jacobians[0]);
                jacob_pose.setZero();
                Eigen::Matrix3d ptc_hat;
                ptc_hat << 0.0, -ptc(2), ptc(1), ptc(2), 0.0, -ptc(0), -ptc(1), ptc(0), 0.0;
                Eigen::Matrix3d hat1;
                hat1 << 0.0, -tcb(2), tcb(1), tcb(2), 0.0, -tcb(0), -tcb(1), tcb(0), 0.0;
                jacob_pose.block<3, 3>(0, 0) = ptc_hat * Rcb.transpose() + Rcb.transpose() * hat1;
                jacob_pose.block<3, 3>(0, 3) = -Rwc;
                jacob_pose = jacobp * jacob_pose;
            }
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacob_pt(jacobians[1]);
                jacob_pt = jacobp * Rwc;
            }
            if (jacobians[2] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacob_ex(jacobians[2]);
                jacob_ex.setZero();
                Eigen::Matrix3d ptc_hat;
                ptc_hat << 0.0, -ptc(2), ptc(1), ptc(2), 0.0, -ptc(0), -ptc(1), ptc(0), 0.0;
                jacob_ex.block<3, 3>(0, 0) = ptc_hat;
                jacob_ex.block<3, 3>(0, 3) = -Rcb.transpose();
                jacob_ex = jacobp * jacob_ex;
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
        if (theta < 1.0e-08) {
            theta = 1.0e-08;
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }
private:
    Eigen::Matrix3d sqrt_info_;
    Eigen::Vector3d obs_;
};
// parameteres: [Rbiw, tbiw] [point_w]
// residual: [camera observation]
class ProjectionError : public ceres::SizedCostFunction<3, 6, 3> {
public:
    ProjectionError(Eigen::Matrix3d sqrt_info, Eigen::Vector3d obs, Eigen::Vector3d tcb, Eigen::Matrix3d Rcb)
        : sqrt_info_(sqrt_info), obs_(obs), tcb_(tcb), Rcb_(Rcb) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rvec(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tbw(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> pt3d(parameters[1]);

        Eigen::Matrix3d Rbw = RotationVector2Matrix3d(rvec);
        Eigen::Matrix3d Rwc = (Rbw * Rcb_).transpose();
        Eigen::Vector3d twc = -Rwc * (tbw + Rbw * tcb_);
        Eigen::Vector3d ptc = Rwc * pt3d + twc;
        double pt_len = ptc.norm();
        Eigen::Map<Eigen::Vector3d> residual(residuals);

        residual = sqrt_info_ * ((ptc / pt_len) - obs_);

        double len3 = pt_len * pt_len * pt_len;
        double x = ptc(0), y = ptc(1), z = ptc(2);
        double len_inv = 1.0 / pt_len;
        Eigen::Matrix3d jacobp;
        jacobp << len_inv - x * x / len3, -x * y / len3, -x * z / len3, -x * y / len3, len_inv - y * y / len3, -y * z / len3, -x * z / len3,
            -y * z / len3, len_inv - z * z / len3;
        jacobp = sqrt_info_ * jacobp;
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacob_pose(jacobians[0]);
                Eigen::Vector3d tmp = Rwc * pt3d - Rwc * tbw;
                Eigen::Matrix3d hat0;
                hat0 << 0.0, -tmp(2), tmp(1), tmp(2), 0.0, -tmp(0), -tmp(1), tmp(0), 0.0;
                Eigen::Matrix3d hat1;
                hat1 << 0.0, -tcb_(2), tcb_(1), tcb_(2), 0.0, -tcb_(0), -tcb_(1), tcb_(0), 0.0;
                jacob_pose.block<3, 3>(0, 0) = hat0 * Rcb_.transpose();  // + Rcb_.transpose() * hat1;
                jacob_pose.block<3, 3>(0, 3) = -Rwc;
                jacob_pose = jacobp * jacob_pose;
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacob_pt(jacobians[1]);
                jacob_pt = jacobp * Rwc;
            }
        }
        return true;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (theta < 1.0e-08) {
            theta = 1.0e-08;
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }

private:
    Eigen::Vector3d obs_, tcb_;
    Eigen::Matrix3d Rcb_, sqrt_info_;
};

class CamProjPoseError : public ceres::SizedCostFunction<3, 6> {
public:
    CamProjPoseError(Eigen::Matrix3d sqrt_info, Eigen::Vector3d obs, Eigen::Vector3d pt_w)
        : sqrt_info_(sqrt_info), obs_(obs), pt_w_(pt_w) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rvec(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> twc(parameters[0] + 3);
        Eigen::Matrix3d Rwc = RotationVector2Matrix3d(rvec);

        Eigen::Vector3d ptc = Rwc * pt_w_ + twc;
        double pt_len = ptc.norm();
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = sqrt_info_ * ((ptc / pt_len) - obs_);

        double len3 = pt_len * pt_len * pt_len;
        double x = ptc(0), y = ptc(1), z = ptc(2);
        double len_inv = 1.0 / pt_len;
        Eigen::Matrix3d jacobp;
        jacobp << len_inv - x * x / len3, -x * y / len3, -x * z / len3, -x * y / len3, len_inv - y * y / len3, -y * z / len3, -x * z / len3,
            -y * z / len3, len_inv - z * z / len3;
        jacobp = sqrt_info_ * jacobp;

        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacob(jacobians[0]);
                Eigen::Matrix3d hat;
                hat << 0.0, -pt_w_(2), pt_w_(1), pt_w_(2), 0.0, -pt_w_(0), -pt_w_(1), pt_w_(0), 0.0;
                Eigen::Matrix3d tmp_Jr = Jr(rvec);
                jacob.block<3, 3>(0, 0) = -Rwc * hat * tmp_Jr;
                jacob.block<3, 3>(0, 3).setIdentity();
                jacob = jacobp * jacob;
            }
        }
        return true;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (theta < 1.0e-08) {
            theta = 1.0e-08;
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
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

private:
    Eigen::Matrix3d sqrt_info_;
    Eigen::Vector3d obs_, pt_w_;
};

class CamProjError : public ceres::SizedCostFunction<3, 6, 3> {
public:
    CamProjError(Eigen::Matrix3d sqrt_info, Eigen::Vector3d obs) : sqrt_info_(sqrt_info), obs_(obs) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rvec(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> twc(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> pt_w_(parameters[1]);

        Eigen::Matrix3d Rwc = RotationVector2Matrix3d(rvec);
        Eigen::Vector3d ptc = Rwc * pt_w_ + twc;
        double pt_len = ptc.norm();
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = sqrt_info_ * ((ptc / pt_len) - obs_);

        double len3 = pt_len * pt_len * pt_len;
        double x = ptc(0), y = ptc(1), z = ptc(2);
        double len_inv = 1.0 / pt_len;
        Eigen::Matrix3d jacobp;
        jacobp << len_inv - x * x / len3, -x * y / len3, -x * z / len3, -x * y / len3, len_inv - y * y / len3, -y * z / len3, -x * z / len3,
            -y * z / len3, len_inv - z * z / len3;
        jacobp = sqrt_info_ * jacobp;

        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacob(jacobians[0]);
                Eigen::Matrix3d hat;
                hat << 0.0, -pt_w_(2), pt_w_(1), pt_w_(2), 0.0, -pt_w_(0), -pt_w_(1), pt_w_(0), 0.0;
                Eigen::Matrix3d tmp_Jr = Jr(rvec);
                jacob.block<3, 3>(0, 0) = -Rwc * hat * tmp_Jr;
                jacob.block<3, 3>(0, 3).setIdentity();
                jacob = jacobp * jacob;
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> pt_jacob(jacobians[1]);
                pt_jacob = jacobp * Rwc;
            }
        }
        return true;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (theta < 1.0e-08) {
            theta = 1.0e-08;
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
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

private:
    Eigen::Matrix3d sqrt_info_;
    Eigen::Vector3d obs_;
};

class SixCamProjError : public ceres::SizedCostFunction<4, 6, 3> {
public:
    SixCamProjError(Eigen::Matrix4d sqrt_info, Eigen::Vector3d obs, Eigen::Vector3d t, double depth) : sqrt_info_(sqrt_info),
        obs_(obs), t_(t), depth_(depth) {
            sqrt_info_(3, 3) *= 0.025;
        }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rvec(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> twc(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> pt_w_(parameters[1]);

        Eigen::Matrix3d Rwc = RotationVector2Matrix3d(rvec);
        Eigen::Vector3d ptc0 = Rwc * pt_w_ + twc;
        Eigen::Vector3d ptc = ptc0 - t_;
        double pt_len = ptc.norm();
        Eigen::Map<Eigen::Vector4d> residual(residuals);
        residual.topRows(3) = ((ptc / pt_len) - obs_);
        residual(3) = (pt_len - depth_);
        residual = sqrt_info_ * residual;

        double len3 = pt_len * pt_len * pt_len;
        double x = ptc(0), y = ptc(1), z = ptc(2);
        double len_inv = 1.0 / pt_len;
        Eigen::Matrix<double, 4, 3> jacobp;
        jacobp.block<3, 3>(0, 0) << len_inv - x * x / len3, -x * y / len3, -x * z / len3, -x * y / len3, len_inv - y * y / len3, -y * z / len3, -x * z / len3,
            -y * z / len3, len_inv - z * z / len3;
        jacobp.block<1, 3>(3, 0) << -x / len3, -y / len3, -z / len3;

        jacobp = sqrt_info_ * jacobp;
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 4, 6, Eigen::RowMajor>> jacob(jacobians[0]);
                jacob.setZero();
                Eigen::Matrix3d hat;
                hat << 0.0, -pt_w_(2), pt_w_(1), pt_w_(2), 0.0, -pt_w_(0), -pt_w_(1), pt_w_(0), 0.0;
                Eigen::Matrix3d tmp_Jr = Jr(rvec);
                jacob.block<3, 3>(0, 0) = -Rwc * hat * tmp_Jr;
                jacob.block<3, 3>(0, 3).setIdentity();
                jacob = jacobp * jacob.block(0, 0, 3, 6);
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> pt_jacob(jacobians[1]);
                pt_jacob.setZero();
                pt_jacob.block(0, 0, 3, 3) = Rwc;
                pt_jacob = jacobp * pt_jacob.block(0, 0, 3, 3);
            }
        }
        return true;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (theta < 1.0e-08) {
            theta = 1.0e-08;
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
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

private:
    Eigen::Matrix4d sqrt_info_;
    Eigen::Vector3d obs_, t_;
    double depth_;
};

class DepthBasedPanoCost : public ceres::SizedCostFunction<4, 6, 3> {
public:
    DepthBasedPanoCost(Eigen::Matrix4d sqrt_info, Eigen::Matrix3d Rcb, Eigen::Vector3d tcb,
        Eigen::Vector3d t, Eigen::Vector3d obs, double depth) : sqrt_info_(sqrt_info), Rcb_(Rcb),
            tcb_(tcb), t_(t), obs_(obs), depth_(depth) {
            sqrt_info_(3, 3) *= 0.025;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Vector3d rv(parameters[0]);
        Eigen::Vector3d tbw(parameters[0] + 3);
        Eigen::Vector3d pt3d(parameters[1]);

        Eigen::Matrix3d Rbw = RotationVector2Matrix3d(rv);
        Eigen::Matrix3d Rwc = (Rbw * Rcb_).transpose();
        Eigen::Vector3d twc = -Rwc * (tbw + Rbw * tcb_);
        Eigen::Vector3d ptc = Rwc * pt3d + twc;

        double pt_len = ptc.norm();
        Eigen::Map<Eigen::Vector4d> res(residuals);
        res.segment<3>(0) = ((ptc / pt_len) - obs_);
        res(3) = (pt_len - depth_) / depth_;
        res = sqrt_info_ * res;

        double len3_inv = 1.0 / (pt_len * pt_len * pt_len);
        double x = ptc(0), y = ptc(1), z = ptc(2);
        double len_inv = 1.0 / pt_len;
        Eigen::Matrix<double, 4, 3> jacobp;
        jacobp.block<3, 3>(0, 0) << len_inv - x * x * len3_inv, -x * y * len3_inv, -x * z * len3_inv, -x * y * len3_inv, len_inv - y * y * len3_inv, -y * z * len3_inv, -x * z * len3_inv,
            -y * z * len3_inv, len_inv - z * z * len3_inv;
        Eigen::Vector3d J_tmp;
        J_tmp << x * len_inv, y * len_inv, z * len_inv;
        jacobp.block<1, 3>(3, 0) = J_tmp / depth_;
        jacobp = sqrt_info_ * jacobp;

        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 4, 6, Eigen::RowMajor>> jacob_pose(jacobians[0]);
                Eigen::Matrix<double, 3, 6> dpt_dp;
                Eigen::Vector3d tmp = Rwc * pt3d - Rwc * tbw;
                Eigen::Matrix3d hat0;
                hat0 << 0.0, -tmp(2), tmp(1), tmp(2), 0.0, -tmp(0), -tmp(1), tmp(0), 0.0;
                Eigen::Matrix3d hat1;
                hat1 << 0.0, -tcb_(2), tcb_(1), tcb_(2), 0.0, -tcb_(0), -tcb_(1), tcb_(0), 0.0;
                dpt_dp.block<3, 3>(0, 0) = hat0 * Rcb_.transpose();  // + Rcb_.transpose() * hat1;
                dpt_dp.block<3, 3>(0, 3) = -Rwc;
                jacob_pose = jacobp * dpt_dp;
            }
            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> jacob_pt(jacobians[1]);
                jacob_pt = jacobp * Rwc;
            }
        }

        return true;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (theta < 1.0e-08) {
            theta = 1.0e-08;
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }
private:
    Eigen::Matrix4d sqrt_info_;
    Eigen::Matrix3d Rcb_;
    Eigen::Vector3d tcb_, t_, obs_;
    double depth_;
};

class StereoPanoCost : public ceres::SizedCostFunction<6, 6, 3> {
public:
    StereoPanoCost(Eigen::Matrix<double, 6, 6> sqrt_info, Eigen::Matrix3d Rcb, Eigen::Vector3d tcb,
        Eigen::Vector3d t0, Eigen::Vector3d t1, Eigen::Vector3d obs0, Eigen::Vector3d obs1)
        : sqrt_info_(sqrt_info), Rcb_(Rcb), tcb_(tcb), t0_(t0), t1_(t1), obs0_(obs0), obs1_(obs1) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rv(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tbw(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> ptw(parameters[1]);

        Eigen::Matrix3d Rbw = RotationVector2Matrix3d(rv);
        Eigen::Matrix3d Rwc = (Rbw * Rcb_).transpose();
        Eigen::Vector3d twc = -Rwc * (tbw + Rbw * tcb_);
        Eigen::Vector3d pt0 = Rwc * ptw + twc;
        Eigen::Vector3d lpt = pt0 + t0_;
        Eigen::Vector3d rpt = pt0 + t1_;

        double lpt_len = lpt.norm();
        double rpt_len = rpt.norm();
        Eigen::Map<Eigen::VectorXd> residual(residuals, 6);

        residual.segment<3>(0) = lpt / lpt_len - obs0_;
        residual.segment<3>(3) = rpt / rpt_len - obs1_;
        residual = sqrt_info_ * residual;

        Eigen::MatrixXd jacobp(6, 3);
        jacobp.setZero();
        jacobp.block<3, 3>(0, 0) = Jpt(lpt, lpt_len);
        jacobp.block<3, 3>(3, 0) = Jpt(rpt, rpt_len);
        jacobp = sqrt_info_ * jacobp;
        if(jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jacob_pose(jacobians[0]);
                jacob_pose.setZero();
                Eigen::Matrix<double, 3, 6> tmpJ;
                Eigen::Vector3d tmp = Rwc * ptw - Rwc * tbw;
                Eigen::Matrix3d hat0;
                hat0 << 0.0, -tmp(2), tmp(1), tmp(2), 0.0, -tmp(0), -tmp(1), tmp(0), 0.0;
                Eigen::Matrix3d hat1;
                hat1 << 0.0, -tcb_(2), tcb_(1), tcb_(2), 0.0, -tcb_(0), -tcb_(1), tcb_(0), 0.0;

                tmpJ.block<3, 3>(0, 0) = hat0 * Rcb_.transpose();
                tmpJ.block<3, 3>(0, 3) = -Rwc;
                jacob_pose = jacobp * tmpJ;
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jacob_pt(jacobians[1]);
                jacob_pt.setZero();
                jacob_pt = jacobp * Rwc;
            }
        }
        return true;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (theta < 1.0e-08) {
            theta = 1.0e-08;
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }
    inline Eigen::Matrix3d Jpt(const Eigen::Vector3d ptc, double len) const {
        double len3_inv = 1.0 / (len * len * len);
        double x = ptc(0), y = ptc(1), z = ptc(2);
        double len_inv = 1.0 / len;
        Eigen::Matrix<double, 3, 3> jacobp;
        jacobp << len_inv - x * x * len3_inv, -x * y * len3_inv, -x * z * len3_inv, -x * y * len3_inv, len_inv - y * y * len3_inv, -y * z * len3_inv, -x * z * len3_inv,
            -y * z * len3_inv, len_inv - z * z * len3_inv;
        return jacobp;
    }
private:
    Eigen::Matrix<double, 6, 6> sqrt_info_;
    Eigen::Matrix3d Rcb_;
    Eigen::Vector3d tcb_, t0_, t1_, obs0_, obs1_;
};

class PanoCost : public ceres::SizedCostFunction<3, 6, 3> {
public:
    PanoCost(Eigen::Matrix3d sqrt_info, Eigen::Matrix3d Rcb, Eigen::Vector3d tcb, Eigen::Vector3d t, Eigen::Vector3d obs)
        : sqrt_info_(sqrt_info), Rcb_(Rcb), tcb_(tcb), t_(t), obs_(obs) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rv(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tbw(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> ptw(parameters[1]);

        Eigen::Matrix3d Rbw = RotationVector2Matrix3d(rv);
        Eigen::Matrix3d Rwc = (Rbw * Rcb_).transpose();
        Eigen::Vector3d twc = -Rwc * (tbw + Rbw * tcb_);
        Eigen::Vector3d ptc0 = Rwc * ptw + twc;
        Eigen::Vector3d ptc = ptc0 + t_;
        double pt_len = ptc.norm();

        Eigen::Map<Eigen::Vector3d> res(residuals);
        res = sqrt_info_ * (ptc / pt_len - obs_);

        double len3 = pt_len * pt_len * pt_len;
        double x = ptc(0), y = ptc(1), z = ptc(2);
        double len_inv = 1.0 / pt_len;
        Eigen::Matrix<double, 3, 3> jacobp;
        jacobp << len_inv - x * x / len3, -x * y / len3, -x * z / len3, -x * y / len3, len_inv - y * y / len3, -y * z / len3, -x * z / len3,
            -y * z / len3, len_inv - z * z / len3;

        jacobp = sqrt_info_ * jacobp;
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacob_pose(jacobians[0]);
                jacob_pose.setZero();
                Eigen::Vector3d tmp = Rwc * ptw - Rwc * tbw;
                Eigen::Matrix3d hat0;
                hat0 << 0.0, -tmp(2), tmp(1), tmp(2), 0.0, -tmp(0), -tmp(1), tmp(0), 0.0;
                Eigen::Matrix3d hat1;
                hat1 << 0.0, -tcb_(2), tcb_(1), tcb_(2), 0.0, -tcb_(0), -tcb_(1), tcb_(0), 0.0;

                jacob_pose.block<3, 3>(0, 0) = hat0 * Rcb_.transpose();
                jacob_pose.block<3, 3>(0, 3) = -Rwc;
                jacob_pose = jacobp * jacob_pose;
            }

            if (jacobians[1] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobpt(jacobians[1]);
                jacobpt.setZero();
                jacobpt = jacobp * Rwc;
            }
        }


        return true;
    }

    inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) const {
        double theta = rvec.norm();
        if (theta < 1.0e-08) {
            theta = 1.0e-08;
        }
        Eigen::Vector3d nvec = rvec / theta;
        Eigen::Matrix3d n_hat;
        n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
        Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
        return R;
    }
private:
    Eigen::Matrix3d sqrt_info_, Rcb_;
    Eigen::Vector3d tcb_, t_, obs_;
};


}  // namespace inslam
#endif
