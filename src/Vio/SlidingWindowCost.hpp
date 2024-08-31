#ifndef SLIDINGWINDOWCOST_H
#define SLIDINGWINDOWCOST_H
#include <math.h>
#include <ceres/ceres.h>

#include "Utils/MathUtil.h"
namespace inslam {
inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d &rvec) {
    double theta = rvec.norm();
    if (theta < 1.0e-09) {
        theta = 1.0e-09;
    }
    Eigen::Vector3d nvec = rvec / theta;
    Eigen::Matrix3d n_hat;
    n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
    Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
    return R;
}

// R = cos(theta) * I33 + (1 - cos(theta)) * u * u^t + sin(theta) * u^
// => trace(R) = 3 * cos(theta) + (1 - cos(theta)) = 1 + 2 * cos(theta)
// => cos(theta) = (trace(R) - 1) * 0.5;
// R * u = u
inline Eigen::Vector3d RotationMatrix2Vector3d(const Eigen::Matrix3d &R) {
    Eigen::AngleAxisd rv(R);
    return rv.angle() * rv.axis();

    double cos_theta = (R.trace() - 1.0) * 0.5;  
    double theta = acos(cos_theta);
    Eigen::Matrix3d Right = (R - R.transpose()) * 0.5;
    Eigen::Vector3d rvec;
    rvec[0] = (Right(2, 1) - Right(1, 2)) * 0.5;
    rvec[1] = (Right(0, 2) - Right(2, 0)) * 0.5;
    rvec[2] = (Right(1, 0) - Right(0, 1)) * 0.5;
    if (cos_theta > 1.0 - 1.0e-09) {
        return Eigen::Vector3d(0, 0, 0);
    }
    if (cos_theta < 1.0e-09 - 1.0) {
        return 3.1415926535898 * rvec / rvec.norm();
    }
    rvec /= rvec.norm();

    return rvec * theta;
}

// parameters = [poseg, poseh, rho]
class WindowProjCost : public ceres::SizedCostFunction<2, 6, 6, 1> {
  public:
    WindowProjCost(Eigen::Matrix2d sqrt_info, Eigen::Matrix<double, 3, 4> Tci, Eigen::Vector3d obs,
                   Eigen::Vector3d host_obs, Eigen::Matrix<double, 2, 3> tangentplane)
                   : sqrt_info_(sqrt_info), Rci_(Tci.leftCols(3)), tci_(Tci.rightCols(1)), host_obs_(host_obs),
                     obs_(obs), tangentplane_(tangentplane) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rvecg(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tiwg(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> rvech(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> tiwh(parameters[1] + 3);
        const double &rho = parameters[2][0];

        Eigen::Matrix3d Riwg = RotationVector2Matrix3d(rvecg);
        Eigen::Matrix3d Riwh = RotationVector2Matrix3d(rvech);

        Eigen::Matrix3d Rwcg = (Riwg * Rci_).transpose();
        Eigen::Vector3d twcg = -Rwcg * (tiwg + Riwg * tci_); // = -Rwcg * tiwg - Ric * tci

        Eigen::Matrix3d Rcwh = Riwh * Rci_;
        Eigen::Vector3d tcwh = tiwh + Riwh * tci_;
        Eigen::Vector3d Pw = Rcwh * host_obs_ / rho + tcwh;
        Eigen::Vector3d Pcg = Rwcg * Pw + twcg;
        double rhog = 1.0 / Pcg.norm();

        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = sqrt_info_ * tangentplane_ * (obs_ - Pcg * rhog);
        Eigen::Vector3d error = obs_ - Pcg / Pcg.norm();

        // de/dPcg = -d(Pcg * rhog)/dPcg = -dPcg/dPcg * rhog - Pcg * d(rhog)/dPcg = -rhog * I - Pcg * d(rhog)/dPcg
        // = -rhog * I + Pcg * Pcg^t * (rhog^3)
        // dr/de = tangplane
        Eigen::Matrix<double, 3, 3> de_dPcg; de_dPcg.setIdentity();
        de_dPcg *= -rhog;
        de_dPcg += Pcg * Pcg.transpose() * (rhog * rhog * rhog);
        Eigen::Matrix<double, 2, 3> dr_dPcg = sqrt_info_ * tangentplane_ * de_dPcg;

        // Jacobians
        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacob_poseg(jacobians[0]);
                // Rwcg * exp(delta1) = (Riwg * exp(delta0) * Rci)^t = (exp(Riwg * delta0) * Riwg * Rci)^t
                // = Rwcg * exp(-Riwg * delta0) => delta1 = -Riwg * delta0
                // twcg + dt1 = -Rwcg * exp(delta1) * (tiwg + dt0) - Ric * tci = -Rwcg * exp(delta1) * tiwg - Rwcg * dt0
                // = -Rwcg * (I + delta1^) * tiwg - Rwcg * dt0 = twcg + Rwcg * tiwg^ * delta1 - Rwcg * dt0
                // dt1 = Rwcg * tiwg^ * delta1 - Rwcg * dt0
                // Pcg + dP = Rwcg * exp(delta1) * Pw + twcg + dt1 = Rwcg * (I + delta1^) * Pw + twcg + dt1
                // =Rwcg * Pw + twcg - Rwcg * Pw^ * delta1 + dt1
                // dP = -Rwcg * Pw^ * delta1 + dt1 = -Rwcg * Pw^ * delta1 + Rwcg * tiwg^ * delta1 - Rwcg * dt0
                // = Rwcg * (tiwg - Pw)^ * delta1 - Rwcg * dt0
                jacob_poseg.block<2, 3>(0, 0) = dr_dPcg * Rwcg * MathUtil::VecToSkew(tiwg - Pw) * (-Riwg);
                jacob_poseg.block<2, 3>(0, 3) = dr_dPcg * (-Rwcg);
            }
            if (jacobians[1] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacob_poseh(jacobians[1]);
                // Pcg + dP1 = Rwcg * (Pw + dP0) + twcg => dP1 = Rwcg * dP0
                // Rcwh * exp(delta1) = Riwh * exp(delta0) * Rci = Riwh * Rci * exp(Rci^t * delta0)
                // => delta1 = Rci^t * delta0
                // tcwh + dt1 = tiwh + dt0 + Riwh * exp(delta0) * tci = tiwh + dt0 + Riwh * (I + delta0^) * tci
                // = tcwh + dt0 + Riwh * delta0^ * tci => dt1 = dt0 -Riwh * tci^ * delta0
                // Pw + dP0 = Rcwh * exp(delta1) * Pch + tcwh + dt1 = Rcwh * (I + delta1^) * Pch + tcwh + dt1
                // = Pw + Rcwh * delta1^ * Pch + dt1 = Pw - Rcwh * Pch^ * delta1 + dt1
                // dP0 = -Rcwh * Pch^ * Rci^t * delta0 - Riwh * tci^ * delta0 + dt0
                jacob_poseh.block<2, 3>(0, 0) = dr_dPcg * Rwcg * (-Rcwh * MathUtil::VecToSkew(host_obs_/rho) * Rci_.transpose()
                                                                  - Riwh * MathUtil::VecToSkew(tci_));
                jacob_poseh.block<2, 3>(0, 3) = dr_dPcg * Rwcg;
            }
            if (jacobians[2] != nullptr) {
                Eigen::Map<Eigen::Vector2d> jacob_rho(jacobians[2]);
                // Ph = rayh / rho => dPh/drho = -rayh / (rho * rho)
                // Pw = Rcwh * Ph + tcwh => dPw/dPh = Rcwh
                jacob_rho = dr_dPcg * Rwcg * Rcwh * (-host_obs_ / (rho * rho));
            }
        }
        return true;
    }
  private:
    Eigen::Matrix3d Rci_;
    Eigen::Matrix<double, 2, 3> tangentplane_;
    Eigen::Vector3d obs_, host_obs_, tci_;
    Eigen::Matrix2d sqrt_info_;
};

class WindowStereoProjCost : public ceres::SizedCostFunction<4, 6, 6, 1> {
  public:
    WindowStereoProjCost(Eigen::Matrix2d sqrt_info, Eigen::Matrix<double, 3, 4> Tci, std::vector<Eigen::Vector3d> obs,
                   Eigen::Vector3d host_obs, std::vector<Eigen::Matrix<double, 2, 3>> tangentplanes,
                   Eigen::Matrix<double, 3, 4> Tc0c1)
                   : sqrt_info_(sqrt_info), Rci_(Tci.leftCols(3)), tci_(Tci.rightCols(1)), host_obs_(host_obs),
                     obs_(obs), tangentplanes_(tangentplanes), Rc0c1_(Tc0c1.leftCols(3)), tc0c1_(Tc0c1.rightCols(1)) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rvecg(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tiwg(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> rvech(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> tiwh(parameters[1] + 3);
        const double &rho = parameters[2][0];

        Eigen::Matrix3d Riwg = RotationVector2Matrix3d(rvecg);
        Eigen::Matrix3d Riwh = RotationVector2Matrix3d(rvech);

        Eigen::Matrix3d Rwcg = (Riwg * Rci_).transpose();
        Eigen::Vector3d twcg = -Rwcg * (tiwg + Riwg * tci_); // = -Rwcg * tiwg - Ric * tci

        Eigen::Matrix3d Rcwh = Riwh * Rci_;
        Eigen::Vector3d tcwh = tiwh + Riwh * tci_;
        Eigen::Vector3d Pw = Rcwh * host_obs_ / rho + tcwh;
        Eigen::Vector3d Pcg0 = Rwcg * Pw + twcg;
        Eigen::Vector3d Pcg1 = Rc0c1_ * Pcg0 + tc0c1_;
        double rhog0 = 1.0 / Pcg0.norm();
        double rhog1 = 1.0 / Pcg1.norm();

        Eigen::Map<Eigen::Vector4d> residual(residuals);
        residual.topRows(2) = sqrt_info_ * tangentplanes_[0] * (obs_[0] - Pcg0 * rhog0);
        residual.bottomRows(2) = sqrt_info_ * tangentplanes_[1] * (obs_[1] - Pcg1 * rhog1);

        // de/dPcg = -d(Pcg * rhog)/dPcg = -dPcg/dPcg * rhog - Pcg * d(rhog)/dPcg = -rhog * I - Pcg * d(rhog)/dPcg
        // = -rhog * I + Pcg * Pcg^t * (rhog^3)
        // dr/de = tangplane
        Eigen::Matrix<double, 3, 3> de_dPcg0; de_dPcg0.setIdentity();
        de_dPcg0 *= -rhog0;
        de_dPcg0 += Pcg0 * Pcg0.transpose() * (rhog0 * rhog0 * rhog0);
        Eigen::Matrix<double, 2, 3> dr_dPcg0 = sqrt_info_ * tangentplanes_[0] * de_dPcg0;

        Eigen::Matrix<double, 3, 3> de_dPcg1; de_dPcg1.setIdentity();
        de_dPcg1 *= -rhog1;
        de_dPcg1 += Pcg1 * Pcg1.transpose() * (rhog1 * rhog1 * rhog1);
        Eigen::Matrix<double, 2, 3> dr_dPcg1 = sqrt_info_ * tangentplanes_[1] * de_dPcg1;

        // Jacobians
        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 4, 6, Eigen::RowMajor>> jacob_poseg(jacobians[0]);
                // Rwcg * exp(delta1) = (Riwg * exp(delta0) * Rci)^t = (exp(Riwg * delta0) * Riwg * Rci)^t
                // = Rwcg * exp(-Riwg * delta0) => delta1 = -Riwg * delta0
                // twcg + dt1 = -Rwcg * exp(delta1) * (tiwg + dt0) - Ric * tci = -Rwcg * exp(delta1) * tiwg - Rwcg * dt0
                // = -Rwcg * (I + delta1^) * tiwg - Rwcg * dt0 = twcg + Rwcg * tiwg^ * delta1 - Rwcg * dt0
                // dt1 = Rwcg * tiwg^ * delta1 - Rwcg * dt0
                // Pcg + dP = Rwcg * exp(delta1) * Pw + twcg + dt1 = Rwcg * (I + delta1^) * Pw + twcg + dt1
                // =Rwcg * Pw + twcg - Rwcg * Pw^ * delta1 + dt1
                // dP = -Rwcg * Pw^ * delta1 + dt1 = -Rwcg * Pw^ * delta1 + Rwcg * tiwg^ * delta1 - Rwcg * dt0
                // = Rwcg * (tiwg - Pw)^ * delta1 - Rwcg * dt0
                Eigen::Matrix3d dPcg0_dthetag = Rwcg * MathUtil::VecToSkew(tiwg - Pw) * (-Riwg);
                jacob_poseg.block<2, 3>(0, 0) = dr_dPcg0 * dPcg0_dthetag;
                jacob_poseg.block<2, 3>(0, 3) = dr_dPcg0 * (-Rwcg);

                // Pcg1 + dP1 = Rc0c1 * (Pcg0 + dP0) + tc0c1 => dP1 = Rc0c1 * dP0
                jacob_poseg.block<2, 3>(2, 0) = dr_dPcg1 * Rc0c1_ * dPcg0_dthetag;
                jacob_poseg.block<2, 3>(2, 3) = dr_dPcg1 * Rc0c1_ * (-Rwcg);
            }
            if (jacobians[1] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 4, 6, Eigen::RowMajor>> jacob_poseh(jacobians[1]);
                // Pcg + dP1 = Rwcg * (Pw + dP0) + twcg => dP1 = Rwcg * dP0
                // Rcwh * exp(delta1) = Riwh * exp(delta0) * Rci = Riwh * Rci * exp(Rci^t * delta0)
                // => delta1 = Rci^t * delta0
                // tcwh + dt1 = tiwh + dt0 + Riwh * exp(delta0) * tci = tiwh + dt0 + Riwh * (I + delta0^) * tci
                // = tcwh + dt0 + Riwh * delta0^ * tci => dt1 = dt0 -Riwh * tci^ * delta0
                // Pw + dP0 = Rcwh * exp(delta1) * Pch + tcwh + dt1 = Rcwh * (I + delta1^) * Pch + tcwh + dt1
                // = Pw + Rcwh * delta1^ * Pch + dt1 = Pw - Rcwh * Pch^ * delta1 + dt1
                // dP0 = -Rcwh * Pch^ * Rci^t * delta0 - Riwh * tci^ * delta0 + dt0
                Eigen::Matrix3d dPcg0_dthetah = Rwcg * (-Rcwh * MathUtil::VecToSkew(host_obs_/rho) * Rci_.transpose()
                                                                  - Riwh * MathUtil::VecToSkew(tci_));
                jacob_poseh.block<2, 3>(0, 0) = dr_dPcg0 * dPcg0_dthetah;
                jacob_poseh.block<2, 3>(0, 3) = dr_dPcg0 * Rwcg;

                // dP1 = Rc0c1 * dP0
                jacob_poseh.block<2, 3>(2, 0) = dr_dPcg1 * Rc0c1_ * dPcg0_dthetah;
                jacob_poseh.block<2, 3>(2, 3) = dr_dPcg1 * Rc0c1_ * Rwcg;
            }
            if (jacobians[2] != nullptr) {
                Eigen::Map<Eigen::Vector4d> jacob_rho(jacobians[2]);
                // Ph = rayh / rho => dPh/drho = -rayh / (rho * rho)
                // Pw = Rcwh * Ph + tcwh => dPw/dPh = Rcwh
                Eigen::Vector3d dPcg0_drho = Rwcg * Rcwh * (-host_obs_ / (rho * rho));
                jacob_rho.topRows(2) = dr_dPcg0 * dPcg0_drho;
                jacob_rho.bottomRows(2) = dr_dPcg1 * Rc0c1_ * dPcg0_drho;
            }
        }
        return true;
    }
  private:
    Eigen::Matrix3d Rci_, Rc0c1_;
    std::vector<Eigen::Matrix<double, 2, 3>> tangentplanes_;
    std::vector<Eigen::Vector3d> obs_;
    Eigen::Vector3d host_obs_, tci_, tc0c1_;
    Eigen::Matrix2d sqrt_info_;
};

class ProjCost : public ceres::SizedCostFunction<2, 6> {
  public:
    ProjCost(Eigen::Matrix<double, 3, 4> Tci, Eigen::Vector3d obs, Eigen::Matrix<double, 3, 4> Tiwh,
            Eigen::Vector3d host_obs, Eigen::Matrix<double, 2, 3> tangentplane, double idepth)
            : Rci_(Tci.leftCols(3)), tci_(Tci.rightCols(1)), host_obs_(host_obs), Riwh_(Tiwh.leftCols(3)),
              tiwh_(Tiwh.rightCols(1)), obs_(obs), tangentplane_(tangentplane), idepth_(idepth) {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Map<const Eigen::Vector3d> rvecg(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tiwg(parameters[0] + 3);

        Eigen::Matrix3d Riwg = RotationVector2Matrix3d(rvecg);

        Eigen::Matrix3d Rwcg = (Riwg * Rci_).transpose();
        Eigen::Vector3d twcg = -Rwcg * (tiwg + Riwg * tci_); // = -Rwcg * tiwg - Ric * tci

        Eigen::Matrix3d Rcwh = Riwh_ * Rci_;
        Eigen::Vector3d tcwh = tiwh_ + Riwh_ * tci_;
        Eigen::Vector3d Pw = Rcwh * host_obs_ / idepth_ + tcwh;
        Eigen::Vector3d Pcg = Rwcg * Pw + twcg;
        double rhog = 1.0 / Pcg.norm();

        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = tangentplane_ * (obs_ - Pcg * rhog);
        Eigen::Vector3d error = obs_ - Pcg / Pcg.norm();

        // de/dPcg = -d(Pcg * rhog)/dPcg = -dPcg/dPcg * rhog - Pcg * d(rhog)/dPcg = -rhog * I - Pcg * d(rhog)/dPcg
        // = -rhog * I + Pcg * Pcg^t * (rhog^3)
        // dr/de = tangplane
        Eigen::Matrix<double, 3, 3> de_dPcg; de_dPcg.setIdentity();
        de_dPcg *= -rhog;
        de_dPcg += Pcg * Pcg.transpose() * (rhog * rhog * rhog);
        Eigen::Matrix<double, 2, 3> dr_dPcg = tangentplane_ * de_dPcg;

        // Jacobians
        if (jacobians != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacob_poseg(jacobians[0]);
            // Rwcg * exp(delta1) = (Riwg * exp(delta0) * Rci)^t = (exp(Riwg * delta0) * Riwg * Rci)^t
            // = Rwcg * exp(-Riwg * delta0) => delta1 = -Riwg * delta0
            // twcg + dt1 = -Rwcg * exp(delta1) * (tiwg + dt0) - Ric * tci = -Rwcg * exp(delta1) * tiwg - Rwcg * dt0
            // = -Rwcg * (I + delta1^) * tiwg - Rwcg * dt0 = twcg + Rwcg * tiwg^ * delta1 - Rwcg * dt0
            // dt1 = Rwcg * tiwg^ * delta1 - Rwcg * dt0
            // Pcg + dP = Rwcg * exp(delta1) * Pw + twcg + dt1 = Rwcg * (I + delta1^) * Pw + twcg + dt1
            // =Rwcg * Pw + twcg - Rwcg * Pw^ * delta1 + dt1
            // dP = -Rwcg * Pw^ * delta1 + dt1 = -Rwcg * Pw^ * delta1 + Rwcg * tiwg^ * delta1 - Rwcg * dt0
            // = Rwcg * (tiwg - Pw)^ * delta1 - Rwcg * dt0
            jacob_poseg.block<2, 3>(0, 0) = dr_dPcg * Rwcg * MathUtil::VecToSkew(tiwg - Pw) * (-Riwg);
            jacob_poseg.block<2, 3>(0, 3) = dr_dPcg * (-Rwcg);
        }
        return true;
    }
  private:
    Eigen::Matrix3d Rci_, Riwh_;
    Eigen::Matrix<double, 2, 3> tangentplane_;
    Eigen::Vector3d obs_, host_obs_, tci_, tiwh_;
    double idepth_;
};

} // namespace inslam
#endif // SLIDINGWINDOWCOST_H