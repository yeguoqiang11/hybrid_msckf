
#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace inslam {

/* Reprojection error for solving 3D point position.
 * The camera poses are fixed.
 */

struct ReproErrorX {
    ReproErrorX(const Eigen::Vector2d &pt, const Eigen::Matrix<double, 3, 4> &pose, const Eigen::Vector4d &fc)
        : pt_(pt), pose_(pose), fc_(fc) {}

    template<typename T>
    bool operator()(const T* xw, T* residual) const {
        // position in camera frame
        T xc[3];
        xc[0] = pose_(0, 0) * xw[0]  + pose_(0, 1) * xw[1] + pose_(0, 2) * xw[2] + pose_(0, 3);
        xc[1] = pose_(1, 0) * xw[0]  + pose_(1, 1) * xw[1] + pose_(1, 2) * xw[2] + pose_(1, 3);
        xc[2] = pose_(2, 0) * xw[0]  + pose_(2, 1) * xw[1] + pose_(2, 2) * xw[2] + pose_(2, 3);

        // projection
        T iz = T(1.0) / xc[2];
        residual[0] = fc_(0) * xc[0] * iz + fc_(2) - pt_(0);
        residual[1] = fc_(1) * xc[1] * iz + fc_(3) - pt_(1);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector2d &pt,
                                       const Eigen::Matrix<double, 3, 4> &pose,
                                       const Eigen::Vector4d &fc) {
        return (new ceres::AutoDiffCostFunction<ReproErrorX, 2, 3>(
                new ReproErrorX(pt, pose, fc) ) );
    }

private:
    Eigen::Vector2d pt_;
    Eigen::Matrix<double, 3, 4> pose_;
    Eigen::Vector4d fc_;    // fx, fy, cx, cy

};

}//namespace inslam {