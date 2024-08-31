//
// Created by d on 2021/1/22.
//

#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace inslam {

struct SphereReproErrorX {
    SphereReproErrorX(const Eigen::Vector3d &ray, const Eigen::Matrix<double, 3, 4> &pose, double sqrtInfo)
        : ray_(ray), pose_(pose), sqrtInfo_(sqrtInfo) {
        // Calculate tangent base of this ray
        Eigen::Vector3d tmp(0, 0, 1);
        if (ray == tmp) {
            tmp << 1, 0, 0;
        }
        Eigen::Vector3d b1 = (tmp - ray * (ray.transpose() * tmp)).normalized();
        Eigen::Vector3d b2 = ray.cross(b1);
        tangentBase_.row(0) = b1.transpose();
        tangentBase_.row(1) = b2.transpose();
    }

    template<typename T>
    bool operator()(const T* xw, T* residual) const {
        // position in camera frame
        T xc[3];
        xc[0] = pose_(0, 0) * xw[0]  + pose_(0, 1) * xw[1] + pose_(0, 2) * xw[2] + pose_(0, 3);
        xc[1] = pose_(1, 0) * xw[0]  + pose_(1, 1) * xw[1] + pose_(1, 2) * xw[2] + pose_(1, 3);
        xc[2] = pose_(2, 0) * xw[0]  + pose_(2, 1) * xw[1] + pose_(2, 2) * xw[2] + pose_(2, 3);

        // normalize
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > pc(xc);
        Eigen::Matrix<T, 3, 1> rayc = pc.normalized();

        Eigen::Map<Eigen::Matrix<T, 2, 1> > resi(residual);
        resi = sqrtInfo_ * tangentBase_ * (ray_ - rayc);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d &ray,
                                       const Eigen::Matrix<double, 3, 4> &pose,
                                       double sqrtInfo) {
        return (new ceres::AutoDiffCostFunction<SphereReproErrorX, 2, 3>(
            new SphereReproErrorX(ray, pose, sqrtInfo) ) );
    }

private:
    Eigen::Vector3d ray_;
    Eigen::Matrix<double, 3, 4> pose_;
    Eigen::Matrix<double, 2, 3> tangentBase_;
    double sqrtInfo_;
};

}//namespace inslam {