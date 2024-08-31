
#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace inslam {

class Triangulator {
public:
    Triangulator() = default;

    /* Solve multi-view triangulation */
    static bool Solve(const std::vector<Eigen::Vector3d> &rays,
                      const std::vector<Eigen::Matrix<double, 3, 4> > &poses,
                      double angularResolution,
                      Eigen::Vector3d &X,
                      bool checkParallax = true);


    /* Triangulation with unit sphere points.
     * Rays: normalized viewing vector.
     * poses: T_Ci_W
     * angularResolution: the angle corresponding to one pixel
     */
    static bool Optimize(const std::vector<Eigen::Vector3d> &rays,
                      const std::vector<Eigen::Matrix<double, 3, 4> > &poses,
                      double angularResolution,
                      Eigen::Vector3d &X);


    /* Check parallax.
     * rays: viewing vectors in world frame
     * minimalAngle: in rad, default value is 2 degree
     */
    static bool CheckParallax(const std::vector<Eigen::Vector3d> &rays,
                              const std::vector<Eigen::Matrix<double, 3, 4> > &poses,
                              double minimalAngle = 0.035);

    /* Check parallax.
     * rays: viewing vectors in world frame
     * minimalAngle: in rad, default value is 2 degree
     */
    static bool CheckParallax(const std::vector<Eigen::Vector3d> &rays, double minimalAngle = 0.035);

};

}//namespace inslam {