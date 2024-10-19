//
// Created by d on 2020/9/21.
//

#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace featslam{

class MathUtil {
public:
    MathUtil() = default;

    static double PoseToYaw(const Eigen::Matrix4d& pose);

    static double QuatToYaw(const Eigen::Quaterniond &q);

    // convert R_world_body to yaw, pitch, roll
    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R);

    // convert yaw, pitch, roll to R_world_body
    static Eigen::Matrix3d ypr2R(const Eigen::Vector3d &ypr);

    static double Angdiff(double a, double b);

    // T^a_c = T^a_b * T^b_c
    static Eigen::Matrix<double, 3, 4> PoseMultiply(
            const Eigen::Matrix<double, 3, 4> &pose1,
            const Eigen::Matrix<double, 3, 4> &pose2);

    /// Q <- qa * (1 - ratio) + qb * ratio
    static Eigen::Quaterniond QuatInterpolate(const Eigen::Quaterniond &qa,
                                                                  const Eigen::Quaterniond &qb,
                                                                  double ratio);

    /// T <- Ta * (1 - ratio) + Tb * ratio
    static Eigen::Matrix4d PoseInterpolate(const Eigen::Matrix4d &Ta,
                                                            const Eigen::Matrix4d&Tb,
                                                            double ratio);

    // Convert a small vector to quaternion
    static Eigen::Quaterniond VecToQuat(const Eigen::Vector3d &v);

    // Create the skew matrix of a vector
    static Eigen::Matrix3d VecToSkew(const Eigen::Vector3d &v);

    // JPL quaternion to rotation matrix
    static Eigen::Matrix3d JPLQuat2Matrix(const Eigen::Vector4d &Q);

    // rotation matrix to JPL quaternion
    static Eigen::Vector4d JPLMatrix2Quat(const Eigen::Matrix3d &m);

    // JPL quaternion multiply: out = Q1 * Q2
    static Eigen::Vector4d JPLQuatMultiply(const Eigen::Vector4d &Q1, const Eigen::Vector4d &Q2);

    // right jacobian of v
    static Eigen::Matrix3d Jr(const Eigen::Vector3d &v);

    // Rodrigues rotation formula
    static Eigen::Matrix3d Vec2RotationMatrix(const Eigen::Vector3d &v);

    // small angle to JPL Quaternion
    static Eigen::Vector4d SmallAngle2Quat(Eigen::Vector3d dtheta);

    // Rotation matrix to vector
    static Eigen::Vector3d RotationMatrix2Vec(const Eigen::Matrix3d &R);

    // Homemade quaternion( in vector form) increment function
    static void IncreQuat(const Eigen::Vector4d &q, const Eigen::Vector3d &v, Eigen::Vector4d& pq);

    // Delete elements from eigen vector
    static void ReduceEigenVector(Eigen::VectorXd &x, const Eigen::VectorXi &isbad);

    // Delete (rows & cols) from eigen matrix
    static void ReduceEigenMatrix(Eigen::MatrixXd &m, const Eigen::VectorXi &isbad);

    static Eigen::Quaterniond GetQfromA(const Eigen::Vector3d &a_corrected);

    static Eigen::Matrix3d NormalizationJacobian(const Eigen::Vector3d &v);

    static Eigen::Matrix<double, 1, 3> InverseNormJacobian(const Eigen::Vector3d &v);

    static void CalcTangentBase(const Eigen::Vector3d &ray, Eigen::Matrix<double, 2, 3> &tangentBase);

    // line: ax + by + c = 0
    // coeff: [a, b, c]
    static bool LineFittingRansac(const std::vector<double> &xs,
                                  const std::vector<double> &ys,
                                  double distThresh,
                                  Eigen::Vector3d &coeff);

    // y = coeff[0] + coeff[1] * x + ... + coeff[K-1] * x^(K-1)
    static bool CurveFitting(const std::vector<double> &xs,
                                const std::vector<double> &ys,
                                int rank,
                                Eigen::VectorXd &coeff);

    // solve equation: y = c0 * x + c1 * x^2 + c2 * x^3 + ...
    static bool SolvePolynomialEquation(const std::vector<double> &coeffs, double y, double &x,
                                        int maxIterations = 50, double eps = 1e-4);

};

}//namespace featslam{

