//
// Created by d on 2020/9/21.
//

#include "Utils/MathUtil.h"
//#include "spdlog/spdlog.h"
#include <random>

using namespace Eigen;

namespace inslam {

double MathUtil::PoseToYaw(const Eigen::Matrix4d &pose) {
    const Eigen::Matrix3d R = pose.topLeftCorner<3, 3>();
    Eigen::Quaterniond q(R);
    return QuatToYaw(q);
}


double MathUtil::QuatToYaw(const Eigen::Quaterniond &q) {
    double yaw = std::atan2(2 * (q.w()*q.z() - q.x()*q.y()), 2 * (q.w()*q.w() + q.x()*q.x()) - 1);
    return yaw;
}


Eigen::Vector3d MathUtil::R2ypr(const Eigen::Matrix3d &R) {
    const Vector3d &n = R.col(0);
    const Vector3d &o = R.col(1);
    const Vector3d &a = R.col(2);

    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));

    return {y, p, r};
}


/* convert yaw, pitch, roll to R_world_body.
 * Suppose the the body frame is aligned with the world frame in the beginning,
 * then it rotates around z axis by 'yaw' degrees, then rotate around its y axis by
 * 'pitch' degrees, then its axis by 'roll' degrees.
 * R_body_world = R(roll) * R(pitch) * R(yaw)
 * R_world_body = R'(yaw) * R'(pitch) * R'(roll)
 */
Eigen::Matrix3d MathUtil::ypr2R(const Eigen::Vector3d &ypr) {
    double y = ypr(0);
    double p = ypr(1);
    double r = ypr(2);

    Eigen::Matrix3d Rz;
    Rz << cos(y), -sin(y), 0,
            sin(y), cos(y), 0,
            0, 0, 1;

    Eigen::Matrix3d Ry;
    Ry << cos(p), 0., sin(p),
            0., 1., 0.,
            -sin(p), 0., cos(p);

    Eigen::Matrix3d Rx;
    Rx << 1., 0., 0.,
            0., cos(r), -sin(r),
            0., sin(r), cos(r);

    return Rz * Ry * Rx;
}


Eigen::Matrix<double, 3, 4> MathUtil::PoseMultiply(const Matrix<double, 3, 4> &pose1,
                                                   const Matrix<double, 3, 4> &pose2) {
    Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
    T2.topRows(3) = pose2;
    return pose1 * T2;
}


double MathUtil::Angdiff(double a, double b) {
    double c = fmod(a-b, CV_PI*2.0);
    if (c > CV_PI) {
        c -= 2*CV_PI;
    } else if (c < -CV_PI) {
        c += 2*CV_PI;
    }
    return c;
}


Eigen::Quaterniond MathUtil::QuatInterpolate(const Eigen::Quaterniond &qa,
                                                                    const Eigen::Quaterniond &qb,
                                                                    double ratio) {
    return qa.slerp(ratio, qb);
}


Eigen::Matrix4d MathUtil::PoseInterpolate(const Eigen::Matrix4d &Ta,
                                                              const Eigen::Matrix4d &Tb,
                                                              double ratio) {
    // Interpolate quaternion
    Matrix3d Ra = Ta.topLeftCorner<3, 3>();
    Matrix3d Rb = Tb.topLeftCorner<3, 3>();
    Quaterniond qa(Ra);
    Quaterniond qb(Rb);
    Quaterniond q = qa.slerp(ratio, qb);

    // Result pose
    Matrix4d T = Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = q.matrix();
    T.topRightCorner<3, 1>() = (1 - ratio) * Ta.topRightCorner<3, 1>() + ratio * Tb.topRightCorner<3, 1>();
    return T;
}


Quaterniond MathUtil::VecToQuat(const Vector3d &v)
{
    Quaterniond q;

    double normv = v.norm();
    if (normv < 1e-7) {
        q.setIdentity();
    } else {
        q.w() = cos(0.5*normv);
        q.vec() = (sin(0.5*normv)/normv) * v;
    }

    return q;
}



Matrix3d MathUtil::VecToSkew(const Vector3d &v)
{
    Matrix3d m;
    m <<   0,    -v(2),     v(1),
        v(2),      0,     -v(0),
        -v(1),    v(0),       0;

    return m;
}


void MathUtil::IncreQuat(const Vector4d &q, const Vector3d &v, Vector4d& pq)
{
    // convert v to a quaternion
    Vector4d p;

    double normv = v.norm();
    if (normv < 1e-7) {
        p << 1, 0, 0, 0;
    } else {
        p << cos(0.5 * normv), (sin(0.5*normv)/normv)*v;
    }

    pq  <<  p(0)*q(0) - p(1)*q(1) - p(2)*q(2) - p(3)*q(3),
        p(0)*q(1) + p(1)*q(0) + p(2)*q(3) - p(3)*q(2),
        p(0)*q(2) + p(2)*q(0) + p(3)*q(1) - p(1)*q(3),
        p(0)*q(3) + p(3)*q(0) + p(1)*q(2) - p(2)*q(1);

    pq.normalize();

}


void MathUtil::ReduceEigenVector(Eigen::VectorXd &x, const Eigen::VectorXi &isbad)
{
    assert(x.size() == isbad.size());

    int N = (int)x.size();
    int k = 0;
    for (int i=0; i<N; i++)
        if (!isbad[i])
            x[k++] = x[i];

    x.conservativeResize(k);

}



void MathUtil::ReduceEigenMatrix(Eigen::MatrixXd &m, const Eigen::VectorXi &isbad)
{
    assert(m.cols() == isbad.size());

    int N = (int)m.cols();
    int k = 0;
    for (int i=0; i<N; i++)
        if (!isbad[i])
            m.row(k++) = m.row(i);
    m.conservativeResize(k, N);

    k = 0;
    for (int i=0; i<N; i++)
        if (!isbad[i])
            m.col(k++) = m.col(i);
    m.conservativeResize(k, k);

}


Quaterniond MathUtil::GetQfromA(const Eigen::Vector3d &a_corrected)
{
    // Normalize
    Vector3d a = a_corrected;
    a.normalize();

    // Euler angle
    double psi = 0;
    double theta = -asin(a(0));
    double phi = atan2(a(1), a(2));

    // Compute quaternion
    Matrix3d Rws;
    Rws = AngleAxisd(psi, Vector3d::UnitZ()) * AngleAxisd(theta, Vector3d::UnitY()) * AngleAxisd(phi, Vector3d::UnitX());
    Quaterniond qsw(Rws);

    return qsw;
}

Matrix3d MathUtil::NormalizationJacobian(const Vector3d &v) {
    double x1 = v(0);
    double x2 = v(1);
    double x3 = v(2);
    double norm = v.norm();
    Matrix3d norm_jaco;
    norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
        - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
        - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
    return norm_jaco;
}


Matrix<double, 1, 3> MathUtil::InverseNormJacobian(const Eigen::Vector3d &v) {
    double norm = v.norm();
    Matrix<double, 1, 3> jaco = -1 / pow(norm, 3) * v.transpose();
    return jaco;
}


void MathUtil::CalcTangentBase(const Vector3d &ray, Matrix<double, 2, 3> &tangentBase) {
    Eigen::Vector3d tmp(0, 0, 1);
    if (ray == tmp) {
        tmp << 1, 0, 0;
    }
    Eigen::Vector3d b1 = (tmp - ray * (ray.transpose() * tmp)).normalized();
    Eigen::Vector3d b2 = ray.cross(b1);
    tangentBase.row(0) = b1.transpose();
    tangentBase.row(1) = b2.transpose();
}


bool MathUtil::LineFittingRansac(const std::vector<double> &xs,
                                 const std::vector<double> &ys,
                                 double distThresh,
                                 Eigen::Vector3d &coeff) {
    const int N = (int)xs.size();
    if (ys.size() != N) {
        //spdlog::error("the size of xs & ys are not the same!");
        return false;
    }
    if (N < 2) {
        //spdlog::error("less than two points, can't fit line!");
        return false;
    }


    // random number generator
    std::random_device randomDevice;
    std::default_random_engine randomEngine(randomDevice());
    std::uniform_int_distribution<int> uniDistribution(0, N-1);

    int maxInliers = 0;
    Eigen::Vector3d bestCoeff;
    int maxIter = 500;
    int iter = 0;
    while (iter < maxIter) {
        iter++;

        // choose two random point
        int k1 = uniDistribution(randomEngine);
        int k2 = uniDistribution(randomEngine);
        while (k2 == k1) {
            k2 = uniDistribution(randomEngine);
        }

        // line: ax + by + c = 0
        double x1 = xs.at(k1);
        double x2 = xs.at(k2);
        double y1 = ys.at(k1);
        double y2 = ys.at(k2);
        double a = y2 - y1;
        double b = x1 - x2;
        double c = -x1 * a - y1 * b;
        double iSqrtAb = 1.0 / std::hypot(a, b);

        // count supporters
        int inliers = 0;
        for (int i = 0; i < N; ++i) {
            double dist = fabs(a * xs[i] + b * ys[i] + c) * iSqrtAb;
            if (dist < distThresh) {
                inliers++;
            }
        }
        if (inliers > maxInliers) {
            bestCoeff << a, b, c;
            maxInliers = inliers;
            double inlierRatio = static_cast<double>(inliers) / static_cast<double>(N);
            maxIter = static_cast<int>(std::ceil(std::log(1.0 - 0.999) / std::log(1.0 - inlierRatio * inlierRatio)));
        }
    }

    coeff = bestCoeff;

    return true;
}


bool MathUtil::CurveFitting(const std::vector<double> &xs,
                               const std::vector<double> &ys,
                               int rank,
                               Eigen::VectorXd &coeff) {
    const int N = static_cast<int>(xs.size());
    if (ys.size() != N) {
        //spdlog::error("the size of xs & ys are not the same!");
        return false;
    }
    if (N < rank) {
        //spdlog::error("number of points is not enough to fit curve: {} < {}", N, rank);
    }

    // coeff[0] + coeff[1] * x + ... + coeff[K-1] * x^(K-1) = y
    Eigen::MatrixXd A(N, rank);
    Eigen::VectorXd B(N);
    for (int i = 0; i < N; i++) {
        const double x = xs[i];
        A(i, 0) = 1;
        for (int j = 1; j < rank; j++) {
            A(i, j) = A(i, j-1) * x;
        }
        B(i) = ys[i];
    }

    coeff = A.colPivHouseholderQr().solve(B);
    return true;
}


bool MathUtil::SolvePolynomialEquation(const std::vector<double> &coeffs,
                                       double y, double &x, int maxIterations, double eps) {
    double x0 = x;
    for (int iter = 0; iter < maxIterations; iter++) {
        // compute y(x) and J = y'(x)
        double yi = 0;
        double Ji = 0;
        double xn = 1;
        int m = 0;
        for (const auto &c : coeffs) {
            m += 1;
            Ji += m * c * xn;
            xn *= x;
            yi += c * xn;
        }

        // compute delta x
        double dy = yi - y;
        if (fabs(dy) < eps) {
            return true;
        }
        x = x - dy / Ji;
    }

    x = x0;
    return false;
}


bool MathUtil::SolveMonotonicPolynomial(const std::vector<double> &coeffs, double y, double xmin, double xmax,
                                        double &x) {
    double xa = xmin, xb = xmax;
    int iter = 0;
    while (iter < 50) {
        x = (xa + xb) * 0.5;

        double f = 0;
        double xn = x;
        for (const auto c : coeffs) {
            f += c * xn;
            xn *= x;
        }

        if (fabs(f - y) < 0.001) {
            return true;
        }

        if (f < y) {
            xa = x;
        } else {
            xb = x;
        }

        iter++;
    }
    return false;
}

}//namespace inslam {