
#include "Vio/Triangulator.h"
#include "Vio/Factor/ReproErrorX.h"
#include "Vio/Factor/SphereReproErrorX.h"

using namespace std;
using namespace Eigen;

namespace inslam {

// -------------------------------------------------------------------------------------
/* Compute n-view triangulation by an efficient L2 minimization of the algebraic error.
 * This minimization is independent of the number of points, which makes it extremely scalable.
 * It gives better reprojection errors in the results and is significantly faster.
 * THIS PIECE OF CODE IS BASED ON THE TriangulateNView() FUNCTION IMPLEMENTED IN
 * TheiaSfM: https://github.com/sweeneychris/TheiaSfM
 */
bool Triangulator::Solve(const vector<Vector3d> &rays,
                         const vector<Matrix<double, 3, 4> > &poses,
                         double angularResolution,
                         Eigen::Vector3d &X,
                         bool checkParallax) {
    const size_t N = rays.size();
    if (N < 2)  return false;

    // rays in world frame
    Matrix4d A = Matrix4d::Zero();
    for (size_t i=0; i<N; i++) {
        const Vector3d &ray = rays[i];
        const Matrix<double,3,4> cost_term = (Matrix3d::Identity() - ray * ray.transpose()) * poses[i];
        A += cost_term.transpose() * cost_term;
    }
    SelfAdjointEigenSolver<Matrix4d> eigen_solver(A);
    Vector4d pt4D = eigen_solver.eigenvectors().col(0);
    if (eigen_solver.info() != Eigen::Success  || fabs(pt4D(3)) <= std::numeric_limits<double>::min()) {
        return false;
    }
    X = pt4D.head(3) / pt4D(3);

    //check depth and reprojection errors
    const double angularErrorThresh = 8 * angularResolution;
    for (size_t i = 0; i < rays.size(); ++i) {
        Vector3d Xc = poses[i].topLeftCorner(3, 3) * X + poses[i].topRightCorner(3, 1);
        Vector3d rayc = Xc.normalized();
        double err = (rayc - rays[i]).norm(); // or use fabs(std::acos(rayc.dot(rays[i]) ) );
        if (err > angularErrorThresh) {
            return false;
        }
    }

    if (!checkParallax) {
        return true;
    }

    // Check parallax
    return CheckParallax(rays, poses);

}


//TODO: numerical simulation check
bool Triangulator::Optimize(const vector<Vector3d> &rays,
                         const vector<Matrix<double, 3, 4>> &poses,
                         double angularResolution,
                         Vector3d &X) {
    double xw[3] = {X(0), X(1), X(2)};

    double sqrtInfo = 1.0 / angularResolution;

    // Build problem
    ceres::Problem problem;
    for (size_t i = 0; i < rays.size(); ++i) {
        ceres::CostFunction* costFunc = SphereReproErrorX::Create(rays[i], poses[i], sqrtInfo);
        ceres::LossFunction* lossFunc = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(costFunc, lossFunc, xw);
    }

    // Run the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    X << xw[0], xw[1], xw[2];

    return summary.IsSolutionUsable();
}


bool Triangulator::CheckParallax(const vector<Vector3d> &rays,
                                 const vector<Matrix<double, 3, 4> > &poses,
                                 double minimalAngle) {
    vector<Vector3d> worldRays;
    for (size_t i = 0; i < rays.size(); ++i) {
        Vector3d rayw = poses[i].topLeftCorner(3, 3) * rays[i];
        worldRays.push_back(rayw);
    }
    return CheckParallax(worldRays, minimalAngle);
}


bool Triangulator::CheckParallax(const vector<Eigen::Vector3d> &rays, double minimalAngle) {
    const int N = static_cast<int>(rays.size());
    if (N < 2) {
        return false;
    }

    const double cosValueThresh = std::cos(minimalAngle * 3.1415926 / 180.0);
    for (int i = 0; i < N-1; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (rays[i].dot(rays[j]) < cosValueThresh) {
                return true;
            }
        }
    }
    return false;
}

}//namespace inslam {