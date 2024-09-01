
#include "Vio/VioFeature.h"
#include "Vio/Triangulator.h"
#include "Utils/MathUtil.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace hybrid_msckf {

VioFeature::VioFeature(size_t id): id_{id}, isInState_(false), initialized_(false) {}


void VioFeature::AddObservation(const FeatureObservation &obser) {
    observations_[obser.frameId] = obser;
    latestFrameId_ = obser.frameId;
}


bool VioFeature::CheckMotion(const map<size_t, Matrix<double, 3, 4> > &framePoses,
                             const shared_ptr<Caimura> &caim) {
    // For simplicity, we assume that the relative motion between the first frame and
    // the last frame is the largest motion.
    const size_t firstFrameId = observations_.begin()->first;
    const size_t lastFrameId = observations_.rbegin()->first;

    // The translation between the first frame and the last frame
    const Matrix<double, 3, 4>& firstPose = framePoses.at(firstFrameId);
    const Matrix<double, 3, 4>& lastPose = framePoses.at(lastFrameId);
    const Vector3d firstPosition = - firstPose.topLeftCorner(3, 3).transpose() * firstPose.topRightCorner(3, 1);
    const Vector3d lastPosition = -lastPose.topLeftCorner(3,3).transpose() * lastPose.topRightCorner(3, 1);
    const Vector3d translation = lastPosition - firstPosition;

    // Get the direction of the feature when it's firstly observed.
    Vector3d featureRayC = observations_.at(firstFrameId).ray0;
    Vector3d featureRayW = firstPose.topLeftCorner(3, 3).transpose() * featureRayC;

    Vector3d parallelTranslation = translation.dot(featureRayW) * featureRayW;
    Vector3d orthogonalTranslation = translation - parallelTranslation;
    return orthogonalTranslation.norm() > 0;
}


bool VioFeature::Triangulate(const map<size_t, Matrix<double, 3, 4>> &framePoses,
                             const shared_ptr<Caimura> &caim, const shared_ptr<Caimura> &cam1) {
    bool useStereo = false;
    Matrix<double, 3, 4> poseRl = Matrix<double, 3, 4>::Identity();
    if (cam1 != nullptr) {
        useStereo = true;
        poseRl.topLeftCorner(3, 3) = cam1->Rci_ * caim->Ric_;
        poseRl.topRightCorner(3, 1) = cam1->pci_ + cam1->Rci_ * caim->pic_;
    }

    // observations
    vector<size_t> ids;
    vector<Vector3d> rays;
    vector<Matrix<double, 3, 4> > poses;
    for (const auto &it : observations_) {
        if (framePoses.find(it.first) != framePoses.end()) {
            ids.push_back(it.first);
            rays.push_back(it.second.ray0);
            poses.push_back(framePoses.at(it.first));
            if (useStereo && it.second.isStereo) {
                rays.push_back(it.second.ray1);
                poses.emplace_back(MathUtil::PoseMultiply(poseRl, framePoses.at(it.first) ) );
            }
        }
    }

    if (rays.size() < 2) {
        return false;
    }

    // Anchor frame
    anchorFrameId_ = ids.back();
    const Matrix3d Raw = framePoses.at(anchorFrameId_).topLeftCorner(3, 3);
    const Vector3d paw = framePoses.at(anchorFrameId_).topRightCorner(3, 1);

    // [(Rac * xc)\times] xa = [(Rac * xc)\times] pac
    MatrixXd A = MatrixXd::Zero(2 * rays.size(), 3);
    MatrixXd b = MatrixXd::Zero(2 * rays.size(), 1);
    int row = 0;
    for (size_t i = 0; i < rays.size(); i++) {
        const Matrix3d Rcw = poses[i].topLeftCorner(3, 3);
        const Vector3d pcw = poses[i].topRightCorner(3, 1);
        Matrix3d Rac = Raw * Rcw.transpose();
        Vector3d pac = paw - Rac * pcw;
        Vector3d v = Rac * rays[i];
        Matrix<double, 2, 3> M;
        M << -v(2), 0, v(0), 0, v(2), -v(1);
        A.middleRows(row, 2) = M;
        b.middleRows(row, 2) = (M * pac);
        row += 2;
    }

    Vector3d xa = A.colPivHouseholderQr().solve(b);

    // Check A and xa
    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd singularValues;
    singularValues.resize(svd.singularValues().rows(), 1);
    singularValues = svd.singularValues();
    double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

    // Check condition number and depth
    if (fabs(condA) > 1000 || xa(2) < 0.25 || xa(2) > 40 || std::isnan(xa.norm())) {
        return false;
    }

    xw_ = Raw.transpose() * (xa - paw);
//    initialized_ = true;
    return true;
}


//TODO: support pano sphere camera
bool VioFeature::Refine(const map<size_t, Matrix<double, 3, 4>> &framePoses,
                        const shared_ptr<Caimura> &caim, const shared_ptr<Caimura> &cam1) {
    bool useStereo = false;
    Matrix<double, 3, 4> poseRl = Matrix<double, 3, 4>::Identity();
    if (cam1 != nullptr) {
        useStereo = true;
        poseRl.topLeftCorner(3, 3) = cam1->Rci_ * caim->Ric_;
        poseRl.topRightCorner(3, 1) = cam1->pci_ + cam1->Rci_ * caim->pic_;
    }

    // observations
    vector<size_t> ids;
    vector<Vector3d> rays;
    vector<Vector2d> points;
    vector<Matrix<double, 3, 4> > poses;
    for (const auto &it : observations_) {
        if (framePoses.find(it.first) != framePoses.end()) {
            ids.push_back(it.first);
            rays.push_back(it.second.ray0);
            points.emplace_back(it.second.ray0.head(2) / it.second.ray0(2));
            poses.push_back(framePoses.at(it.first));
            if (useStereo && it.second.isStereo) {
                rays.push_back(it.second.ray1);
                points.emplace_back(it.second.ray1.head(2) / it.second.ray1(2));
                poses.emplace_back(MathUtil::PoseMultiply(poseRl, framePoses.at(it.first) ) );
            }
        }
    }

    if (rays.size() < 2) {
        return false;
    }

    // Anchor pose
    size_t anchorFrameId = ids.back();
    const Matrix3d Raw = framePoses.at(anchorFrameId).topLeftCorner(3, 3);
    const Vector3d paw = framePoses.at(anchorFrameId).topRightCorner(3, 1);
    Vector3d xa = Raw * xw_ + paw;

    // Relative poses
    vector<Matrix<double, 3, 4> > relativePoses;
    for (size_t i = 0; i < poses.size(); i++) {
        const Matrix3d &Rcw = poses.at(i).topLeftCorner(3, 3);
        const Vector3d &pcw = poses.at(i).topRightCorner(3, 1);
        Matrix3d Rca = Rcw * Raw.transpose();
        Matrix<double, 3, 4> poseCa;
        poseCa.topLeftCorner(3, 3) = Rca;
        poseCa.topRightCorner(3, 1) = pcw - Rca * paw;
        relativePoses.push_back(poseCa);
    }

    // Init value
    double rho = 1 / xa(2);
    double alpha = xa(0) / xa(2);
    double beta = xa(1) / xa(2);

    // optimization parameters
    double lam = 1e-3;
    double eps = 10000;
    int runs = 0;

    // Variables used in the optimization
    bool recompute = true;
    Matrix3d Hess = Matrix3d::Zero();
    Vector3d grad = Vector3d::Zero();

    // Cost at the last iteration
    double cost_old = ComputeError(relativePoses, points, alpha, beta, rho);

    // Loop till we have either
    // 1. Reached max iteration count
    // 2. System is unstable
    // 3. System has converged
    while (runs < 20 && lam < 1e10 && eps > 1e-6) {
        // Recalculate the Jacobians/information/gradients
        if (recompute) {
            Hess.setZero();
            grad.setZero();

            double err = 0;

            // loop through observations
            for (size_t i = 0; i < relativePoses.size(); i++) {
                const Matrix3d &Rca = relativePoses[i].topLeftCorner(3, 3);
                const Vector3d &pca = relativePoses[i].topRightCorner(3, 1);

                // Middle variables
                double hi1 = Rca(0, 0) * alpha + Rca(0, 1) * beta + Rca(0, 2) + rho * pca(0, 0);
                double hi2 = Rca(1, 0) * alpha + Rca(1, 1) * beta + Rca(1, 2) + rho * pca(1, 0);
                double hi3 = Rca(2, 0) * alpha + Rca(2, 1) * beta + Rca(2, 2) + rho * pca(2, 0);
                // Calculate jacobian
                double d_z1_d_alpha = (Rca(0, 0) * hi3 - hi1 * Rca(2, 0)) / (pow(hi3, 2));
                double d_z1_d_beta = (Rca(0, 1) * hi3 - hi1 * Rca(2, 1)) / (pow(hi3, 2));
                double d_z1_d_rho = (pca(0, 0) * hi3 - hi1 * pca(2, 0)) / (pow(hi3, 2));
                double d_z2_d_alpha = (Rca(1, 0) * hi3 - hi2 * Rca(2, 0)) / (pow(hi3, 2));
                double d_z2_d_beta = (Rca(1, 1) * hi3 - hi2 * Rca(2, 1)) / (pow(hi3, 2));
                double d_z2_d_rho = (pca(1, 0) * hi3 - hi2 * pca(2, 0)) / (pow(hi3, 2));
                Eigen::Matrix<double, 2, 3> H;
                H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;

                // Calculate residual
                Vector2d z;
                z << hi1 / hi3, hi2 / hi3;
                Vector2d res = points[i] - z;

                // sum
                err += res.squaredNorm();
                grad.noalias() += H.transpose() * res;
                Hess.noalias() += H.transpose() * H;
            }
        }

        // Solve Levenberg iteration
        Matrix3d Hess_l = Hess;
        for (size_t r=0; r < (size_t)Hess.rows(); r++) {
            Hess_l(r,r) *= (1.0+lam);
        }
        Vector3d dx = Hess_l.colPivHouseholderQr().solve(grad);

        // Check error is descending?
        double cost = ComputeError(relativePoses, points, alpha + dx(0), beta + dx(1), rho + dx(2));

        // Converged?
        if (cost <= cost_old && (cost_old-cost)/cost_old < 1e-6) {
            alpha += dx(0);
            beta += dx(1);
            rho += dx(2);
            eps = 0;
            break;
        }

        // If cost is lowered, accept step
        // Otherwise use bigger lambda
        if (cost <= cost_old) {
            recompute = true;
            cost_old = cost;
            alpha += dx(0);
            beta += dx(1);
            rho += dx(2);
            runs++;
            lam = lam / 10;
            eps = dx.norm();
        } else {
            recompute = false;
            lam = lam * 10;
            continue;
        }
    }

    // Revert to standard, and set to all
    xa << alpha/rho, beta/rho, 1.0/rho;

    // Get tangent plane to x_hat
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(xa);
    Eigen::MatrixXd Q = qr.householderQ();

    // Max baseline we have between poses
    double base_line_max = 0.0;

    // Check maximum baseline
    // Loop through each camera for this feature
    for (size_t i = 0; i < relativePoses.size(); i++) {
        const Matrix3d &Rca = relativePoses[i].topLeftCorner(3, 3);
        const Vector3d &pca = relativePoses[i].topRightCorner(3, 1);
        Vector3d pac = -Rca.transpose() * pca;

        // Dot product camera pose and nullspace
        double base_line = ((Q.block(0,1,3,2)).transpose() * pac).norm();
        if (base_line > base_line_max) {
            base_line_max = base_line;
        }
    }

    // Check this feature
    if (xa(2) < 0.25 // too close
        || xa(2) > 40 // too far
        || (xa.norm() / base_line_max) > 40 // baseline ratio is too large
        || std::isnan(xa.norm()) ) // nan
    {
        return false;
    }

    // convert to position in world frame
    initialized_ = true;
    xw_ = Raw.transpose() * (xa - paw);
    return true;
}


double VioFeature::ComputeError(const vector<Matrix<double, 3, 4>> &poses,
                                const vector<Vector2d> &points, double alpha, double beta, double rho) {
    double err =  0;

    const Vector3d xa(alpha / rho, beta / rho, 1.0 / rho);
    for (size_t i = 0; i < poses.size(); ++i) {
        const Matrix<double, 3, 4> &poseCa = poses.at(i);
        Vector3d xc = poseCa.topLeftCorner(3, 3) * xa + poseCa.topRightCorner(3, 1);
        Vector2d res = points.at(i) - xc.head(2) / xc(2);
        err += res.squaredNorm();
    }

    return err;
}


bool VioFeature::InitializePosition(const map<size_t, Matrix<double, 3, 4>> &framePoses,
                                    const shared_ptr<Caimura> &caim,
                                    const shared_ptr<Caimura> &cam1,
                                    bool refine) {
    bool useStereo = false;
    Matrix<double, 3, 4> poseRl = Matrix<double, 3, 4>::Identity();
    if (cam1 != nullptr) {
        useStereo = true;
        poseRl.topLeftCorner(3, 3) = cam1->Rci_ * caim->Ric_;
        poseRl.topRightCorner(3, 1) = cam1->pci_ + cam1->Rci_ * caim->pic_;
    }

    // Observations
    vector<Vector3d> rays;
    vector<Matrix<double, 3, 4> > poses;
    for (const auto &it : observations_) {
        if (framePoses.find(it.first) != framePoses.end()) {
            rays.push_back(it.second.ray0);
            poses.push_back(framePoses.at(it.first));
            if (useStereo && it.second.isStereo) {
                rays.push_back(it.second.ray1);
                poses.emplace_back(MathUtil::PoseMultiply(poseRl, framePoses.at(it.first) ) );
            }
        }
    }
    if (rays.size() < 2) {
        return false;
    }

    // Triangulate
    double angularResolution = caim->GetAngularResolution();
    Vector3d xw;
    if (!Triangulator::Solve(rays, poses, angularResolution, xw, true)) {
        return false;
    }

    // Optimize the 3D position
    if (refine && !Triangulator::Optimize(rays, poses, angularResolution, xw)) {
        return false;
    }

    //TODO: Check depths for pinhole projection (perspective and opencv_fisheye) camera

    xw_ = xw;
    initialized_ = true;
    return true;
}


bool VioFeature::RefinePosition(const std::map<size_t, Eigen::Matrix<double, 3, 4>> &framePoses,
                                const std::shared_ptr<Caimura> &caim) {
    if (!initialized_) {
        cerr << "can't refine an un-initialized feature!" << endl;
        return false;
    }

    // Observations
    vector<Vector3d> rays;
    vector<Matrix<double, 3, 4> > poses;
    for (const auto &it : observations_) {
        if (framePoses.find(it.first) != framePoses.end()) {
            rays.push_back(it.second.ray0);
            poses.push_back(framePoses.at(it.first));
        }
    }
    if (rays.size() < 2) {
        return false;
    }

    // Triangulate
    double angularResolution = caim->GetAngularResolution();
    Vector3d xw = xw_;

    // Optimize the 3D position
    if (!Triangulator::Optimize(rays, poses, angularResolution, xw)) {
        return false;
    }

    //TODO: Check depths for pinhole projection (perspective and opencv_fisheye) camera
    xw_ = xw;
    initialized_ = true;
    return true;
}


void VioFeature::SetAnchorFrame(size_t anchorFrameId, const std::shared_ptr<Caimura>& caim) {
    anchorFrameId_ = anchorFrameId;
    const auto &obser = observations_.at(anchorFrameId);
    anchorRay_ = obser.ray0;
}

}//namespace hybrid_msckf {