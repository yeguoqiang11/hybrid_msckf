//
//  Caimura.cpp
//  DearVins
//

#include "Vio/Caimura.hpp"
#include "Utils/MathUtil.h"


using namespace cv;
using namespace std;
using namespace Eigen; 

namespace inslam {

Caimura::Caimura(const nlohmann::json &config) {
    id_ = config["id"];

    string modelStr = config["model_type"].get<string>();
    model_ = StringToModel(modelStr);

    width_ = config["width"];
    height_ = config["height"];

    if (!IsEquirectangular()) { // equirectangular camera has no intrinsics
        fx_ = config["fx"];
        fy_ = config["fy"];
        cx_ = config["cx"];
        cy_ = config["cy"];
        ifx_ = 1.0 / fx_;
        ify_ = 1.0 / fy_;
        cvK_ = (Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
        K_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;

        distCoeffs_.clear();
        for (size_t i = 0; i < config["dist_coeffs"].size(); ++i) {
            distCoeffs_.push_back(config["dist_coeffs"][i].get<double>());
        }
        cvD_ = cv::Mat(static_cast<int>(distCoeffs_.size()), 1, CV_64F, &distCoeffs_[0]);

        fovAngle_ = config["fov_angle"].get<double>() * CV_PI / 180.0;
        tanHalfFov_ = tan(fovAngle_ * 0.5);

        if (config.find("radius") != config.end()) {
            radius_ = config["radius"];
        }
    }
    angularResolution_ = CalcAngularResolution();

    if (config.find("scale") != config.end()) {
        double scale = config["scale"];
        SetScale(scale);
    }

    // focal length * stereo baseline
    if (config.find("focal_baseline") != config.end()) {
        fb_ = config["focal_baseline"];
    }

    // Imu-Camera extrinsics
    Matrix4d T_I_c;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T_I_c(i, j) = config["T_I_C"][i * 4 + j];
        }
    }
    Matrix3d R_I_c = T_I_c.topLeftCorner<3, 3>();
    Quaterniond q_I_c(R_I_c.transpose());
    Vector3d p_C_i = -R_I_c.transpose() * T_I_c.topRightCorner<3, 1>();

    SetICExtrinsics(q_I_c, p_C_i);

}


void Caimura::SetScale(double scale) {
    cout << "Setting camera scale: " << scale << endl;
    width_ = static_cast<int>(width_ * scale);
    height_ = static_cast<int>(height_ * scale);
    fx_ *= scale;
    fy_ *= scale;
    cx_ *= scale;
    cy_ *= scale;
    ifx_ = 1.0 / fx_;
    ify_ = 1.0 / fy_;
    radius_ *= scale;
    cvK_ = (Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
    K_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
    angularResolution_ = CalcAngularResolution();
}


void Caimura::SetICExtrinsics(const Eigen::Quaterniond &q_Sc, const Eigen::Vector3d &p_Cs)
{
    qic_ = q_Sc;
    
    qci_ = qic_.inverse();
    Rci_ = qic_.matrix();
    Ric_ = Rci_.transpose();
    
    pci_ = p_Cs;
    pic_ = -Ric_ * pci_;

    Tci_.setIdentity();
    Tci_.topRows(3) << Rci_, pci_;
    Tic_.setIdentity();
    Tic_.topRows(3) << Ric_, pic_;
}



void Caimura::SetImuNoise(double gyroSigma, double gyroRW, double accSigma, double accRW) {
    gyroSigma_ = gyroSigma;
    gyroRandomWalk_ = gyroRW;
    accSigma_ = accSigma;
    accRandomWalk_ = accRW;
}

void Caimura::SetImuRange(double gyroRange, double accRange) {
    gyroRange_ = gyroRange;
    accRange_ = accRange;
}

void Caimura::SetImuBias(const Eigen::Vector3d &gyroBias, const Eigen::Vector3d &accBias) {
    bg_ = gyroBias;
    ba_ = accBias;
}


void Caimura::SetScenePriors(double rho, double rho_sigma, double rho_min, double rho_max)
{
    priorInverseDepth_ = rho;
    priorInverseDepthSigma_ = rho_sigma;
    minInverseDepth_   = rho_min;
    maxInverseDepth_   = rho_max;
}



void Caimura::UndistortPts(const vector<Point2f> &vPts, vector<Point2f> &vPtsUn)
{
    vPtsUn = vPts;
    
    const size_t N = vPts.size();
    if (N < 1) {
        return;
    }

    if (distCoeffs_.empty()) {
        return;
    }

    for (size_t i = 0; i < vPts.size(); ++i) {
        vPtsUn[i] = UndistortPoint(vPts[i]);
    }

}

Point2f Caimura::UndistortPoint(const cv::Point2f &pt) {
    Point2f upt = pt;
    if (IsPerspective() || IsOpencvFisheye()) {
        Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = pt.x;
        mat.at<float>(0, 1) = pt.y;

        mat = mat.reshape(2);
        if (IsPerspective()) {
            cv::undistortPoints(mat, mat, cvK_, cvD_, Mat(), cvK_);
        } else {
            cv::fisheye::undistortPoints(mat, mat, cvK_, cvD_, Mat(), cvK_);
        }
        mat = mat.reshape(1);

        upt.x = mat.at<float>(0, 0);
        upt.y = mat.at<float>(0, 1);
    } else if (IsEquidistant()) {
        double un = (pt.x - cx_) * ifx_;
        double vn = (pt.y - cy_) * ify_;
        double r = std::hypot(un, vn);
        double theta = r;
        if (r == 0) {
            upt.x = static_cast<float>(cx_);
            upt.y = static_cast<float>(cy_);
        } else {
            double thetad = UndistortThetaEquidistant(theta);
            upt.x = static_cast<float>(fx_ * thetad * un / r + cx_);
            upt.y = static_cast<float>(fy_ * thetad * vn / r + cy_);
        }
    }
    return upt;
}

void Caimura::DistortPts(const vector<Point2f> &vPtsUn, vector<Point2f> &vPts) {
    vPts = vPtsUn;
    const size_t N = vPtsUn.size();
    if (N < 1) {
        return;
    }
    if (distCoeffs_.empty()) {
        return;
    }

    for (size_t i = 0; i < N; ++i) {
        cv::Point2f dpt = DistortPoint(vPtsUn[i]);
        vPts[i] = dpt;
    }
}


cv::Point2f Caimura::DistortPoint(const cv::Point2f &pt) {
    Point2f dpt = pt;
    if (IsEquirectangular()) {
        return dpt;
    }

    double x = (pt.x - cx_) * ifx_;
    double y = (pt.y - cy_) * ify_;
    double r = std::sqrt(x * x + y * y);
    double xd = x;
    double yd = y;

    if (IsPerspective()) {
        double k1 = distCoeffs_[0];
        double k2 = distCoeffs_[1];
        double p1 = distCoeffs_[2];
        double p2 = distCoeffs_[3];
        double k3 = distCoeffs_.size() > 4 ? distCoeffs_[4] : 0;

        double r2 = r * r;
        double r4 = r2 * r2;
        double coeff = 1 + k1 * r2 + k2 * r4 + k3 * r4 * r2;

        xd = x * coeff + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        yd = y * coeff + 2 * p2 * x * y + p1 * (r2 + 2 * y * y);
    } else if (IsOpencvFisheye()) {
        if (r == 0) {
            xd = x;
            yd = y;
        } else {
            double theta = atan(r);
            double thetad = DistortThetaOpencvFisheye(theta);
            xd = thetad * x / r;
            yd = thetad * y / r;
        }
    } else if (IsEquidistant()) {
        if (r == 0) {
            xd = x;
            yd = y;
        } else {
            double theta = r;
            double thetad = DistortThetaEquidistant(theta);
            xd = thetad * x / r;
            yd = thetad * y / r;
        }
    }

    dpt.x = static_cast<float>(fx_ * xd + cx_);
    dpt.y = static_cast<float>(fy_ * yd + cy_);
    return dpt;
}


Vector3d Caimura::LiftSphere(const Vector2d &uv, bool applyUndistortion) {
    Vector3d ray;
    if (IsPerspective() || IsOpencvFisheye()) {
        Point2f ptUn(static_cast<float>(uv(0)), static_cast<float>(uv(1)));
        if (applyUndistortion) {
            ptUn = UndistortPoint(Point2f(static_cast<float>(uv(0)), static_cast<float>(uv(1))) );
        }
        double x = (ptUn.x - cx_) * ifx_;
        double y = (ptUn.y - cy_) * ify_;
        ray = Vector3d(x, y, 1).normalized();
    }
    else if (IsEquidistant()) {
        double un = (uv(0) - cx_) * ifx_;
        double vn = (uv(1) - cy_) * ify_;
        double r = std::hypot(un, vn);
        double theta = r;
        if (applyUndistortion) {
            theta = UndistortThetaEquidistant(theta);
        }
        double phi = atan2(vn, un);
        ray << sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta);
    }
    else if (IsEquirectangular()) {
        // (Lef, Back, Up), consistent with Pano2FisheyeXL
        double theta = CV_PI * uv(1) / static_cast<double>(height_);
        double phi = CV_2PI - CV_2PI * uv(0) / static_cast<double>(width_);
        ray << sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta);
    }
    return ray;
}


void Caimura::Reproject(const Vector3d &xc, Vector2d &uv, bool applyDistortion) {
    double x = xc(0), y = xc(1), z = xc(2);

    if (IsPerspective() || IsOpencvFisheye()) {
        double iz = 1.0 / z;
        uv(0) = fx_ * x * iz + cx_;
        uv(1) = fy_ * y * iz + cy_;
    } else if (IsEquidistant()) {
        double rho = xc.head(2).norm();
        if (rho > 0) {
            double theta = atan2(rho, xc(2));
            double irho = 1.0 / rho;
            double r = theta;
            uv(0) = xc(0) * irho * r * fx_ + cx_;
            uv(1) = xc(1) * irho * r * fy_ + cy_;
        } else {
            uv << cx_, cy_;
        }

    } else if (IsEquirectangular()) {
        Vector3d v = xc.normalized();
        double theta = std::acos(v(2)); // [0, pi]
        double phi = std::atan2(v(1), v(0)); // (-pi, pi]
        if (phi < 0) {
            phi += CV_2PI;  // (0, 2*pi]
        }
        uv(0) = width_ * (1 - phi / CV_2PI);
        uv(1) = height_ * theta / CV_PI;
    }

    if (applyDistortion) {
        Point2f dpt = DistortPoint(Point2f(static_cast<float>(uv(0)), static_cast<float>(uv(1))) );
        uv(0) = dpt.x;
        uv(1) = dpt.y;
    }
}


void Caimura::ReprojectJacobian(const Eigen::Vector3d &xc, Eigen::Vector2d &uv,
                                Eigen::Matrix<double, 2, 3> &jocobian) {
    if (IsEquirectangular() || IsPerspective() || IsEquidistant()) {
        cerr << "reprojection jacobian not implemented for camera type: " << ModelString() <<endl;
        exit(-1);
    }

    double iz = 1.0 / xc(2);
    double un = xc(0) * iz;
    double vn = xc(1) * iz;
    Matrix<double, 2, 1> uv_norm(un, vn);

    double r = std::sqrt(un * un + vn * vn);
    double theta = std::atan(r);
    double thetad = DistortThetaOpencvFisheye(theta);

    double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
    double cdist = (r > 1e-8) ? thetad * inv_r : 1.0;

    uv << fx_ * cdist * un + cx_, fy_ * cdist * vn + cy_;

    Matrix2d duv_dxy;
    duv_dxy << fx_, 0, 0, fy_;

    Matrix2d dxy_dxyn = thetad * inv_r * Matrix2d::Identity();

    Matrix<double, 2, 1> dxy_dr = -thetad * inv_r * inv_r * uv_norm;

    Matrix<double, 1, 2> dr_dxyn = inv_r * uv_norm.transpose();

    Matrix<double, 2, 1> dxy_dthetad = inv_r * uv_norm;

    double theta2 = theta * theta;
    double theta4 = theta2 * theta2;
    double dthetad_dtheta = 1 + 3 * distCoeffs_[0] * theta2 + 5 * distCoeffs_[1] * theta4
                                      + 7 * distCoeffs_[2] * theta2 * theta4 + 9 * distCoeffs_[3] * theta4 * theta4;

    double dtheta_dr = 1 / (r * r + 1);

    Matrix2d duv_duvn = duv_dxy*(dxy_dxyn + (dxy_dr+dxy_dthetad*dthetad_dtheta*dtheta_dr)*dr_dxyn );

    Matrix<double, 2, 3> duvn_dxc;
    duvn_dxc << iz, 0, -xc(0) * iz*iz,
                      0, iz, -xc(1) * iz*iz;

    jocobian = duv_duvn * duvn_dxc;

}


bool Caimura::PinholeProjection(const Vector3d &xc, Vector2d &uv, Matrix3d &jacobian, bool calcJacobian) {
    if (!IsPerspective() && !IsOpencvFisheye()) {
        cerr << "Pinhole projection is not suitable for " << ModelString() << " camera" << endl;
        return false;
    }
    double iz = 1.0 / xc(2);
    if (iz < 0.1 || iz > 30) {
        return false;
    }

    uv(0) = fx_ * xc(0) * iz + cx_;
    uv(1) = fy_ * xc(1) * iz + cy_;

    // Jacobian = partial([u, v, fb/2]) / partial([x, y, z])
    if (calcJacobian) {
        double iz2 = iz * iz;
        jacobian << iz * fx_,  0,  -fx_ * xc(0) * iz2,
                        0,  iz * fy_,  -fy_ * xc(1) * iz2,
                        0,           0,            -fb_ * iz2;
    }

    return true;
}


Matrix<double, 2, 3> Caimura::ProjectionJacobian(const Eigen::Vector3d &xc) {
    Matrix<double, 2, 3> jacobian;
    if (!IsPerspective() && !IsOpencvFisheye()) {
        cerr << "Projection jacobian for " << ModelString() << " camera hasn't been implemented" << endl;
        jacobian.setIdentity();
    } else {
        double iz = 1.0 / xc(2);
        double iz2 = iz * iz;
        jacobian << iz * fx_, 0, -fx_ * xc(0) * iz2,
                        0, iz * fy_, -fy_ * xc(1) * iz2;
    }
    return jacobian;
}

// thetad = theta * (1 + c0 * theta^2 + c1 * theta^4 + c2 * theta^6 + c3 * theta^8)
double Caimura::DistortThetaOpencvFisheye(double theta) {
    double theta2 = theta * theta;
    double thetaN = theta;
    double res = theta;
    for (const auto &coeff : distCoeffs_) {
        thetaN *= theta2;
        res += coeff * thetaN;
    }
    return res;
}


// thetad = c0 * theta + c1 * theta^2 + c2 * theta^3 + c3 * theta^4
double Caimura::DistortThetaEquidistant(double theta) {
    if (distCoeffs_.empty()) {
        return theta;
    }

    double thetaN = 1.0;
    double res = 0;
    for (const auto &c : distCoeffs_) {
        thetaN *= theta;
        res += c * thetaN;
    }

    return res;
}

double Caimura::UndistortThetaEquidistant(double theta) {
    double res = theta;
//    if (!MathUtil::SolvePolynomialEquation(distCoeffs_, theta, res, 50, 3.0e-4)) {
//        cerr << "failed to undistort theta: " << theta << endl;
//    }

    const double halfMaxFov = fovAngle_ * 0.5;
    if (DistortThetaEquidistant(halfMaxFov) < theta) {
        cerr << "exceeds max fov: " << theta * 180 / CV_PI << std::endl;
        return halfMaxFov;
    }

    if (!MathUtil::SolveMonotonicPolynomial(distCoeffs_, theta, 0, halfMaxFov, res)) {
        cerr << "failed to undistort theta: " << theta << endl;
    }

    return res;
}


double Caimura::CalcAngularResolution() {
    double angularResolution = 0;
    if (IsPerspective() || IsOpencvFisheye()) {
        angularResolution = 1.0 / (1.5 * fx_);
    } else if (IsEquidistant()) {
        angularResolution = fovAngle_  / static_cast<double>(max(height_, width_));
    } else if (IsEquirectangular()) {
        angularResolution = CV_PI / static_cast<double>(height_);
    } else {
        cerr << "Unknown camera type: " << ModelString() << endl;
    }
    return angularResolution;
}


void Caimura::SetGyroBias(const Eigen::Vector3d &bg)
{
	bg_ = bg;
}

CameraModelType Caimura::StringToModel(const string &str) {
    if (str == "perspective") {
        return CameraModelType::Perspective;
    } else if (str == "opencv_fisheye") {
        return CameraModelType::OpencvFisheye;
    } else if (str == "equidistant") {
        return CameraModelType::Equidistant;
    } else if (str == "equirectangular") {
        return CameraModelType::Equirectangular;
    } else {
        cerr << "unknown model type string: " << str << endl;
        exit(-1);
    }
}

string Caimura::ModelString() {
    if (model_ == CameraModelType::Perspective) {
        return "perspective";
    } else if (model_ == CameraModelType::OpencvFisheye) {
        return "opencv_fisheye";
    } else if (model_ == CameraModelType::Equidistant) {
        return "equidistant";
    } else if (model_ == CameraModelType::Equirectangular) {
        return "equirectangular";
    } else {
        return "unknown";
    }
}

void Caimura::PrintParams() {
    cout << "config parameters: " << endl;
    cout << "\tfx: " << fx_ << endl;
    cout << "\tfy: " << fy_ << endl;
    cout << "\tcx: " << cx_ << endl;
    cout << "\tcy: " << cy_ << endl;
    cout << "\tfov: " << fovAngle_ * 180.0 / CV_PI << endl;
    cout << "\tangular resolution: " << angularResolution_ * 180 / CV_PI << endl;
    cout << "\tdistortion: " << cvD_.t() << endl;
    cout << "\tmodel type: " << ModelString() << endl;
    cout << "\tq_I_c: " << qic_.w() << ", " << qic_.vec().transpose() << endl;
    cout << "\tp_C_i: " << pci_.transpose() << endl;
    cout << "\timu noise: " << gyroSigma_ << ", " << gyroRandomWalk_ << ", " << accSigma_ << ", " << accRandomWalk_ << endl;
    cout << "\timu range: " << gyroRange_ << ", " << accRange_ << endl;
    cout << "\tgyro bias: " << bg_.transpose() << endl;
    cout << "\tacc bias: " << ba_.transpose() << endl;
    cout << "\tprior inverse depth: " << priorInverseDepth_ << endl;
    cout << "\tprior inverse depth sigma: " << priorInverseDepthSigma_ << endl;
    cout << "\tmin inverse depth: " << minInverseDepth_ << endl;
    cout << "\tmax inverse depth: " << maxInverseDepth_ << endl;
}

}//namespace inslam {
