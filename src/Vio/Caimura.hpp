//
//  Caimura.hpp
//  DearVins
//

#ifndef Caimura_hpp
#define Caimura_hpp

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "Utils/json.hpp"

namespace inslam {

enum class CameraModelType {
    Perspective = 0,
    OpencvFisheye = 1,
    Equidistant = 2,
    Equirectangular = 3
};

class Caimura
{
public:

    explicit Caimura(const nlohmann::json &config);

    // Set scale
    void SetScale(double scale);

    // Set Imu-Camera extrinsics
    void SetICExtrinsics(const Eigen::Quaterniond &q_Sc, const Eigen::Vector3d &p_Cs);

    // Set IMU parameters
    void SetImuNoise(double gyroSigma, double gyroRW, double accSigma, double accRW);
    void SetImuRange(double gyroRange, double accRange);
    void SetImuBias(const Eigen::Vector3d &gyroBias, const Eigen::Vector3d &accBias);

    // Set inverse depth limitation
    void SetScenePriors(double rho, double rho_sigma, double rho_min, double rho_max);
    
    // Undistort points
    void UndistortPts(const std::vector<cv::Point2f>& vPts, std::vector<cv::Point2f>& vPtsUn);
    cv::Point2f UndistortPoint(const cv::Point2f &pt);

    // Distort points
    void DistortPts(const std::vector<cv::Point2f>& vPtsUn, std::vector<cv::Point2f> &vPts);
    cv::Point2f DistortPoint(const cv::Point2f &pt);

    //
    Eigen::Vector3d LiftSphere(const Eigen::Vector2d &uv, bool applyUndistortion = false);

    // Project a 3D point in camera frame onto the image plane.
    void Reproject(const Eigen::Vector3d &xc, Eigen::Vector2d &uv, bool applyDistortion = true);

    void ReprojectJacobian(const Eigen::Vector3d &xc, Eigen::Vector2d &uv, Eigen::Matrix<double, 2, 3> &jocobian);

    // Project a 3D point to plane
    // jacobian is partial([u, v]) / partial([x, y, z]) or partial([u, v, fb/z]) / partial([x, y, z])
    bool PinholeProjection(const Eigen::Vector3d &xc, Eigen::Vector2d &uv,
                           Eigen::Matrix3d &jacobian, bool calcJacobian = true);

    Eigen::Matrix<double, 2, 3> ProjectionJacobian(const Eigen::Vector3d &xc);

    double DistortThetaOpencvFisheye(double theta);
    double DistortThetaEquidistant(double theta);
    double UndistortThetaEquidistant(double theta);

    // Set gyroscope bias
    void SetGyroBias(const Eigen::Vector3d &bg);

    // Print parameters
    void PrintParams();

    inline int id() const { return id_; }

    inline int width() const { return width_; }

    inline int height() const { return height_; }

    inline double fx() const { return fx_; }

    inline double fy() const { return fy_; }

    inline double cx() const { return cx_; }

    inline double cy() const { return cy_; }

    inline double ifx() const { return ifx_; }

    inline double ify() const { return ify_; }

    inline double fb() const { return fb_; }

    inline double radius() const { return radius_; }

    inline double fovAngle() const { return fovAngle_; }

    inline double GetAngularResolution() const { return angularResolution_; }

    inline cv::Mat GetCvK() const { return cvK_.clone(); }

    inline cv::Mat GetCvD() const { return cvD_.clone(); }

    inline Eigen::Matrix3d GetK() const { return K_; }

    inline std::vector<double> GetDistCoeffs() const {return distCoeffs_; }

    inline bool IsPerspective() const { return model_ == CameraModelType::Perspective; }

    inline bool IsOpencvFisheye() const { return model_ == CameraModelType::OpencvFisheye; }

    inline bool IsEquidistant() const { return model_ == CameraModelType::Equidistant; }

    inline bool IsEquirectangular() const { return model_ == CameraModelType::Equirectangular; }

private:
    CameraModelType StringToModel(const std::string &str);
    std::string ModelString();


    /* Calculate angular resolution.
     * Angular resolution is the magnitude of angle corresponding to one pixel in image plane.
     * For perspective or equirectangular camera, the angular resolution is different at
     * different image location, but we just treat it as constant for simplicity.
     */
    double CalcAngularResolution();

    // Info
    int id_;
    CameraModelType model_;

    // Image Size
    int width_;
    int height_;

    // Intrinsics
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    double ifx_;
    double ify_;
    double fovAngle_;    // in rad
    double tanHalfFov_;
    double radius_ = -1;

    double angularResolution_;

    Eigen::Matrix3d K_;
    cv::Mat cvK_;

    // Distortion parameters
    std::vector<double> distCoeffs_;
    cv::Mat cvD_;

    // Pseudo focal_length * baseline
    double fb_ = 20;

    // Camera extrinsics (c: camera, b: body)
    Eigen::Matrix4d T_c_b_;

public:
    Eigen::Vector3d pci_;
    Eigen::Vector3d pic_;
    Eigen::Quaterniond qci_;
    Eigen::Quaterniond qic_;
    Eigen::Matrix3d Rci_;
    Eigen::Matrix3d Ric_;
    Eigen::Matrix4d Tci_;
    Eigen::Matrix4d Tic_;
    
    
    // Imu parameters
    double gyroSigma_;
    double gyroRandomWalk_;
    double gyroRange_;
    double accSigma_;
    double accRandomWalk_;
    double accRange_;
    
    Eigen::Vector3d bg_;
    Eigen::Vector3d ba_;
    
    
    // Scene depth bound
    double minInverseDepth_;
    double maxInverseDepth_;
    double priorInverseDepth_;
    double priorInverseDepthSigma_;
    
    
};

}//namespace inslam {

#endif /* Caimura_hpp */
