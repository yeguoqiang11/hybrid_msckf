#ifndef HYBRIDSLAM_H
#define HYBRIDSLAM_H
#include <Eigen/Dense>
#include <Eigen/Householder>
#include <iostream>
#include <map>
#include <unordered_map>

#include "Utils/json.hpp"
#include "Utils/MathUtil.h"
#include "Vio/Caimura.hpp"
#include "Vio/FeatureObservation.h"
#include "Vio/Insfeature.hpp"
#include "Vio/VioFeature.h"

namespace featslam{
struct SensorState {
    Eigen::Matrix3d Rwi, Rwi_null;
    Eigen::Vector3d tiw, tiw_null;
    Eigen::Vector3d Vw, Vw_null;
    Eigen::Vector3d gyro; // angular velocity
    int frame_id;
    int state_id;
    int cov_id;
    std::unordered_map<size_t, double> obs_depth_inv;
    std::unordered_map<size_t, Eigen::Vector3d> obs0, obs1;
    std::unordered_map<size_t, Eigen::Vector2d> pt0s, pt1s;
};

struct ImuType {
    Eigen::Vector4d Qwi, Qwi_null; // JPL quaternion world to imu coordinate
    Eigen::Matrix3d Rwi, Rwi_null; // rotation from world to imu coordinate
    Eigen::Vector3d tiw, tiw_null; // translation from imu to world coordinate
    Eigen::Vector3d Vw, Vw_null;  // velocity in world
    Eigen::Vector3d gyro; // instant angular velocity
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    double td = 0;
    double tr = 0;
    double timestamp = -1.;
    int frame_id;
    std::unordered_map<size_t, Eigen::Vector3d> obs0, obs1;
    std::unordered_map<size_t, Eigen::Vector2d> pt0s, pt1s;
};

struct FeatureType {
  int cov_id;
  int frame_id;
  int host_id;
  double depth_inv = -1.0;
  Eigen::Vector3d ray;
  Eigen::Matrix<double, 2, 3> tangentplane;
};


// hybrid slam based on li yangming phd thesis and inverse depth parameterization
class HybridSlam {
  public:
    HybridSlam(const nlohmann::json &config, std::shared_ptr<Caimura> cam0, std::shared_ptr<Caimura> cam1 = nullptr);
    void Initialize(Eigen::Vector4d& q, Eigen::Vector3d& p, Eigen::Vector3d &v, int frame_id);
    void GridPoints();
    void Run(const std::vector<FeatureObservation> &featureMsgs,
             double t_img,
             int frame_id,
             const std::vector<Eigen::Matrix<double,7,1> > &vimu);
    void Propagate(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu);
    void RK4Propagate(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu);
    void MedianPropagate(const std::vector<Eigen::Matrix<double, 7, 1>> &vimu);
    void OCMedianPropagate(const std::vector<Eigen::Matrix<double, 7, 1>> &vimu);
    void FEJMedianPropagate(const std::vector<Eigen::Matrix<double, 7, 1>> &vimu);
    void PredictNewState(Eigen::Vector3d &acc, Eigen::Vector3d &w, double dt);
    void MapUpdate(const std::vector<FeatureObservation> &featureMsgs);
    void FrameStateAugment();
    bool IsKeyFrame();
    void CovarianceReduceUpdate(int idx, int len);
    void FeatureAugment();
    bool TryTriangulation(const std::vector<int> &seen_ids, size_t pt_id, int host_id, double &rho);
    bool Triangulation(const std::vector<int> &seen_ids, size_t pt_id, int host_id, double &rho_out);
    void FeatureStateAugment(const std::vector<int> &seen_ids, size_t pt_id, int host_id, const double &rho);
    void FeatureJacobian(const Eigen::Vector3d &ray0, int host_id, double rho0, Eigen::Vector3d &pf1,
                         Eigen::Matrix<double, 3, 6> &dpfg_dx1, Eigen::Matrix<double, 3, 6> &dpfg_dx0,
                         Eigen::Matrix3d &drayg_dpfg, Eigen::Vector3d &dpfg_drho0, int guest_id, bool is_stereo = false);
    void NewFeatureJacobians(size_t pt_id, const std::vector<int> &seen_ids, Eigen::VectorXd &r, Eigen::MatrixXd &H);
    void TwoFrameJacobian(const Eigen::Vector3d &ray0, const double &rho0, const Eigen::Vector3d &ray1,
                          const Eigen::Matrix3d &Rwi0, const Eigen::Vector3d &tiw0, const Eigen::Matrix3d &Rwi1,
                          const Eigen::Vector3d &tiw1, Eigen::MatrixXd &Hx0, Eigen::MatrixXd &Hx1, Eigen::MatrixXd &Hf,
                          Eigen::VectorXd &r, bool is_stereo = false);
    void TwoFrameJacobian(const Eigen::Vector3d &ray0, const double &rho0, const Eigen::Vector3d &ray1,
                          const Eigen::Matrix3d &Rwi0, const Eigen::Vector3d &tiw0, const Eigen::Matrix3d &Rwi1,
                          const Eigen::Vector3d &tiw1, Eigen::MatrixXd &Hx0, Eigen::MatrixXd &Hx1, Eigen::MatrixXd &Hf,
                          Eigen::MatrixXd &Hic, Eigen::MatrixXd &Htd, Eigen::VectorXd &r, bool is_stereo = false);
    void TwoFrameJacobian(int frame_id0, int frame_id1, size_t pt_id, const Eigen::Vector3d &ray0, const double &rho0,
                          Eigen::MatrixXd &Hx0, Eigen::MatrixXd &Hx1, Eigen::MatrixXd &Hf, Eigen::MatrixXd &Hic,
                          Eigen::MatrixXd &Htd, Eigen::MatrixXd &Htr, Eigen::VectorXd &r, bool is_stereo = false);
    void OCTwoFrameJacobian(const Eigen::Vector3d &ray0, const double &rho0, const Eigen::Vector3d &ray1,
                            const Eigen::Matrix3d &Rwi0, const Eigen::Vector3d &tiw0, const Eigen::Matrix3d &Rwi1,
                            const Eigen::Vector3d &tiw1, Eigen::MatrixXd &Hx0, Eigen::MatrixXd &Hx1, Eigen::MatrixXd &Hf,
                            Eigen::VectorXd &r, bool is_stereo = false);
    void StereoJacobian(const Eigen::Vector3d &ray0, const double &rho0, const Eigen::Vector3d &ray1,
                        Eigen::MatrixXd &Hf, Eigen::VectorXd &r);

    void MeasurementUpdate(const Eigen::MatrixXd &H, const Eigen::VectorXd &r);

    void NewObsFeatureUpdate();
    void NewObsFeatureJacobian(const size_t &pt_id, Eigen::MatrixXd &J, Eigen::VectorXd &r);
    void LostFeaturesUpdate();
    void RemoveOutlier(size_t pt_id);
    void LostFeatureJacobian(const std::vector<int> &seenids, size_t pt_id, Eigen::MatrixXd &J, Eigen::VectorXd &r);

    // for test
    void JacobiansTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r);    
    void UpdateTest(Eigen::MatrixXd &H, Eigen::VectorXd &r);

    // Output current imu pose
    Eigen::Vector3d ImuPose() { return t_imu_.tiw; }
    Eigen::Vector4d ImuOrieation() { return t_imu_.Qwi; }
    double ImageDelayTime() { return t_imu_.td; } // image_t = input_image_time + td;
    std::vector<Eigen::Matrix4d> SlidingWindowPose();
    void DrawDebugInformation(cv::Mat &vimg);
    void DrawBias();

    std::vector<Eigen::Vector3d> MapPoints();
    std::vector<Eigen::Vector3d>& AllMapPoints();

    bool GatingTest(Eigen::MatrixXd &H, Eigen::VectorXd &r, int cam_num);

    // Based on Li mingyang error definition, rotation error defined at left
    void GlobalPropagate(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu);


  private:
    Eigen::MatrixXd P_; // [dtheta, dp, dv, dbg, dba]
    ImuType t_imu_;

    int slidingwindow_num_;
    int state_features_num_;
    bool use_unitsphere_error_;
    bool use_extrinsic_calib_;
    bool use_td_calib_;
    bool use_rolling_shutter_calib_;
    bool use_FEJ_;
    int features_obs_num_thres_;
    int triangulation_max_iteration_;
    int least_landmark_num_thres_;
    int max_landmark_num_thres_;
    int imu_state_dim_;
    size_t extrinsic_cov_id_;
    size_t td_cov_id_;
    size_t rs_cov_id_;

    int img_width0_, img_width1_, img_height0_, img_height1_;
    int cx0_, cy0_, cx1_, cy1_;

    double triangulation_huber_thres_;
    double reproj_err_thres_;
    double max_depth_inv_;
    double min_depth_inv_;

    Eigen::Vector3d gravity_;

    std::shared_ptr<Caimura> cam0_;
    std::shared_ptr<Caimura> cam1_;

    double features_noise_;

    Eigen::Matrix3d Rclcr_, Ric_;
    Eigen::Vector3d tclcr_, tci_;

    // std::vector<Insfeature> feature_states_;
    std::vector<SensorState> frame_states_;
    std::map<size_t, FeatureType> feature_states_;

    Eigen::MatrixXd imu_noise_; // [ng, na, nwg, nwa]

    std::unordered_map<size_t, VioFeature> map_;

    std::unordered_map<size_t, Eigen::Vector3d> totalMapPoints_;
    std::vector<Eigen::Vector3d> allMapPoints_;

    double keyframe_rot_thres_, keyframe_translation_thres_;

    // test variaty
    std::vector<Eigen::Vector3d> acc_bias_, gyro_bias_;
    std::vector<double> td_vec_, tr_vec_;
};
} // namespace inslam
#endif // HYBRIDSLAM_H