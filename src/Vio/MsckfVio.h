
#pragma once

#include <iostream>
#include <memory>
#include <set>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "Vio/Caimura.hpp"
#include "Vio/VioFeature.h"
#include "Vio/FeatureObservation.h"

namespace inslam {

class MsckfVio {
public:
    MsckfVio(int maxSlidingWindowSize,
             std::shared_ptr<Caimura> cam0,
             std::shared_ptr<Caimura> cam1 = nullptr);

    // Initialize
    void Initialize(Eigen::Vector4d& q, Eigen::Vector3d& p, Eigen::Vector3d &v, int frameId, double timestamp);

    // Run
    void Run(const std::vector<FeatureObservation> &featureMsgs,
             double t_img, int frameId,
             const std::vector<Eigen::Matrix<double,7,1> > &imuDatas);

    // Get IMU pose
    inline Eigen::Matrix<double, 10, 1> GetPoseVel() const {
        return state_.head(10);
    }

    // Get IMU state (t, q, p, v, bg, ba)
    inline Eigen::Matrix<double, 17, 1> GetImuState() const {
        Eigen::Matrix<double, 17, 1> state;
        state << currTimestamp_, state_.head(16);
        return state;
    }

    std::vector<Eigen::Matrix4d> QuerySlidingWindowPoses();

    std::vector<Eigen::Vector3d> lostMapPoints_;

private:
    /* Predict state & covariance with IMU data (from last frame to current frame).
     * Imu: timestamp, wx, wy, wz, ax, ay, az.
     */
    void Propagate(const std::vector<Eigen::Matrix<double, 7, 1> > &imuDatas);

    /* Update feature server */
    void UpdateFeatureServer(const std::vector<FeatureObservation> &featureMsgs);

    /* Remove lost features.
     * It will remove two kinds of features from the feature server:
     * 1. lost tracking at current frame, all of them will be erased from feature server. (qualified features will be used for MSCKF update)
     * 2. still tracked, but the tracked frames >= max frames:
     *     -Some of these features are qualified to be MSCKF features, they will be used to update the filter, then erased from the feature server.
     *     -Other features will remain in the feature server
     */
    void RemoveLostFeatures();

    /* Remove redundant IMU states from sliding window. */
    void PruneImuStates();

    /* Augment state.
     * Add current frame to the sliding window
     */
    void AugmentState();


    /* Msckf update. */
    void MsckfUpdate(const std::vector<size_t> &featureIds, const std::set<size_t> &involvedFrameIds);

    /* Calculate feature measurement Jacobian and residual */
     bool FeatureJacobian(size_t featureId, const std::vector<size_t> &frameIds,
                          Eigen::MatrixXd &Hoi, Eigen::VectorXd &roi);

     /* Mahanalobis gating test */
     bool GatingTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r);

     void FindRedundantFrames(std::vector<size_t> &redundantFrameIds);

    /* Calculate new state vector with the corrections */
    void CalcNewState(const Eigen::VectorXd &dx, Eigen::VectorXd &x);

    void UpdatePoseInfos(bool updateEntries = true);

    /* Collect camera poses from sliding PoseInfo window */
    void CollectCameraPoses(std::map<size_t, Eigen::Matrix<double, 3, 4> > &camPoses);

private:

    /* State and Error state covariance.
     * state: [heading IMU state, sliding window of IMU poses]
     * heading IMU state: [current IMU orientation, current IMU position, current IMU velocity, gyro bias, acc bias]
     * sliding window : [q_I0_W, p_W_I0, q_I1_W, p_W_I1, ... ] (W : world, I: IMU)
     */
    Eigen::VectorXd state_;
    Eigen::MatrixXd errCov_;

    /* Dimension of the heading IMU state
     * For now, it contains [imu position, imu orientation, imu velocity, gyro bias, acc bias],
     * we might add the camera-imu extrinsics later, so the dimension will change.
     */
    int headDim_ = 16; // (q: 4, p: 3, v: 3, bg: 3, ba: 3)
    int errHeadDim_ = 15;  // (q_err: 4, p_err: 3, v_err: 3, bg_err: 3, ba_err: 3)

    /* Sliding window of IMU poses. */
    struct PoseInfo {
        PoseInfo() {}
        PoseInfo(size_t frameId) : frameId_(frameId) {}
        size_t frameId_;
        int stateEntry_; // location in the state vector
        int errCovEntry_;    // location in the error state covariance
        Eigen::Matrix3d R_I_W_;
        Eigen::Vector3d p_W_I_;
        Eigen::Matrix3d R_C_W_;
        Eigen::Vector3d p_W_C_;
    };
    std::map<size_t, PoseInfo> framePoseInfos_;

    int maxSlidingWindowSize_;
    std::shared_ptr<Caimura> caim_;
    std::shared_ptr<Caimura> cam1_;
    Eigen::Matrix3d Rrl_;
    Eigen::Vector3d prl_;

    /* current frame info */
    size_t currFrameId_;
    double currTimestamp_;

    /* Feature manager */
    std::map<size_t, VioFeature> featureServer_;
    double trackingRate_;

    /* parameters */
    int leastObservationNum_ = 3;
    int maxTrackLength_ = 6;

    // image feature noise in pixel
    double imageFeatureNoise_ = 1.0;

    double translationThresh_ = 0.4; // (meter)
    double rotationThresh_ = 0.261799;  // (rad), 15 degree
    double trackingRateThresh_ = 0.5;

    /* Some constant values */
    Eigen::Vector3d gravity_;

    Eigen::MatrixXd noiseNc_;

};

}//namespace inslam {