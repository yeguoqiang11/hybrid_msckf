//
//  EkfVio.hpp
//  DearVins

#pragma once

#include <set>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "Vio/Caimura.hpp"
#include "Vio/Insfeature.hpp"
#include "Vio/VioFeature.h"
#include "Vio/FeatureObservation.h"
#include "FeatureGrid.hpp"
#include "Utils/json.hpp"

namespace inslam {

// vec_in_A = R_A_B * (vec_in_B - p_B_A)
struct SensorPose {
    Eigen::Matrix3d Rcw, Rwc;  // rotation between camera and world frame
    Eigen::Matrix3d Riw, Rwi;  // rotation between IMU and world frame
    Eigen::Vector3d pwc;  // camera position in world frame
    Eigen::Vector3d pwi;  // IMU position in world frame
};

struct Group {
    int frameId;
    int count;      // num of features associated to this group
    int stateEntry;   // pose entry in state vector
    int errCovEntry;  // pose error entry in err covariance matrix
    SensorPose pose;
};

class EkfVio {
public:
    
    // Constructor (sling window size, in-state features size, featuree pool, camera)
    EkfVio(const nlohmann::json &config, std::shared_ptr<Caimura> cam0, std::shared_ptr<Caimura> cam1 = nullptr);

    // Initialize
    void Initialize(Eigen::Vector4d& q, Eigen::Vector3d& p, Eigen::Vector3d &v, int frame_id, double timestamp);
    
    // Run
    void Run(const std::vector<FeatureObservation> &featureMsgs,
             double t_img,
             int frame_id,
             const std::vector<Eigen::Matrix<double,7,1> > &vimu);
    
    // Query current sensor pose
    inline Eigen::Matrix<double,10,1> GetPoseVel() const {
        return state_.head(10);
    }

    // Get IMU state (t, q, p, v, bg, ba)
    inline Eigen::Matrix<double, 17, 1> GetImuState() const {
        Eigen::Matrix<double, 17, 1> state;
        state << currTimestamp_, state_.head(16);
        return state;
    }

    // For visualization
    std::vector<Eigen::Vector3d> QueryActiveFeatures();
    std::vector<Eigen::Matrix4d> QuerySlidingWindowPoses();
    void DrawDebugInfo(cv::Mat &img);
    
public:
    
    std::vector<Eigen::Vector3d> mvMapPoints;
    
private:
    
    // Calculate Nc
    void SetNoiseNc();

    /* Predict state & covariance with IMU data (from last frame to current frame).
     * Imu: timestamp, wx, wy, wz, ax, ay, az.
     */
    void Propagate(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu);

    // Update feature server
    void UpdateFeatureServer(const std::vector<FeatureObservation> &featureMsgs);
    
    // Track instate features
    void TrackInsfeatures(std::vector<size_t> &ok_indices);
    
    
    /* Calculate h, H & S for tracked features.
     * h: predicted position on current frame image. (or viewing ray if use uint sphere reprojection error).
     * H: measurement Jacobian
     * S: innovation matrix (S = H * P * H^t + R, where R is the measurement noise covariance matrix)
     */
    std::vector<size_t> PredictSHh(std::vector<size_t> &ok_indices);

    /* Calculate measurement Jacobian and residual for an out-of-state feature.
     * An out-of-state feature is a feature that hasn't been added to the state vector.
     * If oos feature is lost at current frame, and has enough observations, and can be
     * well triangulated, then we may use it to update the filter like the MSCKF does. */
    bool OosFeatureJacobian(size_t featureId, const std::vector<size_t> &frameIds,
                            Eigen::MatrixXd &Hoi, Eigen::VectorXd &roi);

    /* Calculate out-of-state features */
    int CalcOosFeatures(Eigen::MatrixXd &Ho, Eigen::VectorXd &ro);

    /* Mahalanobis distance test */
    bool GatingTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r);
    
    /* One point ransac with instantaneous/process testing.
     * Civera, Javier , et al. "1-Point RANSAC for EKF-based structure from motion."
     * It repeatedly choose 1 random in-state feature to update the filter, and count
     * inliers using reprojection errors(<2pixels). The best feature and its supporters
     * will be called "Low innovation inliers", and used to update the filter together.
     */
    std::vector<size_t> OnePointRansac(const std::vector<size_t> &ok_indices);
    
    // Count supporters (Instantaneous Reprojection Error Testing / Process Whiteness Testing)
    int CountVotes(const Eigen::VectorXd& x, const std::vector<size_t>& indices, std::vector<uchar>& status);
    
    // EKF update using the selected instate features
    bool EKFUpdate(const std::vector<size_t> &indices);

    bool HybridUpdate(const std::vector<size_t> &indices);

    void UpdateFilter(const Eigen::MatrixXd &H, const Eigen::VectorXd &r);
    
    /* Calculate new state vector using state_ and dx.
     * dx is the corrections. Notice that since quaternion error is represented as
     * axis angle, the dimensions of state_ and dx are different.
     */
    void CalcNewXhat(const Eigen::VectorXd &dx, Eigen::VectorXd &xnew);
    
    
    /* Remove bad instate features from state / covariance / instateFeatures,
     * meanwhile bad features will also be erased from featureServer.
     * And if a group has no associated instate features any more, it might be
     * removed from state/ covariance / groups.
     */
    void RemoveGFs(const std::vector<uchar> &flags);
    
    
    /* Add a group of new features into filter.
     * Group's pose and features' inverse depths will be added into state/ covariance.
     */
    int AddNewGFs();

    int AddNewInsfeatures();
    
    
    /* Infer the locations of group pose and features in the state vector
     * and error state vector.
     */
    void UpdateEntries();
    
    
    /* Calculate IMU & camera poses using the state vector*/
    void RefreshPoses();
    
    
    // Calculate sensor pose from qp (quaternion, position)
    void QpToPose(const Eigen::Matrix<double,7,1> &qp, SensorPose &sp);
    
    // Separate inliers from outliers
    void SeparateVector(const std::vector<size_t> &v, const std::vector<uchar> &flags, std::vector<size_t> &vi, std::vector<size_t> &vo);
    
    // Calculate 3D position for insfeature i
    bool Calc3DInsfea(const Insfeature& insfeat, Eigen::Vector3d &xw);

private:

    /* State and Error state covariance.
     * state: [heading IMU state, sliding window of IMU poses, inverse depths of features]
     * heading IMU state: [current IMU orientation, current IMU position, current IMU velocity, gyro bias, acc bias]
     * sliding window : [q_I0_W, p_W_I0, q_I1_W, p_W_I1, ... ] (W : world, I: IMU)
     * features: [invDepth_0, invDepth_1, ...], the inverse depth is represented in anchor frame
     */
    Eigen::VectorXd state_;
    Eigen::MatrixXd errCov_;
    
    
    // In-state features / groups
    std::vector<Insfeature> instateFeatures_;
    std::map<size_t, Group> groups_;

    // Sliding window size & instate features size
    int slidingWindowSize_;
    int maxNumInstateFeatures_;

    // Use unit sphere reprojection error or image plane reprojection error
    bool useUnitSphereError_ = true;

    bool compressStereoError_ = true;

    // Use one point RANSAC or not
    bool useOnePointRansac_;

    // Use out-of-state features like MSCKF ?
    bool useOosFeatures_;

    // Camera & Imu
    std::shared_ptr<Caimura> cam_;
    std::shared_ptr<Caimura> cam1_;
    Eigen::Matrix3d Rrl_;
    Eigen::Vector3d prl_;

    // Some constants
    Eigen::Vector3d gravity_;
    Eigen::Matrix<double, 15, 15> noiseNc_;
    double featureMeasurementNoise_;

    // Current FrameID
    int currFrameId_;
    double currTimestamp_;

    // Current pose in world frame
    SensorPose posel_;

    // Feature manager
    std::map<size_t, VioFeature> featureServer_;

    // Uniform distribute insFeature
    std::shared_ptr<FeatureGrid> featureGrid_;

    // Statistics
    int numTrackedInsfeatures_ = 0;
    int numCompatibleInsfeatures_ = 0;
    int numInlierInsfeatures_ = 0;
    int numOosFeatures_ = 0;
    int numNewInsfeatures_ = 0;

};

}//namespace inslam {