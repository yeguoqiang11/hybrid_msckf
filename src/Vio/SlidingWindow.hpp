#ifndef SLIDINGWINDOW_H
#define SLIDINGWINDOW_H
#include <iostream>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <opencv2/opencv.hpp>

#include "Utils/json.hpp"
#include "Vio/Caimura.hpp"
#include "Vio/FeatureObservation.h"
#include "Vio/Initialize/Imu/CamCost.h"
#include "Vio/Initialize/Imu/Imu.h"
#include "Vio/Initialize/Imu/ImuCostFunction.h"
#include "Vio/Insfeature.hpp"
#include "Vio/SlidingWindowCost.hpp"
#include "Vio/VioFeature.h"

namespace inslam {
struct ImuObs {
    ImuObs(){ bg.setZero(); ba.setZero(); }
    ImuObs(double an, double gn, double arw, double grw, double g);
    void SetParam(double an, double gn, double arw, double grw, double g);
    void Initialize();
    void InitializeNext(const Eigen::Vector3d &t_ba, const Eigen::Vector3d &t_bg);
    void SetFrameId1(int frame_id) { frameId1 = frame_id; }
    void IntegrateImu(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro, double time);
    void MergeOldImu(ImuObs imu0);
    void SetBaBg(const Eigen::Vector3d &t_ba, const Eigen::Vector3d &t_bg) { ba = t_ba; bg = t_bg; }
    Preintegrated imu;
    double t0 = -1, t1 = -1;
    double Na, Ng, Nba, Nbg, gravity;
    Eigen::Vector3d ba, bg;
    int frameId0 = -1, frameId1 = -1;
};

struct LandMark {
    int pt_id = -1;
    int host_id = -1;
    double depth_inv = -1;
    std::vector<int> frame_ids; // seen in frame of frame_id
    bool is_good = false;
};

struct FrameType {
    void Initialize(double time, int frameId);
    double timestamp = -1;
    int frame_id = -1;
    Eigen::Matrix3d Riw;
    Eigen::Vector3d tiw;
    Eigen::Vector3d ba, bg, Vw;
    ImuObs imu;
    std::unordered_map<int, Eigen::Vector3d> obs0, obs1;
    std::unordered_map<int, cv::Point2f> pt0s, pt1s;
    std::vector<int> host_ptIds;
};

struct Camera {
    Camera() { Rc0cj.setIdentity(); tc0cj.setZero(); }
    int cam_id = -1;
    Eigen::Matrix3d Rc0cj;
    Eigen::Vector3d tc0cj;
    int width = -1, height = -1;
    double feature_noise_;
};

struct Imu {
    Imu() { ba.setZero(); bg.setZero();}
    Eigen::Matrix3d Ric0, Rc0i;
    Eigen::Vector3d tc0i, tic0;

    double Na, Ng, Nba, Nbg;
    Eigen::Vector3d gravity_; // acc_m = Rwi * (acc_t - gravity), acc_t = gravity + Riw * acc_m

    Eigen::MatrixXd noise_; // [Ng, Nbg, Na, Nba]
    Eigen::Vector3d ba, bg;
};

struct Parameter {
    Imu imu;
    std::vector<Camera> cams;
};

class SlidingWindow {
public:
    SlidingWindow(const nlohmann::json &config, std::shared_ptr<Caimura> cam0, std::shared_ptr<Caimura> cam1 = nullptr);
    void Run(const std::vector<FeatureObservation> &featureMsgs, int frame_id,
             const Eigen::Matrix<double, 17, 1> &frame_pose, // pose = [timestamp, orientation, position, vel, bg, ba]
             const std::vector<Eigen::Matrix<double,7,1> > &vimu); // imu = [gyro, acc]

    void Preintegration(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu, int frame_id);

    void FrameUpdate(const std::vector<FeatureObservation> &featureMsgs, int frame_id,
                     const Eigen::Matrix<double, 17, 1> &framePose);

    void WindowUpdate();

    void WindowOptimization();

    bool IsKeyFrame();

    bool CreateLandMark(int pt_id, int host_id);

    int FrameIdRetrieval(int frame_id) const;

    bool Triangulation(const std::vector<int> &frame_ids, int pt_id, int host_id, double &idepth, double &avg_err);

    void CreateKeyFrame();
    
    void RecoveryPose(double poses[][6]);

    void RecoveryVelBias(double vel_bias[][9]);

    void LandMarkChangeHostFrame(int pt_id, int rm_frameId);

    Eigen::Matrix<double, 10, 1> PoseVel();

    void StateIntegration(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu);

    double InitPoseOptimization();

    Eigen::Matrix<double, 3, 4> EPNP(const std::vector<Eigen::Vector3d> &ptcs, const std::vector<Eigen::Vector3d> &ptws);

    Eigen::Matrix<double, 3, 4> DLTPNP(const std::vector<Eigen::Vector3d> &ptcs, const std::vector<Eigen::Vector3d> &ptws);

    bool Initialization();

    Eigen::Matrix3d NisterEssentialMatrixSolver(const std::vector<Eigen::Vector3d> &ptcs,
                                                const std::vector<Eigen::Vector3d> &ptws);
    Eigen::Matrix3d DLTEssentialMatrixSolver(const std::vector<Eigen::Vector3d> &pt0s,
                                             const std::vector<Eigen::Vector3d> &pt1s, const std::vector<int> &ids);
    
    std::vector<int> EssentialInlierIndexs(const Eigen::Matrix3d &E, const std::vector<Eigen::Vector3d> &pt0s,
                             const std::vector<Eigen::Vector3d> &pt1s, double inlier_thres);

    std::vector<int> EssentialOutlierIndexs(const Eigen::Matrix3d &E, const std::vector<Eigen::Vector3d> &pt0s,
                             const std::vector<Eigen::Vector3d> &pt1s, double inlier_thres);

    Eigen::Matrix3d RansacEssentialMatrix(const std::vector<Eigen::Vector3d> &pt0s, const std::vector<Eigen::Vector3d> &pt1s,
                                          double confidence = 0.99, double thres = 0.002);
    
    void EssentialDecompose(const Eigen::Matrix3d &E, Eigen::Matrix3d &R1, Eigen::Matrix3d &R2, Eigen::Vector3d &t);

    // Simulation
    double TwoFrameReprojectionError(Eigen::Matrix3d Riwg, Eigen::Matrix3d Riwh, Eigen::Vector3d tiwg, Eigen::Vector3d tiwh,
                              Eigen::Vector3d obsg, Eigen::Vector3d  obsh, double rho, bool is_stereo = false);
    double TwoFrameImuError(ImuObs &imu, int frame_id0, int frame_id1, bool show = false) const;

    void Test();

    void TestDemo();

    void EssentialTest();

    void VoxelHashMapTest();

    bool PnPOptimization(const std::vector<Eigen::Vector3d> &Pws, std::vector<Eigen::Vector3d> &obses,
                         Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc);

    int CreateData(std::vector<Eigen::Vector3d> &Pws, std::vector<std::vector<Eigen::Vector3d>> &obs_v,
                    std::vector<Eigen::Matrix<double, 3, 4>> &Twcs);

    Eigen::VectorXd GaussianSolveNXNLinearEquations(const Eigen::MatrixXd &H, const Eigen::VectorXd &b);

    void AddNoiseAndOutliers(std::vector<Eigen::Vector3d> &obses);

    void LocalOptimization(std::vector<Eigen::Vector3d> &Pws, std::vector<std::vector<Eigen::Vector3d>> &obs_v,
                           std::vector<Eigen::Matrix<double, 3, 4>> &Twcs);

    double CalcCost(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &twc, const std::vector<Eigen::Vector3d> &Pws,
                    const std::vector<Eigen::Vector3d> &obses, std::vector<Eigen::Matrix<double, 2, 3>> &planes,
                    double huber_thres = 0.01, bool calc_plane = false);

    void HessianAndResidualCalc(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &twc, const std::vector<Eigen::Vector3d> &Pws,
                                const std::vector<Eigen::Vector3d> &obses, std::vector<Eigen::Matrix<double, 2, 3>> &planes,
                                Eigen::MatrixXd &H, Eigen::VectorXd &r, int pose_id, int pt_id0, double huber_thres = 0.01);

    void JacobAndResidualCalc(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &twc, const std::vector<Eigen::Vector3d> &Pws,
                                const std::vector<Eigen::Vector3d> &obses, std::vector<Eigen::Matrix<double, 2, 3>> &planes,
                                Eigen::MatrixXd &J, Eigen::VectorXd &r, int pose_id, int pt_id0, int &idx, 
                                double huber_thres = 0.01);
    
private:
    std::vector<FrameType> window_frames_;
    std::unordered_map<int, LandMark> map_;
    Parameter params_;
    std::shared_ptr<Caimura> cam0_;
    std::shared_ptr<Caimura> cam1_;
    ImuObs imu_;
    FrameType t_frame_;

    // key frame threshold
    double key_translation_thres_, key_rotation_thres_;
    int key_landmark_num_;
    int keyframe_num_;

    // triangulation threshold
    double landmark_move_thres_;
    int landmark_obs_num_;
    double landmark_huber_thres_;
    int landmark_iter_num_;
    double landmark_cost_thres_;

    // last frame pose
    Eigen::Matrix<double, 3, 4> Tiw_last_;
    int tracked_num_;
    FrameType last_frame_;
};
} // namespace inslam

#endif // SLIDINGWINDOW_H